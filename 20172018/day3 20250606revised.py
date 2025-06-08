#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# 🧼 清除可能污染 vectorbt 的名字
for polluted in ['value', 'rename']:
    if polluted in globals():
        print(f"⚠️ 清理污染变量: {polluted}")
        del globals()[polluted]

# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# -----------------------------------------------------------------------------
# Signal preparation
# -----------------------------------------------------------------------------

def generate_daily_signals(
    tweet_pred_csv: str | Path,
    date_col: str = "date",
    ticker_col: str = "ticker",
    p_pos_col: str = "p_pos",
    p_neg_col: str = "p_neg",
    method: str = "net",
) -> pd.DataFrame:
    df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])
    if method == "net":
        df["signal"] = df[p_pos_col] - df[p_neg_col]
    elif method == "proba":
        df["signal"] = df[p_pos_col]
    else:
        raise ValueError("method must be 'net' or 'proba'")
    daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
    return daily.sort_index()

# -----------------------------------------------------------------------------
# Weight assignment
# -----------------------------------------------------------------------------

def make_long_short_weights(
    signal_wide: pd.DataFrame,
    top_n: int = 50,
    weight_mode: str = "equal"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_long = signal_wide.copy(deep=False) * np.nan
    w_short = w_long.copy(deep=False)

    for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
        s = row.dropna()
        if s.empty:
            continue
        top = s.nlargest(top_n)
        bottom = s.nsmallest(top_n)

        if weight_mode == "equal":
            w_long.loc[dt, top.index] = 1.0 / len(top)
            w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
        else:
            pos_w = top.abs() / top.abs().sum()
            neg_w = bottom.abs() / bottom.abs().sum()
            w_long.loc[dt, pos_w.index] = pos_w
            w_short.loc[dt, neg_w.index] = -neg_w

    return w_long.fillna(0.0), w_short.fillna(0.0)

# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------

def perf_metrics(ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    ann_factor = 252
    ann_ret = (1 + ret).prod() ** (ann_factor / len(ret)) - 1
    ann_vol = ret.std(ddof=0) * math.sqrt(ann_factor)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((1 + ret).cumprod().cummax() - (1 + ret).cumprod()).max()
    t_stat, p_val = stats.ttest_1samp(ret, 0.0, nan_policy="omit")
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                max_dd=max_dd, t_stat=t_stat, p_value=p_val)

# -----------------------------------------------------------------------------
# VectorBT Backtest
# -----------------------------------------------------------------------------

def run_vectorbt(
    signals_path: str | Path,
    prices_path: str | Path,
    top_n: int = 50,
    cost_bps: float = 10,
    weight_scheme: str = "equal",
    outdir: str | Path = "results_vbt",
):
    import vectorbt as vbt
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        seaborn_ok = True
    except ImportError:
        seaborn_ok = False

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading signals & prices …")
    signals = pd.read_parquet(signals_path)
    prices = pd.read_parquet(prices_path)

    common = signals.columns.intersection(prices.columns)
    signals = signals[common]
    prices = prices[common].loc[signals.index]

    w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
    weights = w_long + w_short

    logging.info("Running VectorBT portfolio …")
    long_entries = weights > 0
    short_entries = weights < 0
    long_exits = long_entries.shift(-1).fillna(False)
    short_exits = short_entries.shift(-1).fillna(False)
    long_size = weights.clip(lower=0)
    short_size = (-weights).clip(lower=0)

    pf_long = vbt.Portfolio.from_signals(
        close=prices, entries=long_entries, exits=long_exits,
        size=long_size, fees=cost_bps / 10000, freq="D", init_cash=0.5
    )
    pf_short = vbt.Portfolio.from_signals(
        close=prices, entries=short_entries, exits=short_exits,
        size=short_size, fees=cost_bps / 10000, freq="D", init_cash=0.5
    )

    # ✅ 使用底层属性 _value 避免污染
    nav_series = (pf_long.asset_value().sum(axis=1) + pf_short.asset_value().sum(axis=1)).rename("nav")


    # 保存交易记录
    pf_long.trades.records.to_parquet(outdir / "long_trades.parquet")
    pf_short.trades.records.to_parquet(outdir / "short_trades.parquet")

    # 性能指标
    ret = nav_series.pct_change().dropna()
    metrics = perf_metrics(ret)
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)

    # 绘图
    nav_series.plot(title="Net Asset Value", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(outdir / "01_nav_curve.png", dpi=150)
    plt.close()

    (1 - nav_series / nav_series.cummax()).plot(title="Drawdown", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(outdir / "02_drawdown.png", dpi=150)
    plt.close()

    if seaborn_ok:
        latest_sig = signals.iloc[-1]
        buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        sns.boxplot(x=buckets, y=latest_sig)
        plt.title("Signal Distribution (Last Day)")
        plt.tight_layout()
        plt.savefig(outdir / "03_signal_boxplot.png", dpi=150)
        plt.close()

    logging.info("✅ VectorBT run complete ⇒ %s", outdir)

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Sentiment Backtest with VectorBT")
    parser.add_argument("--signals", default="signals.parquet", help="Path to signal file")
    parser.add_argument("--prices", default="prices.parquet", help="Path to price matrix")
    parser.add_argument("--top_n", type=int, default=50, help="Top-N longs/shorts")
    parser.add_argument("--cost_bps", type=float, default=10.0, help="Fees in bps")
    parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal", help="Weighting style")
    parser.add_argument("--outdir", default="results_vbt", help="Where to store results")
    return parser

def main():
    setup_logging()
    args = build_arg_parser().parse_args()
    run_vectorbt(
        signals_path=args.signals,
        prices_path=args.prices,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        weight_scheme=args.weight_scheme,
        outdir=args.outdir
    )

# 支持直接运行
if __name__ == "__main__":
    # Debug 模拟 CLI 参数（调试用）
    import sys
    sys.argv = [
        "day3.py",
        "--signals", "signals.parquet",
        "--prices", "prices.parquet",
        "--top_n", "30",
        "--cost_bps", "5"
    ]
    main()
