#!/usr/bin/env python
# day3_backtest.py
# Coding: UTF-8

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------------------------------------------------------
# 0. Signal Preparation
# ---------------------------------------------------------------------------

def generate_daily_signals(
    tweet_pred_csv: str | Path,
    date_col: str = "date",
    ticker_col: str = "ticker",
    p_pos_col: str = "p_pos",
    p_neg_col: str = "p_neg",
    method: str = "net",
) -> pd.DataFrame:
    """Aggregate tweet-level predictions into **daily stock sentiment signal**.

    Returns
    -------
    DataFrame indexed by [`date`, `ticker`] with single column `signal`.
    """
    df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])
    if method == "net":
        df["signal"] = df[p_pos_col] - df[p_neg_col]
    elif method == "proba":
        df["signal"] = df[p_pos_col]
    else:
        raise ValueError("method must be 'net' or 'proba'")

    daily = (
        df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
    )
    daily.sort_index(inplace=True)
    return daily                  # rows=date, cols=ticker (wide format)

# ---------------------------------------------------------------------------
# 1. Portfolio Construction helper
# ---------------------------------------------------------------------------

def make_long_short_weights(
    signal_wide: pd.DataFrame,
    top_n: int = 50,
    weight_scheme: str = "equal",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create **long & short weight matrices** aligned with `signal_wide` index."""
    w_long = signal_wide.copy(deep=False) * np.nan
    w_short = w_long.copy(deep=False)

    for dt, row in tqdm(signal_wide.iterrows(),
                        total=len(signal_wide),
                        desc="assign weights"):
        s = row.dropna()
        if s.empty:
            continue
        top = s.nlargest(top_n)
        bottom = s.nsmallest(top_n)

        if weight_scheme == "equal":
            w_long.loc[dt, top.index] = 1.0 / len(top)
            w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
        else:                      # value-weight by |signal|
            pos_w = top.abs() / top.abs().sum()
            neg_w = bottom.abs() / bottom.abs().sum()
            w_long.loc[dt, pos_w.index] = pos_w
            w_short.loc[dt, neg_w.index] = -neg_w

    return w_long.fillna(0.0), w_short.fillna(0.0)

# ---------------------------------------------------------------------------
# 2. Evaluation metrics
# ---------------------------------------------------------------------------

def perf_metrics(ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    ann_factor = 252
    ann_ret = (1 + ret).prod() ** (ann_factor / len(ret)) - 1
    ann_vol = ret.std(ddof=0) * math.sqrt(ann_factor)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((1 + ret).cumprod().cummax() - (1 + ret).cumprod()).max()
    t_stat, p_val = stats.ttest_1samp(ret, 0.0, nan_policy="omit")
    return dict(ann_ret=ann_ret,
                ann_vol=ann_vol,
                sharpe=sharpe,
                max_dd=max_dd,
                t_stat=t_stat,
                p_value=p_val)

# ---------------------------------------------------------------------------
# 3A. VectorBT Pipeline
# ---------------------------------------------------------------------------

def run_vectorbt(
    signals_path: str | Path,
    prices_path: str | Path,
    top_n: int = 50,
    cost_bps: float = 10,
    weight_scheme: str = "equal",
    outdir: str | Path = "results_vbt",
):
    import vectorbt as vbt           # lazy import → keep zipline env clean
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        seaborn_ok = True
    except ModuleNotFoundError:
        seaborn_ok = False

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ---------- #
    logging.info("Loading signals & prices …")
    signals = pd.read_parquet(signals_path)
    prices = pd.read_parquet(prices_path)

    # 对齐交集
    common_cols = signals.columns.intersection(prices.columns)
    signals, prices = signals[common_cols], prices[common_cols]
    prices = prices.loc[signals.index]          # 行对齐

    # ---------- Build weights ---------- #
    w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
    weights = w_long + w_short                 # 市值净 0

    # ---------- Back-test ---------- #
    logging.info("Running VectorBT portfolio …")
    entries = weights != 0
    exits = weights.shift(-1).fillna(0) == 0    # T+1 调仓

    # 拆分多空信号和仓位
    long_entries = weights > 0
    short_entries = weights < 0
    long_exits = long_entries.shift(-1).fillna(False)
    short_exits = short_entries.shift(-1).fillna(False)
    long_size = weights.clip(lower=0)
    short_size = (-weights).clip(lower=0)

    # 多头组合
    long_pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=long_entries,
        exits=long_exits,
        size=long_size,
        fees=cost_bps / 10000,
        freq="D",
        init_cash=0.5
    )

    # 空头组合
    short_pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=short_entries,
        exits=short_exits,
        size=short_size,
        fees=cost_bps / 10000,
        freq="D",
        init_cash=0.5
    )

    # 合并 NAV
    nav = (long_pf.value() + short_pf.value()).rename("nav")





    pf.trades.records.to_parquet(outdir / "trades.parquet")

    # ---------- Metrics ---------- #
    nav = pf.value().rename("nav")
    ret = nav.pct_change().dropna()
    met = perf_metrics(ret)
    pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

    # ---------- Plots ---------- #
    nav.plot(title="Cumulative Net Value", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(outdir / "01_net_value.png", dpi=150)
    plt.close()

    (1 - nav / nav.cummax()).plot(title="Drawdown", figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(outdir / "02_drawdown.png", dpi=150)
    plt.close()

    if seaborn_ok:
        latest_sig = signals.iloc[-1]
        buckets = pd.qcut(latest_sig, 5,
                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        sns.boxplot(x=buckets, y=latest_sig)
        plt.title("Signal Distribution (Last Day)")
        plt.tight_layout()
        plt.savefig(outdir / "03_long_short_box.png", dpi=150)
        plt.close()

    logging.info("VectorBT run complete ⇒ %s", outdir)

# ---------------------------------------------------------------------------
# 3B. Zipline Algorithm
# ---------------------------------------------------------------------------

def make_zipline_algorithm(
    signals_path: str | Path,
    top_n: int = 50,
    cost_bps: float = 10,
):
    from zipline.api import (
        order_target_percent, symbol, record,
        set_commission, set_slippage,
        commission, slippage
    )

    sig_df = pd.read_parquet(signals_path)      # daily wide
    tickers = sig_df.columns.tolist()

    def initialize(context):
        context.top_n = top_n
        context.signals = sig_df
        context.asset_map = {tkr: symbol(tkr) for tkr in tickers}
        set_commission(commission.PerDollar(cost_bps / 10000))
        set_slippage(slippage.FixedSlippage(spread=cost_bps / 10000))

    def handle_data(context, data):
        today = data.current_dt.normalize()
        if today not in context.signals.index:
            return
        row = context.signals.loc[today].dropna()
        if row.empty:
            return

        longs = row.nlargest(context.top_n).index
        shorts = row.nsmallest(context.top_n).index
        weight = 1.0 / (2 * context.top_n)      # 等权

        for tkr in longs:
            order_target_percent(context.asset_map[tkr], +weight)
        for tkr in shorts:
            order_target_percent(context.asset_map[tkr], -weight)
        for tkr in set(context.asset_map) - set(longs) - set(shorts):
            order_target_percent(context.asset_map[tkr], 0)

        record(longs=len(longs), shorts=len(shorts))

    return initialize, handle_data

def run_zipline(
    signals_path: str | Path,
    start: str,
    end: str,
    top_n: int = 50,
    cost_bps: float = 10,
    capital_base: float = 1e6,
    outdir: str | Path = "results_zip",
):
    from zipline import run_algorithm

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    init, handle = make_zipline_algorithm(signals_path, top_n, cost_bps)

    perf = run_algorithm(
        start=pd.Timestamp(start, tz="utc"),
        end=pd.Timestamp(end, tz="utc"),
        initialize=init,
        handle_data=handle,
        capital_base=capital_base,
        data_frequency="daily",
        bundle="quandl",            # 修改为自己的数据包
    )

    perf.to_csv(outdir / "perf.csv")
    ret = perf["portfolio_value"].pct_change().dropna()
    met = perf_metrics(ret)
    pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

    logging.info("Zipline run complete ⇒ %s", outdir)

# ---------------------------------------------------------------------------
# 4. CLI entry
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day-3 sentiment back-test runner"
    )
    subparsers = parser.add_subparsers(dest="mode")

    # ---------------- VectorBT ----------------
    p_vbt = subparsers.add_parser(
        "vectorbt", help="research-style fast back-test"
    )
    p_vbt.add_argument("--signals", default="signals.parquet",
                       help="日度信号文件 (parquet)")
    p_vbt.add_argument("--prices", default="prices.parquet",
                       help="收盘价矩阵 (parquet)")
    p_vbt.add_argument("--top_n", type=int, default=50,
                       help="多空两侧各取前 N 只股票")
    p_vbt.add_argument("--cost_bps", type=float, default=10,
                       help="单边交易成本，单位 bps")
    p_vbt.add_argument("--weight_scheme", choices=["equal", "abs"],
                       default="equal", help="权重方案")
    p_vbt.add_argument("--outdir", default="results_vbt",
                       help="结果输出目录")

    # ---------------- Zipline -----------------
    p_zip = subparsers.add_parser(
        "zipline", help="event-driven Zipline run"
    )
    p_zip.add_argument("--signals", default="signals.parquet",
                       help="日度信号文件 (parquet)")
    p_zip.add_argument("--start", default="2019-01-01",
                       help="回测开始日期 YYYY-MM-DD")
    p_zip.add_argument("--end", default="2020-12-31",
                       help="回测结束日期 YYYY-MM-DD")
    p_zip.add_argument("--top_n", type=int, default=50,
                       help="多空两侧各取前 N 只股票")
    p_zip.add_argument("--cost_bps", type=float, default=10,
                       help="单边交易成本，单位 bps")
    p_zip.add_argument("--capital_base", type=float, default=1e6,
                       help="初始资金")
    p_zip.add_argument("--outdir", default="results_zip",
                       help="结果输出目录")

    # ---------- 全局默认：vectorbt + 参数缺省 ----------
    parser.set_defaults(
        mode="vectorbt",
        signals="signals.parquet",
        prices="prices.parquet",
        top_n=50,
        cost_bps=10.0,
        weight_scheme="equal",
        outdir="results_vbt",
        start="2019-01-01",
        end="2020-12-31",
        capital_base=1e6,
    )
    return parser

def main():
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "vectorbt":
        run_vectorbt(
            signals_path=args.signals,
            prices_path=args.prices,
            top_n=args.top_n,
            cost_bps=args.cost_bps,
            weight_scheme=args.weight_scheme,
            outdir=args.outdir,
        )
    elif args.mode == "zipline":
        run_zipline(
            signals_path=args.signals,
            start=args.start,
            end=args.end,
            top_n=args.top_n,
            cost_bps=args.cost_bps,
            capital_base=args.capital_base,
            outdir=args.outdir,
        )
    else:          # 理论不会触发，防御
        parser.error("Unknown mode. Choose 'vectorbt' or 'zipline'.")

if __name__ == "__main__":
    main()
