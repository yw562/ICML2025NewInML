#!/usr/bin/env python
# day3_backtest.py
# Coding: UTF-8


import argparse
import logging
import math
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# Signal preparation
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
    daily.sort_index(inplace=True)
    return daily

# Portfolio weights
def make_long_short_weights(
    signal_wide: pd.DataFrame,
    top_n: int = 50,
    weight_scheme: str = "equal",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_long = signal_wide.copy(deep=False) * np.nan
    w_short = w_long.copy(deep=False)

    for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
        s = row.dropna()
        if s.empty:
            continue
        top = s.nlargest(top_n)
        bottom = s.nsmallest(top_n)

        if weight_scheme == "equal":
            w_long.loc[dt, top.index] = 1.0 / len(top)
            w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
        else:
            pos_w = top.abs() / top.abs().sum()
            neg_w = bottom.abs() / bottom.abs().sum()
            w_long.loc[dt, pos_w.index] = pos_w
            w_short.loc[dt, neg_w.index] = -neg_w

    return w_long.fillna(0.0), w_short.fillna(0.0)

# Metrics
def perf_metrics(returns, rf=0.0, periods_per_year=252):
    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    if isinstance(ann_vol, pd.Series):
        # element-wise check
        sharpe = (ann_ret - rf) / ann_vol
        sharpe = sharpe.where(ann_vol > 0, np.nan)
    else:
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        # add more metrics if needed
    }


# VectorBT backtest
import vectorbt as vbt

import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt

def run_vectorbt(
    signals_path: str | Path,
    prices_path: str | Path,
    top_n: int = 50,
    cost_bps: float = 10,
    weight_scheme: str = "equal",
    outdir: str | Path = "results_vbt",
):
    import seaborn as sns

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    signals = pd.read_parquet(signals_path)
    prices = pd.read_parquet(prices_path)

    common_cols = signals.columns.intersection(prices.columns)
    signals, prices = signals[common_cols], prices[common_cols]
    prices = prices.loc[signals.index]

    # Build weights
    w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
    weights = w_long + w_short

    # Prepare signals for vectorbt portfolios
    long_entries = w_long > 0
    long_exits = long_entries.shift(-1).fillna(False)
    short_entries = w_short < 0
    short_exits = short_entries.shift(-1).fillna(False)

    long_size = w_long.clip(lower=0)
    short_size = (-w_short).clip(lower=0)

    fees = cost_bps / 10000

    # Long portfolio
    long_pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=long_entries,
        exits=long_exits,
        size=long_size,
        fees=fees,
        freq="D",
        init_cash=0.5,
    )

    # Short portfolio
    short_pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=short_entries,
        exits=short_exits,
        size=short_size,
        fees=fees,
        freq="D",
        init_cash=0.5,
    )

    # Combine NAV (net asset value) of both portfolios — this is a pd.Series indexed by date
    combined_nav = long_pf.value() + short_pf.value()

    # Calculate returns and metrics from combined NAV
    returns = combined_nav.pct_change().dropna()
    met = perf_metrics(returns)

    pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

    # Plot NAV
    plt.figure(figsize=(10, 5))
    combined_nav.plot(title="Combined Long-Short Portfolio NAV")
    plt.savefig(outdir / "01_net_value.png", dpi=150)
    plt.close()

    # Plot drawdown
    plt.figure(figsize=(10, 5))
    drawdown = 1 - combined_nav / combined_nav.cummax()
    drawdown.plot(title="Drawdown")
    plt.savefig(outdir / "02_drawdown.png", dpi=150)
    plt.close()

    # Optional seaborn plot of last day signals
    if 'seaborn' in globals():
        latest_sig = signals.iloc[-1]
        buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=buckets, y=latest_sig)
        plt.title("Signal Distribution (Last Day)")
        plt.savefig(outdir / "03_signal_distribution.png", dpi=150)
        plt.close()

    logging.info("VectorBT run complete ⇒ %s", outdir)



# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment back-test runner")
    parser.add_argument("--signals", default="signals.parquet")
    parser.add_argument("--prices", default="prices.parquet")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--cost_bps", type=float, default=10)
    parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
    parser.add_argument("--outdir", default="results_vbt")
    return parser

def main():
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    run_vectorbt(
        signals_path=args.signals,
        prices_path=args.prices,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        weight_scheme=args.weight_scheme,
        outdir=args.outdir,
    )

if __name__ == "__main__":
    main()
