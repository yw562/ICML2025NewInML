import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import vectorbt as vbt
from tqdm import tqdm
from markdown2 import markdown

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def generate_daily_signals(
    tweet_pred_csv: str | Path,
    date_col: str = "date",
    ticker_col: str = "ticker",
    p_pos_col: str = "p_pos",
    p_neg_col: str = "p_neg",
    method: str = "net",
) -> pd.DataFrame:
    df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])

    # 构建信号列
    if method == "net":
        df["signal"] = df[p_pos_col] - df[p_neg_col]
    elif method == "proba":
        df["signal"] = df[p_pos_col]
    else:
        raise ValueError("method must be 'net' or 'proba'")

    # 构建日期-股票宽表
    daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
    daily.sort_index(inplace=True)

    # 向后移动1天，防止未来信息穿越
    daily = daily.shift(1)

    return daily

def make_long_short_weights(
    signal_wide: pd.DataFrame,
    top_n: int = 50,
    weight_scheme: str = "equal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

def format_number(x, pct=False, digits=2):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
    else:
        return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"

def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
    if isinstance(returns, pd.DataFrame):
        assert returns.shape[1] == 1, "returns DataFrame must have only one column"
        returns = returns.squeeze()
    assert isinstance(returns, pd.Series), "returns must be a pandas Series"
    assert returns.ndim == 1, "returns must be one-dimensional"
    assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
    t_stat = float(t_stat_val)
    p_value = float(p_value_val)

    nav = (1 + returns).cumprod()
    dd = 1 - nav / nav.cummax()
    max_dd = dd.max()

    return {
        "ann_ret": format_number(ann_ret, pct=True, digits=4),
        "ann_vol": format_number(ann_vol, digits=2),
        "sharpe": format_number(sharpe, digits=2),
        "max_dd": format_number(max_dd, pct=True, digits=2),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 4),
    }

def signal_quantile_returns(signals: pd.DataFrame, prices: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    quantiles = signals.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop'), axis=1)
    forward_returns = prices.pct_change(fill_method=None).shift(-1)
    binned_returns = {
        f"Q{i+1}": forward_returns.where(quantiles == i).mean(axis=1)
        for i in range(n_bins)
    }
    return pd.DataFrame(binned_returns)

def run_vectorbt(
    signals_path: str | Path,
    prices_path: str | Path,
    top_n: int = 50,
    cost_bps: float = 10,
    weight_scheme: str = "equal",
    outdir: str | Path = "results_vbt",
    benchmark_ticker: str = "SPY",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = pd.read_parquet(signals_path)
    prices = pd.read_parquet(prices_path)

    common_cols = signals.columns.intersection(prices.columns)
    signals, prices = signals[common_cols], prices[common_cols]
    prices = prices.loc[signals.index]

    w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
    long_entries = w_long > 0
    long_exits = long_entries.shift(-1).fillna(False)
    short_entries = w_short < 0
    short_exits = short_entries.shift(-1).fillna(False)

    long_size = w_long.clip(lower=0)
    short_size = (-w_short).clip(lower=0)
    fees = cost_bps / 10000

    long_pf = vbt.Portfolio.from_signals(close=prices, entries=long_entries, exits=long_exits, size=long_size, fees=fees, freq="D", init_cash=0.5)
    short_pf = vbt.Portfolio.from_signals(close=prices, entries=short_entries, exits=short_exits, size=short_size, fees=fees, freq="D", init_cash=0.5)

    combined_nav = long_pf.value() + short_pf.value()
    combined_nav.name = "NAV"
    returns = combined_nav.pct_change().dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)

    met = perf_metrics(returns)
    pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    combined_nav.plot(label="Strategy NAV", title="Combined Long-Short Portfolio NAV")
    if benchmark_ticker in prices.columns:
        benchmark_nav = prices[benchmark_ticker] / prices[benchmark_ticker].iloc[0]
        benchmark_nav.plot(label=f"Benchmark ({benchmark_ticker})")
    plt.legend()
    plt.ylabel("Net Asset Value")
    plt.tight_layout()
    plt.savefig(outdir / "01_net_value.png")
    plt.close()

    drawdown = 1 - combined_nav / combined_nav.cummax()
    plt.figure(figsize=(10, 5))
    drawdown.plot(title="Drawdown")
    plt.tight_layout()
    plt.savefig(outdir / "02_drawdown.png")
    plt.close()

    latest_sig = signals.iloc[-1]
    buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=buckets, y=latest_sig)
    plt.title("Signal Distribution (Last Day)")
    plt.tight_layout()
    plt.savefig(outdir / "03_signal_distribution.png")
    plt.close()

    quantile_ret = signal_quantile_returns(signals, prices)
    quantile_ret.cumsum().plot(title="Cumulative Returns by Signal Quantile", figsize=(10, 5))
    plt.tight_layout()
    plt.savefig(outdir / "04_quantile_returns.png")
    plt.close()

    with open(outdir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Backtest Summary\n\n")
        f.write("## Key Metrics\n")
        for k, v in met.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Charts\n")
        for fig in ["01_net_value", "02_drawdown", "03_signal_distribution", "04_quantile_returns"]:
            f.write(f"![{fig}](./{fig}.png)\n")

    html_content = markdown((outdir / "summary.md").read_text(encoding="utf-8"))
    with open(outdir / "summary.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    logging.info("VectorBT run complete → %s", outdir)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment back-test runner")
    parser.add_argument("--signals", default="signals.parquet")
    parser.add_argument("--prices", default="prices.parquet")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--cost_bps", type=float, default=10)
    parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
    parser.add_argument("--outdir", default="results_vbt")
    parser.add_argument("--benchmark_ticker", default="SPY")
    return parser

def run_grid(
    signals_path: str | Path,
    prices_path: str | Path,
    topn_list: list[int],
    outdir_root: str = "grid_results",
    **kwargs
):
    outdir_root = Path(outdir_root)
    outdir_root.mkdir(parents=True, exist_ok=True)
    metrics = []

    for top_n in topn_list:
        run_dir = outdir_root / f"topn_{top_n}"
        run_vectorbt(
            signals_path=signals_path,
            prices_path=prices_path,
            top_n=top_n,
            outdir=run_dir,
            **kwargs
        )
        met = pd.read_csv(run_dir / "metrics.csv")
        met.insert(0, "top_n", top_n)
        metrics.append(met)

    summary_df = pd.concat(metrics, ignore_index=True)
    summary_df.to_csv(outdir_root / "summary_metrics.csv", index=False)

    html = "<html><head><title>Grid Backtest Report</title></head><body>"
    html += "<h1>Grid Backtest Summary</h1>"
    html += summary_df.to_html(index=False)
    html += "<h2>Individual Reports</h2><ul>"
    for top_n in topn_list:
        html += f"<li><a href='topn_{top_n}/summary.html'>top_n = {top_n}</a></li>"
    html += "</ul></body></html>"
    with open(outdir_root / "grid_summary.html", "w", encoding="utf-8") as f:
        f.write(html)

def main():
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    run_grid(
        signals_path=args.signals,
        prices_path=args.prices,
        topn_list=[10, 30, 50],
        cost_bps=args.cost_bps,
        weight_scheme=args.weight_scheme,
        benchmark_ticker=args.benchmark_ticker,
        outdir_root="grid_results"
    )

if __name__ == "__main__":
    import sys
    sys.argv = ["", "--signals", "signals.parquet", "--prices", "prices.parquet"]
    main()

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import vectorbt as vbt
# from tqdm import tqdm
# from markdown2 import markdown

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )

# def generate_daily_signals(
#     tweet_pred_csv: str | Path,
#     date_col: str = "date",
#     ticker_col: str = "ticker",
#     p_pos_col: str = "p_pos",
#     p_neg_col: str = "p_neg",
#     method: str = "net",
# ) -> pd.DataFrame:
#     df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])

#     # 构建信号列
#     if method == "net":
#         df["signal"] = df[p_pos_col] - df[p_neg_col]
#     elif method == "proba":
#         df["signal"] = df[p_pos_col]
#     else:
#         raise ValueError("method must be 'net' or 'proba'")

#     # 构建日期-股票宽表
#     daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
#     daily.sort_index(inplace=True)

#     # 向后移动1天，防止未来信息穿越
#     daily = daily.shift(1)

#     return daily

# def make_long_short_weights(
#     signal_wide: pd.DataFrame,
#     top_n: int = 50,
#     weight_scheme: str = "equal",
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     w_long = signal_wide.copy(deep=False) * np.nan
#     w_short = w_long.copy(deep=False)

#     for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
#         s = row.dropna()
#         if s.empty:
#             continue
#         top = s.nlargest(top_n)
#         bottom = s.nsmallest(top_n)

#         if weight_scheme == "equal":
#             w_long.loc[dt, top.index] = 1.0 / len(top)
#             w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
#         else:
#             pos_w = top.abs() / top.abs().sum()
#             neg_w = bottom.abs() / bottom.abs().sum()
#             w_long.loc[dt, pos_w.index] = pos_w
#             w_short.loc[dt, neg_w.index] = -neg_w

#     return w_long.fillna(0.0), w_short.fillna(0.0)

# def format_number(x, pct=False, digits=2):
#     if isinstance(x, (pd.Series, pd.DataFrame)):
#         return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
#     else:
#         return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"

# def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
#     if isinstance(returns, pd.DataFrame):
#         assert returns.shape[1] == 1, "returns DataFrame must have only one column"
#         returns = returns.squeeze()
#     assert isinstance(returns, pd.Series), "returns must be a pandas Series"
#     assert returns.ndim == 1, "returns must be one-dimensional"
#     assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

#     ann_ret = returns.mean() * periods_per_year
#     ann_vol = returns.std() * np.sqrt(periods_per_year)

#     sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
#     t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
#     t_stat = float(t_stat_val)
#     p_value = float(p_value_val)

#     nav = (1 + returns).cumprod()
#     dd = 1 - nav / nav.cummax()
#     max_dd = dd.max()

#     return {
#         "ann_ret": format_number(ann_ret, pct=True, digits=4),
#         "ann_vol": format_number(ann_vol, digits=2),
#         "sharpe": format_number(sharpe, digits=2),
#         "max_dd": format_number(max_dd, pct=True, digits=2),
#         "t_stat": round(t_stat, 2),
#         "p_value": round(p_value, 4),
#     }

# def signal_quantile_returns(signals: pd.DataFrame, prices: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
#     quantiles = signals.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop'), axis=1)
#     forward_returns = prices.pct_change(fill_method=None).shift(-1)
#     binned_returns = {
#         f"Q{i+1}": forward_returns.where(quantiles == i).mean(axis=1)
#         for i in range(n_bins)
#     }
#     return pd.DataFrame(binned_returns)

# def run_vectorbt(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     top_n: int = 50,
#     cost_bps: float = 10,
#     weight_scheme: str = "equal",
#     outdir: str | Path = "results_vbt",
#     benchmark_ticker: str = "SPY",
# ):
#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     signals = pd.read_parquet(signals_path)
#     prices = pd.read_parquet(prices_path)

#     common_cols = signals.columns.intersection(prices.columns)
#     signals, prices = signals[common_cols], prices[common_cols]
#     prices = prices.loc[signals.index]

#     w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
#     long_entries = w_long > 0
#     long_exits = long_entries.shift(-1).fillna(False)
#     short_entries = w_short < 0
#     short_exits = short_entries.shift(-1).fillna(False)

#     long_size = w_long.clip(lower=0)
#     short_size = (-w_short).clip(lower=0)
#     fees = cost_bps / 10000

#     long_pf = vbt.Portfolio.from_signals(close=prices, entries=long_entries, exits=long_exits, size=long_size, fees=fees, freq="D", init_cash=0.5)
#     short_pf = vbt.Portfolio.from_signals(close=prices, entries=short_entries, exits=short_exits, size=short_size, fees=fees, freq="D", init_cash=0.5)

#     combined_nav = long_pf.value() + short_pf.value()
#     combined_nav.name = "NAV"
#     returns = combined_nav.pct_change().dropna()
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.mean(axis=1)

#     met = perf_metrics(returns)
#     pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

#     plt.figure(figsize=(10, 5))
#     combined_nav.plot(label="Strategy NAV", title="Combined Long-Short Portfolio NAV")
#     if benchmark_ticker in prices.columns:
#         benchmark_nav = prices[benchmark_ticker] / prices[benchmark_ticker].iloc[0]
#         benchmark_nav.plot(label=f"Benchmark ({benchmark_ticker})")
#     plt.legend()
#     plt.ylabel("Net Asset Value")
#     plt.tight_layout()
#     plt.savefig(outdir / "01_net_value.png")
#     plt.close()

#     drawdown = 1 - combined_nav / combined_nav.cummax()
#     plt.figure(figsize=(10, 5))
#     drawdown.plot(title="Drawdown")
#     plt.tight_layout()
#     plt.savefig(outdir / "02_drawdown.png")
#     plt.close()

#     latest_sig = signals.iloc[-1]
#     buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=buckets, y=latest_sig)
#     plt.title("Signal Distribution (Last Day)")
#     plt.tight_layout()
#     plt.savefig(outdir / "03_signal_distribution.png")
#     plt.close()

#     quantile_ret = signal_quantile_returns(signals, prices)
#     quantile_ret.cumsum().plot(title="Cumulative Returns by Signal Quantile", figsize=(10, 5))
#     plt.tight_layout()
#     plt.savefig(outdir / "04_quantile_returns.png")
#     plt.close()

#     with open(outdir / "summary.md", "w", encoding="utf-8") as f:
#         f.write("# Backtest Summary\n\n")
#         f.write("## Key Metrics\n")
#         for k, v in met.items():
#             f.write(f"- **{k}**: {v}\n")
#         f.write("\n## Charts\n")
#         for fig in ["01_net_value", "02_drawdown", "03_signal_distribution", "04_quantile_returns"]:
#             f.write(f"![{fig}](./{fig}.png)\n")

#     html_content = markdown((outdir / "summary.md").read_text(encoding="utf-8"))
#     with open(outdir / "summary.html", "w", encoding="utf-8") as html_file:
#         html_file.write(html_content)

#     logging.info("VectorBT run complete → %s", outdir)

# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Sentiment back-test runner")
#     parser.add_argument("--signals", default="signals.parquet")
#     parser.add_argument("--prices", default="prices.parquet")
#     parser.add_argument("--top_n", type=int, default=50)
#     parser.add_argument("--cost_bps", type=float, default=10)
#     parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
#     parser.add_argument("--outdir", default="results_vbt")
#     parser.add_argument("--benchmark_ticker", default="SPY")
#     return parser

# def run_grid(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     topn_list: list[int],
#     outdir_root: str = "grid_results",
#     **kwargs
# ):
#     outdir_root = Path(outdir_root)
#     outdir_root.mkdir(parents=True, exist_ok=True)
#     metrics = []

#     for top_n in topn_list:
#         run_dir = outdir_root / f"topn_{top_n}"
#         run_vectorbt(
#             signals_path=signals_path,
#             prices_path=prices_path,
#             top_n=top_n,
#             outdir=run_dir,
#             **kwargs
#         )
#         met = pd.read_csv(run_dir / "metrics.csv")
#         met.insert(0, "top_n", top_n)
#         metrics.append(met)

#     summary_df = pd.concat(metrics, ignore_index=True)
#     summary_df.to_csv(outdir_root / "summary_metrics.csv", index=False)

#     html = "<html><head><title>Grid Backtest Report</title></head><body>"
#     html += "<h1>Grid Backtest Summary</h1>"
#     html += summary_df.to_html(index=False)
#     html += "<h2>Individual Reports</h2><ul>"
#     for top_n in topn_list:
#         html += f"<li><a href='topn_{top_n}/summary.html'>top_n = {top_n}</a></li>"
#     html += "</ul></body></html>"
#     with open(outdir_root / "grid_summary.html", "w", encoding="utf-8") as f:
#         f.write(html)

# def main():
#     setup_logging()
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     run_grid(
#         signals_path=args.signals,
#         prices_path=args.prices,
#         topn_list=[10, 30, 50],
#         cost_bps=args.cost_bps,
#         weight_scheme=args.weight_scheme,
#         benchmark_ticker=args.benchmark_ticker,
#         outdir_root="grid_results"
#     )

# if __name__ == "__main__":
#     import sys
#     sys.argv = ["", "--signals", "signals.parquet", "--prices", "prices.parquet"]
#     main()

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import vectorbt as vbt
# from tqdm import tqdm
# from markdown2 import markdown

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )

# def generate_daily_signals(
#     tweet_pred_csv: str | Path,
#     date_col: str = "date",
#     ticker_col: str = "ticker",
#     p_pos_col: str = "p_pos",
#     p_neg_col: str = "p_neg",
#     method: str = "net",
# ) -> pd.DataFrame:
#     df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])

#     # 构建信号列
#     if method == "net":
#         df["signal"] = df[p_pos_col] - df[p_neg_col]
#     elif method == "proba":
#         df["signal"] = df[p_pos_col]
#     else:
#         raise ValueError("method must be 'net' or 'proba'")

#     # 构建日期-股票宽表
#     daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
#     daily.sort_index(inplace=True)

#     # 向后移动1天，防止未来信息穿越
#     daily = daily.shift(1)

#     return daily

# def make_long_short_weights(
#     signal_wide: pd.DataFrame,
#     top_n: int = 50,
#     weight_scheme: str = "equal",
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     w_long = signal_wide.copy(deep=False) * np.nan
#     w_short = w_long.copy(deep=False)

#     for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
#         s = row.dropna()
#         if s.empty:
#             continue
#         top = s.nlargest(top_n)
#         bottom = s.nsmallest(top_n)

#         if weight_scheme == "equal":
#             w_long.loc[dt, top.index] = 1.0 / len(top)
#             w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
#         else:
#             pos_w = top.abs() / top.abs().sum()
#             neg_w = bottom.abs() / bottom.abs().sum()
#             w_long.loc[dt, pos_w.index] = pos_w
#             w_short.loc[dt, neg_w.index] = -neg_w

#     return w_long.fillna(0.0), w_short.fillna(0.0)

# def format_number(x, pct=False, digits=2):
#     if isinstance(x, (pd.Series, pd.DataFrame)):
#         return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
#     else:
#         return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"

# def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
#     if isinstance(returns, pd.DataFrame):
#         assert returns.shape[1] == 1, "returns DataFrame must have only one column"
#         returns = returns.squeeze()
#     assert isinstance(returns, pd.Series), "returns must be a pandas Series"
#     assert returns.ndim == 1, "returns must be one-dimensional"
#     assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

#     ann_ret = returns.mean() * periods_per_year
#     ann_vol = returns.std() * np.sqrt(periods_per_year)

#     sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
#     t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
#     t_stat = float(t_stat_val)
#     p_value = float(p_value_val)

#     nav = (1 + returns).cumprod()
#     dd = 1 - nav / nav.cummax()
#     max_dd = dd.max()

#     return {
#         "ann_ret": format_number(ann_ret, pct=True, digits=4),
#         "ann_vol": format_number(ann_vol, digits=2),
#         "sharpe": format_number(sharpe, digits=2),
#         "max_dd": format_number(max_dd, pct=True, digits=2),
#         "t_stat": round(t_stat, 2),
#         "p_value": round(p_value, 4),
#     }

# def signal_quantile_returns(signals: pd.DataFrame, prices: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
#     quantiles = signals.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop'), axis=1)
#     forward_returns = prices.pct_change(fill_method=None).shift(-1)
#     binned_returns = {
#         f"Q{i+1}": forward_returns.where(quantiles == i).mean(axis=1)
#         for i in range(n_bins)
#     }
#     return pd.DataFrame(binned_returns)

# def run_vectorbt(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     top_n: int = 50,
#     cost_bps: float = 10,
#     weight_scheme: str = "equal",
#     outdir: str | Path = "results_vbt",
#     benchmark_ticker: str = "SPY",
# ):
#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     signals = pd.read_parquet(signals_path)
#     prices = pd.read_parquet(prices_path)

#     common_cols = signals.columns.intersection(prices.columns)
#     signals, prices = signals[common_cols], prices[common_cols]
#     prices = prices.loc[signals.index]

#     w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
#     long_entries = w_long > 0
#     long_exits = long_entries.shift(-1).fillna(False)
#     short_entries = w_short < 0
#     short_exits = short_entries.shift(-1).fillna(False)

#     long_size = w_long.clip(lower=0)
#     short_size = (-w_short).clip(lower=0)
#     fees = cost_bps / 10000

#     long_pf = vbt.Portfolio.from_signals(close=prices, entries=long_entries, exits=long_exits, size=long_size, fees=fees, freq="D", init_cash=0.5)
#     short_pf = vbt.Portfolio.from_signals(close=prices, entries=short_entries, exits=short_exits, size=short_size, fees=fees, freq="D", init_cash=0.5)

#     combined_nav = long_pf.value() + short_pf.value()
#     combined_nav.name = "NAV"
#     returns = combined_nav.pct_change().dropna()
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.mean(axis=1)

#     met = perf_metrics(returns)
#     pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

#     plt.figure(figsize=(10, 5))
#     combined_nav.plot(label="Strategy NAV", title="Combined Long-Short Portfolio NAV")
#     if benchmark_ticker in prices.columns:
#         benchmark_nav = prices[benchmark_ticker] / prices[benchmark_ticker].iloc[0]
#         benchmark_nav.plot(label=f"Benchmark ({benchmark_ticker})")
#     plt.legend()
#     plt.ylabel("Net Asset Value")
#     plt.tight_layout()
#     plt.savefig(outdir / "01_net_value.png")
#     plt.close()

#     drawdown = 1 - combined_nav / combined_nav.cummax()
#     plt.figure(figsize=(10, 5))
#     drawdown.plot(title="Drawdown")
#     plt.tight_layout()
#     plt.savefig(outdir / "02_drawdown.png")
#     plt.close()

#     latest_sig = signals.iloc[-1]
#     buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=buckets, y=latest_sig)
#     plt.title("Signal Distribution (Last Day)")
#     plt.tight_layout()
#     plt.savefig(outdir / "03_signal_distribution.png")
#     plt.close()

#     quantile_ret = signal_quantile_returns(signals, prices)
#     quantile_ret.cumsum().plot(title="Cumulative Returns by Signal Quantile", figsize=(10, 5))
#     plt.tight_layout()
#     plt.savefig(outdir / "04_quantile_returns.png")
#     plt.close()

#     with open(outdir / "summary.md", "w", encoding="utf-8") as f:
#         f.write("# Backtest Summary\n\n")
#         f.write("## Key Metrics\n")
#         for k, v in met.items():
#             f.write(f"- **{k}**: {v}\n")
#         f.write("\n## Charts\n")
#         for fig in ["01_net_value", "02_drawdown", "03_signal_distribution", "04_quantile_returns"]:
#             f.write(f"![{fig}](./{fig}.png)\n")

#     html_content = markdown((outdir / "summary.md").read_text(encoding="utf-8"))
#     with open(outdir / "summary.html", "w", encoding="utf-8") as html_file:
#         html_file.write(html_content)

#     logging.info("VectorBT run complete → %s", outdir)

# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Sentiment back-test runner")
#     parser.add_argument("--signals", default="signals.parquet")
#     parser.add_argument("--prices", default="prices.parquet")
#     parser.add_argument("--top_n", type=int, default=50)
#     parser.add_argument("--cost_bps", type=float, default=10)
#     parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
#     parser.add_argument("--outdir", default="results_vbt")
#     parser.add_argument("--benchmark_ticker", default="SPY")
#     return parser

# def run_grid(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     topn_list: list[int],
#     outdir_root: str = "grid_results",
#     **kwargs
# ):
#     outdir_root = Path(outdir_root)
#     outdir_root.mkdir(parents=True, exist_ok=True)
#     metrics = []

#     for top_n in topn_list:
#         run_dir = outdir_root / f"topn_{top_n}"
#         run_vectorbt(
#             signals_path=signals_path,
#             prices_path=prices_path,
#             top_n=top_n,
#             outdir=run_dir,
#             **kwargs
#         )
#         met = pd.read_csv(run_dir / "metrics.csv")
#         met.insert(0, "top_n", top_n)
#         metrics.append(met)

#     summary_df = pd.concat(metrics, ignore_index=True)
#     summary_df.to_csv(outdir_root / "summary_metrics.csv", index=False)

#     html = "<html><head><title>Grid Backtest Report</title></head><body>"
#     html += "<h1>Grid Backtest Summary</h1>"
#     html += summary_df.to_html(index=False)
#     html += "<h2>Individual Reports</h2><ul>"
#     for top_n in topn_list:
#         html += f"<li><a href='topn_{top_n}/summary.html'>top_n = {top_n}</a></li>"
#     html += "</ul></body></html>"
#     with open(outdir_root / "grid_summary.html", "w", encoding="utf-8") as f:
#         f.write(html)

# def main():
#     setup_logging()
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     run_grid(
#         signals_path=args.signals,
#         prices_path=args.prices,
#         topn_list=[10, 30, 50],
#         cost_bps=args.cost_bps,
#         weight_scheme=args.weight_scheme,
#         benchmark_ticker=args.benchmark_ticker,
#         outdir_root="grid_results"
#     )

# if __name__ == "__main__":
#     import sys
#     sys.argv = ["", "--signals", "signals.parquet", "--prices", "prices.parquet"]
#     main()


# 以下是已经跑通了的，top103050代码，上面是进一步优化的，需要去做剔除稀疏股票，shift+1防止泄露

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import vectorbt as vbt
# from tqdm import tqdm
# from markdown2 import markdown


# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )


# def generate_daily_signals(
#     tweet_pred_csv: str | Path,
#     date_col: str = "date",
#     ticker_col: str = "ticker",
#     p_pos_col: str = "p_pos",
#     p_neg_col: str = "p_neg",
#     method: str = "net",
# ) -> pd.DataFrame:
#     df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])
#     if method == "net":
#         df["signal"] = df[p_pos_col] - df[p_neg_col]
#     elif method == "proba":
#         df["signal"] = df[p_pos_col]
#     else:
#         raise ValueError("method must be 'net' or 'proba'")
#     daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
#     daily.sort_index(inplace=True)
#     return daily


# def make_long_short_weights(
#     signal_wide: pd.DataFrame,
#     top_n: int = 50,
#     weight_scheme: str = "equal",
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     w_long = signal_wide.copy(deep=False) * np.nan
#     w_short = w_long.copy(deep=False)

#     for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
#         s = row.dropna()
#         if s.empty:
#             continue
#         top = s.nlargest(top_n)
#         bottom = s.nsmallest(top_n)

#         if weight_scheme == "equal":
#             w_long.loc[dt, top.index] = 1.0 / len(top)
#             w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
#         else:
#             pos_w = top.abs() / top.abs().sum()
#             neg_w = bottom.abs() / bottom.abs().sum()
#             w_long.loc[dt, pos_w.index] = pos_w
#             w_short.loc[dt, neg_w.index] = -neg_w

#     return w_long.fillna(0.0), w_short.fillna(0.0)


# def format_number(x, pct=False, digits=2):
#     if isinstance(x, (pd.Series, pd.DataFrame)):
#         return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
#     else:
#         return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


# def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
#     if isinstance(returns, pd.DataFrame):
#         assert returns.shape[1] == 1, "returns DataFrame must have only one column"
#         returns = returns.squeeze()
#     assert isinstance(returns, pd.Series), "returns must be a pandas Series"
#     assert returns.ndim == 1, "returns must be one-dimensional"
#     assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

#     ann_ret = returns.mean() * periods_per_year
#     ann_vol = returns.std() * np.sqrt(periods_per_year)

#     sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
#     t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
#     t_stat = float(t_stat_val)
#     p_value = float(p_value_val)

#     nav = (1 + returns).cumprod()
#     dd = 1 - nav / nav.cummax()
#     max_dd = dd.max()

#     return {
#         "ann_ret": format_number(ann_ret, pct=True, digits=4),
#         "ann_vol": format_number(ann_vol, digits=2),
#         "sharpe": format_number(sharpe, digits=2),
#         "max_dd": format_number(max_dd, pct=True, digits=2),
#         "t_stat": round(t_stat, 2),
#         "p_value": round(p_value, 4),
#     }


# def signal_quantile_returns(signals: pd.DataFrame, prices: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
#     quantiles = signals.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop'), axis=1)
#     forward_returns = prices.pct_change(fill_method=None).shift(-1)
#     binned_returns = {
#         f"Q{i+1}": forward_returns.where(quantiles == i).mean(axis=1)
#         for i in range(n_bins)
#     }
#     return pd.DataFrame(binned_returns)


# def run_vectorbt(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     top_n: int = 50,
#     cost_bps: float = 10,
#     weight_scheme: str = "equal",
#     outdir: str | Path = "results_vbt",
#     benchmark_ticker: str = "SPY",
# ):
#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     signals = pd.read_parquet(signals_path)
#     prices = pd.read_parquet(prices_path)

#     common_cols = signals.columns.intersection(prices.columns)
#     signals, prices = signals[common_cols], prices[common_cols]
#     prices = prices.loc[signals.index]

#     w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
#     long_entries = w_long > 0
#     long_exits = long_entries.shift(-1).fillna(False)
#     short_entries = w_short < 0
#     short_exits = short_entries.shift(-1).fillna(False)

#     long_size = w_long.clip(lower=0)
#     short_size = (-w_short).clip(lower=0)
#     fees = cost_bps / 10000

#     long_pf = vbt.Portfolio.from_signals(close=prices, entries=long_entries, exits=long_exits, size=long_size, fees=fees, freq="D", init_cash=0.5)
#     short_pf = vbt.Portfolio.from_signals(close=prices, entries=short_entries, exits=short_exits, size=short_size, fees=fees, freq="D", init_cash=0.5)

#     combined_nav = long_pf.value() + short_pf.value()
#     combined_nav.name = "NAV"
#     returns = combined_nav.pct_change().dropna()
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.mean(axis=1)

#     met = perf_metrics(returns)
#     pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

#     plt.figure(figsize=(10, 5))
#     combined_nav.plot(label="Strategy NAV", title="Combined Long-Short Portfolio NAV")
#     if benchmark_ticker in prices.columns:
#         benchmark_nav = prices[benchmark_ticker] / prices[benchmark_ticker].iloc[0]
#         benchmark_nav.plot(label=f"Benchmark ({benchmark_ticker})")
#     plt.legend()
#     plt.ylabel("Net Asset Value")
#     plt.tight_layout()
#     plt.savefig(outdir / "01_net_value.png")
#     plt.close()

#     drawdown = 1 - combined_nav / combined_nav.cummax()
#     plt.figure(figsize=(10, 5))
#     drawdown.plot(title="Drawdown")
#     plt.tight_layout()
#     plt.savefig(outdir / "02_drawdown.png")
#     plt.close()

#     latest_sig = signals.iloc[-1]
#     buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=buckets, y=latest_sig)
#     plt.title("Signal Distribution (Last Day)")
#     plt.tight_layout()
#     plt.savefig(outdir / "03_signal_distribution.png")
#     plt.close()

#     quantile_ret = signal_quantile_returns(signals, prices)
#     quantile_ret.cumsum().plot(title="Cumulative Returns by Signal Quantile", figsize=(10, 5))
#     plt.tight_layout()
#     plt.savefig(outdir / "04_quantile_returns.png")
#     plt.close()

#     with open(outdir / "summary.md", "w", encoding="utf-8") as f:
#         f.write("# Backtest Summary\n\n")
#         f.write("## Key Metrics\n")
#         for k, v in met.items():
#             f.write(f"- **{k}**: {v}\n")
#         f.write("\n## Charts\n")
#         for fig in ["01_net_value", "02_drawdown", "03_signal_distribution", "04_quantile_returns"]:
#             f.write(f"![{fig}](./{fig}.png)\n")

#     html_content = markdown((outdir / "summary.md").read_text(encoding="utf-8"))
#     with open(outdir / "summary.html", "w", encoding="utf-8") as html_file:
#         html_file.write(html_content)

#     logging.info("VectorBT run complete → %s", outdir)


# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Sentiment back-test runner")
#     parser.add_argument("--signals", default="signals.parquet")
#     parser.add_argument("--prices", default="prices.parquet")
#     parser.add_argument("--top_n", type=int, default=50)
#     parser.add_argument("--cost_bps", type=float, default=10)
#     parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
#     parser.add_argument("--outdir", default="results_vbt")
#     parser.add_argument("--benchmark_ticker", default="SPY")
#     return parser


# def run_grid(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     topn_list: list[int],
#     outdir_root: str = "grid_results",
#     **kwargs
# ):
#     outdir_root = Path(outdir_root)
#     outdir_root.mkdir(parents=True, exist_ok=True)
#     metrics = []

#     for top_n in topn_list:
#         run_dir = outdir_root / f"topn_{top_n}"
#         run_vectorbt(
#             signals_path=signals_path,
#             prices_path=prices_path,
#             top_n=top_n,
#             outdir=run_dir,
#             **kwargs
#         )
#         met = pd.read_csv(run_dir / "metrics.csv")
#         met.insert(0, "top_n", top_n)
#         metrics.append(met)

#     summary_df = pd.concat(metrics, ignore_index=True)
#     summary_df.to_csv(outdir_root / "summary_metrics.csv", index=False)

#     html = "<html><head><title>Grid Backtest Report</title></head><body>"
#     html += "<h1>Grid Backtest Summary</h1>"
#     html += summary_df.to_html(index=False)
#     html += "<h2>Individual Reports</h2><ul>"
#     for top_n in topn_list:
#         html += f"<li><a href='topn_{top_n}/summary.html'>top_n = {top_n}</a></li>"
#     html += "</ul></body></html>"
#     with open(outdir_root / "grid_summary.html", "w", encoding="utf-8") as f:
#         f.write(html)

# def main():
#     setup_logging()
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     # run_vectorbt(...)  # replaced with grid
#     run_grid(
#         signals_path=args.signals,
#         prices_path=args.prices,
#         topn_list=[10, 30, 50],
#         cost_bps=args.cost_bps,
#         weight_scheme=args.weight_scheme,
#         benchmark_ticker=args.benchmark_ticker,
#         outdir_root="grid_results"
#     )


# if __name__ == "__main__":
#     main()


# enhanced 2nd

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import vectorbt as vbt
# from tqdm import tqdm
# from markdown2 import markdown


# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )


# def generate_daily_signals(
#     tweet_pred_csv: str | Path,
#     date_col: str = "date",
#     ticker_col: str = "ticker",
#     p_pos_col: str = "p_pos",
#     p_neg_col: str = "p_neg",
#     method: str = "net",
# ) -> pd.DataFrame:
#     df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])
#     if method == "net":
#         df["signal"] = df[p_pos_col] - df[p_neg_col]
#     elif method == "proba":
#         df["signal"] = df[p_pos_col]
#     else:
#         raise ValueError("method must be 'net' or 'proba'")
#     daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
#     daily.sort_index(inplace=True)
#     return daily


# def make_long_short_weights(
#     signal_wide: pd.DataFrame,
#     top_n: int = 50,
#     weight_scheme: str = "equal",
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     w_long = signal_wide.copy(deep=False) * np.nan
#     w_short = w_long.copy(deep=False)

#     for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
#         s = row.dropna()
#         if s.empty:
#             continue
#         top = s.nlargest(top_n)
#         bottom = s.nsmallest(top_n)

#         if weight_scheme == "equal":
#             w_long.loc[dt, top.index] = 1.0 / len(top)
#             w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
#         else:
#             pos_w = top.abs() / top.abs().sum()
#             neg_w = bottom.abs() / bottom.abs().sum()
#             w_long.loc[dt, pos_w.index] = pos_w
#             w_short.loc[dt, neg_w.index] = -neg_w

#     return w_long.fillna(0.0), w_short.fillna(0.0)


# def format_number(x, pct=False, digits=2):
#     if isinstance(x, (pd.Series, pd.DataFrame)):
#         return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
#     else:
#         return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


# def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
#     if isinstance(returns, pd.DataFrame):
#         assert returns.shape[1] == 1, "returns DataFrame must have only one column"
#         returns = returns.squeeze()
#     assert isinstance(returns, pd.Series), "returns must be a pandas Series"
#     assert returns.ndim == 1, "returns must be one-dimensional"
#     assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

#     ann_ret = returns.mean() * periods_per_year
#     ann_vol = returns.std() * np.sqrt(periods_per_year)

#     sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
#     t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
#     t_stat = float(t_stat_val)
#     p_value = float(p_value_val)

#     nav = (1 + returns).cumprod()
#     dd = 1 - nav / nav.cummax()
#     max_dd = dd.max()

#     return {
#         "ann_ret": format_number(ann_ret, pct=True, digits=4),
#         "ann_vol": format_number(ann_vol, digits=2),
#         "sharpe": format_number(sharpe, digits=2),
#         "max_dd": format_number(max_dd, pct=True, digits=2),
#         "t_stat": round(t_stat, 2),
#         "p_value": round(p_value, 4),
#     }


# def signal_quantile_returns(signals: pd.DataFrame, prices: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
#     quantiles = signals.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop'), axis=1)
#     forward_returns = prices.pct_change(fill_method=None).shift(-1)
#     binned_returns = {
#         f"Q{i+1}": forward_returns.where(quantiles == i).mean(axis=1)
#         for i in range(n_bins)
#     }
#     return pd.DataFrame(binned_returns)


# def run_vectorbt(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     top_n: int = 50,
#     cost_bps: float = 10,
#     weight_scheme: str = "equal",
#     outdir: str | Path = "results_vbt",
#     benchmark_ticker: str = "SPY",
# ):
#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     signals = pd.read_parquet(signals_path)
#     prices = pd.read_parquet(prices_path)

#     common_cols = signals.columns.intersection(prices.columns)
#     signals, prices = signals[common_cols], prices[common_cols]
#     prices = prices.loc[signals.index]

#     w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)
#     long_entries = w_long > 0
#     long_exits = long_entries.shift(-1).fillna(False)
#     short_entries = w_short < 0
#     short_exits = short_entries.shift(-1).fillna(False)

#     long_size = w_long.clip(lower=0)
#     short_size = (-w_short).clip(lower=0)
#     fees = cost_bps / 10000

#     long_pf = vbt.Portfolio.from_signals(close=prices, entries=long_entries, exits=long_exits, size=long_size, fees=fees, freq="D", init_cash=0.5)
#     short_pf = vbt.Portfolio.from_signals(close=prices, entries=short_entries, exits=short_exits, size=short_size, fees=fees, freq="D", init_cash=0.5)

#     combined_nav = long_pf.value() + short_pf.value()
#     combined_nav.name = "NAV"
#     returns = combined_nav.pct_change().dropna()
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.mean(axis=1)

#     met = perf_metrics(returns)
#     pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

#     plt.figure(figsize=(10, 5))
#     combined_nav.plot(label="Strategy NAV", title="Combined Long-Short Portfolio NAV")
#     if benchmark_ticker in prices.columns:
#         benchmark_nav = prices[benchmark_ticker] / prices[benchmark_ticker].iloc[0]
#         benchmark_nav.plot(label=f"Benchmark ({benchmark_ticker})")
#     plt.legend()
#     plt.ylabel("Net Asset Value")
#     plt.tight_layout()
#     plt.savefig(outdir / "01_net_value.png")
#     plt.close()

#     drawdown = 1 - combined_nav / combined_nav.cummax()
#     plt.figure(figsize=(10, 5))
#     drawdown.plot(title="Drawdown")
#     plt.tight_layout()
#     plt.savefig(outdir / "02_drawdown.png")
#     plt.close()

#     latest_sig = signals.iloc[-1]
#     buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=buckets, y=latest_sig)
#     plt.title("Signal Distribution (Last Day)")
#     plt.tight_layout()
#     plt.savefig(outdir / "03_signal_distribution.png")
#     plt.close()

#     quantile_ret = signal_quantile_returns(signals, prices)
#     quantile_ret.cumsum().plot(title="Cumulative Returns by Signal Quantile", figsize=(10, 5))
#     plt.tight_layout()
#     plt.savefig(outdir / "04_quantile_returns.png")
#     plt.close()

#     with open(outdir / "summary.md", "w", encoding="utf-8") as f:
#         f.write("# Backtest Summary\n\n")
#         f.write("## Key Metrics\n")
#         for k, v in met.items():
#             f.write(f"- **{k}**: {v}\n")
#         f.write("\n## Charts\n")
#         for fig in ["01_net_value", "02_drawdown", "03_signal_distribution", "04_quantile_returns"]:
#             f.write(f"![{fig}](./{fig}.png)\n")

#     html_content = markdown((outdir / "summary.md").read_text(encoding="utf-8"))
#     with open(outdir / "summary.html", "w", encoding="utf-8") as html_file:
#         html_file.write(html_content)

#     logging.info("VectorBT run complete → %s", outdir)


# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Sentiment back-test runner")
#     parser.add_argument("--signals", default="signals.parquet")
#     parser.add_argument("--prices", default="prices.parquet")
#     parser.add_argument("--top_n", type=int, default=50)
#     parser.add_argument("--cost_bps", type=float, default=10)
#     parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
#     parser.add_argument("--outdir", default="results_vbt")
#     parser.add_argument("--benchmark_ticker", default="SPY")
#     return parser


# def main():
#     setup_logging()
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     run_vectorbt(
#         signals_path=args.signals,
#         prices_path=args.prices,
#         top_n=args.top_n,
#         cost_bps=args.cost_bps,
#         weight_scheme=args.weight_scheme,
#         outdir=args.outdir,
#         benchmark_ticker=args.benchmark_ticker,
#     )


# if __name__ == "__main__":
#     main()

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# import vectorbt as vbt
# from tqdm import tqdm


# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         datefmt="%H:%M:%S",
#     )


# def generate_daily_signals(
#     tweet_pred_csv: str | Path,
#     date_col: str = "date",
#     ticker_col: str = "ticker",
#     p_pos_col: str = "p_pos",
#     p_neg_col: str = "p_neg",
#     method: str = "net",
# ) -> pd.DataFrame:
#     df = pd.read_csv(tweet_pred_csv, parse_dates=[date_col])
#     if method == "net":
#         df["signal"] = df[p_pos_col] - df[p_neg_col]
#     elif method == "proba":
#         df["signal"] = df[p_pos_col]
#     else:
#         raise ValueError("method must be 'net' or 'proba'")
#     daily = df.groupby([date_col, ticker_col])["signal"].mean().unstack(fill_value=np.nan)
#     daily.sort_index(inplace=True)
#     return daily


# def make_long_short_weights(
#     signal_wide: pd.DataFrame,
#     top_n: int = 50,
#     weight_scheme: str = "equal",
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     w_long = signal_wide.copy(deep=False) * np.nan
#     w_short = w_long.copy(deep=False)

#     for dt, row in tqdm(signal_wide.iterrows(), total=len(signal_wide), desc="assign weights"):
#         s = row.dropna()
#         if s.empty:
#             continue
#         top = s.nlargest(top_n)
#         bottom = s.nsmallest(top_n)

#         if weight_scheme == "equal":
#             w_long.loc[dt, top.index] = 1.0 / len(top)
#             w_short.loc[dt, bottom.index] = -1.0 / len(bottom)
#         else:
#             pos_w = top.abs() / top.abs().sum()
#             neg_w = bottom.abs() / bottom.abs().sum()
#             w_long.loc[dt, pos_w.index] = pos_w
#             w_short.loc[dt, neg_w.index] = -neg_w

#     return w_long.fillna(0.0), w_short.fillna(0.0)


# def format_number(x, pct=False, digits=2):
#     if isinstance(x, (pd.Series, pd.DataFrame)):
#         return x.applymap(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}") if isinstance(x, pd.DataFrame) else x.apply(lambda v: f"{v:.{digits}%}" if pct else f"{v:.{digits}f}")
#     else:
#         return f"{x:.{digits}%}" if pct else f"{x:.{digits}f}"


# def perf_metrics(returns: pd.Series, rf=0.0, periods_per_year=252) -> dict:
#     if isinstance(returns, pd.DataFrame):
#         assert returns.shape[1] == 1, "returns DataFrame must have only one column"
#         returns = returns.squeeze()
#     assert isinstance(returns, pd.Series), "returns must be a pandas Series"
#     assert returns.ndim == 1, "returns must be one-dimensional"
#     assert returns.dropna().size > 1, "returns must contain more than one non-NaN value"

#     ann_ret = returns.mean() * periods_per_year
#     ann_vol = returns.std() * np.sqrt(periods_per_year)

#     if isinstance(ann_vol, pd.Series):
#         sharpe = (ann_ret - rf) / ann_vol
#         sharpe = sharpe.where(ann_vol > 0, np.nan)
#     else:
#         sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

#     t_stat_val, p_value_val = stats.ttest_1samp(returns.dropna().values, 0)
#     t_stat = float(t_stat_val)
#     p_value = float(p_value_val)

#     nav = (1 + returns).cumprod()
#     dd = 1 - nav / nav.cummax()
#     max_dd = dd.max()

#     return {
#         "ann_ret": format_number(ann_ret, pct=True, digits=4),
#         "ann_vol": format_number(ann_vol, digits=2),
#         "sharpe": format_number(sharpe, digits=2),
#         "max_dd": format_number(max_dd, pct=True, digits=2),
#         "t_stat": round(t_stat, 2),
#         "p_value": round(p_value, 4),
#     }


# def run_vectorbt(
#     signals_path: str | Path,
#     prices_path: str | Path,
#     top_n: int = 50,
#     cost_bps: float = 10,
#     weight_scheme: str = "equal",
#     outdir: str | Path = "results_vbt",
# ):
#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     signals = pd.read_parquet(signals_path)
#     prices = pd.read_parquet(prices_path)

#     common_cols = signals.columns.intersection(prices.columns)
#     signals, prices = signals[common_cols], prices[common_cols]
#     prices = prices.loc[signals.index]

#     w_long, w_short = make_long_short_weights(signals, top_n, weight_scheme)

#     long_entries = w_long > 0
#     long_exits = long_entries.shift(-1).fillna(False)
#     short_entries = w_short < 0
#     short_exits = short_entries.shift(-1).fillna(False)

#     long_size = w_long.clip(lower=0)
#     short_size = (-w_short).clip(lower=0)

#     fees = cost_bps / 10000

#     long_pf = vbt.Portfolio.from_signals(
#         close=prices,
#         entries=long_entries,
#         exits=long_exits,
#         size=long_size,
#         fees=fees,
#         freq="D",
#         init_cash=0.5,
#     )

#     short_pf = vbt.Portfolio.from_signals(
#         close=prices,
#         entries=short_entries,
#         exits=short_exits,
#         size=short_size,
#         fees=fees,
#         freq="D",
#         init_cash=0.5,
#     )

#     combined_nav = long_pf.value() + short_pf.value()
#     combined_nav.name = "NAV"
#     returns = combined_nav.pct_change().dropna()
#     if isinstance(returns, pd.DataFrame):
#         returns = returns.mean(axis=1)

#     met = perf_metrics(returns)
#     pd.DataFrame([met]).to_csv(outdir / "metrics.csv", index=False)

#     plt.figure(figsize=(10, 5))
#     combined_nav.plot(title="Combined Long-Short Portfolio NAV")
#     plt.ylabel("Net Asset Value")
#     plt.tight_layout()
#     plt.savefig(outdir / "01_net_value.png")
#     plt.close()

#     drawdown = 1 - combined_nav / combined_nav.cummax()
#     plt.figure(figsize=(10, 5))
#     drawdown.plot(title="Drawdown")
#     plt.tight_layout()
#     plt.savefig(outdir / "02_drawdown.png")
#     plt.close()

#     latest_sig = signals.iloc[-1]
#     buckets = pd.qcut(latest_sig, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=buckets, y=latest_sig)
#     plt.title("Signal Distribution (Last Day)")
#     plt.tight_layout()
#     plt.savefig(outdir / "03_signal_distribution.png")
#     plt.close()

#     logging.info("VectorBT run complete → %s", outdir)

# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Sentiment back-test runner")
#     parser.add_argument("--signals", default="signals.parquet")
#     parser.add_argument("--prices", default="prices.parquet")
#     parser.add_argument("--top_n", type=int, default=50)
#     parser.add_argument("--cost_bps", type=float, default=10)
#     parser.add_argument("--weight_scheme", choices=["equal", "abs"], default="equal")
#     parser.add_argument("--outdir", default="results_vbt")
#     return parser

# def main():
#     setup_logging()
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     run_vectorbt(
#         signals_path=args.signals,
#         prices_path=args.prices,
#         top_n=args.top_n,
#         cost_bps=args.cost_bps,
#         weight_scheme=args.weight_scheme,
#         outdir=args.outdir,
#     )


# if __name__ == "__main__":
#     main()

# # (This message ensures that robust validation has been added to perf_metrics.)
