from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import read_table
from thesis_finllm.regressions import predictive_return_regression, write_regression_summary
from thesis_finllm.sentiment import (
    NoveltyFilterConfig,
    build_return_labels,
    combine_text_fields,
    daily_sentiment_signal,
    novelty_filter,
    score_news_with_lexicon,
)
from thesis_finllm.trading import run_long_short_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM/news-sentiment trading pipeline.")
    parser.add_argument("--config", default="configs/sentiment_trading.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    cols = cfg["columns"]
    out_dir = ensure_dir(paths["output_dir"])

    news = read_table(paths["news"], parse_dates=[cols["timestamp"]])
    returns = read_table(paths["returns"], parse_dates=[cols["date"]])

    news = combine_text_fields(news, cols["text_fields"], output_col="text")
    if cols["sentiment_score"] not in news.columns:
        news = score_news_with_lexicon(news, text_col="text", output_col=cols["sentiment_score"])

    nf = cfg.get("novelty_filter", {})
    if nf.get("enabled", True):
        news = novelty_filter(
            news,
            NoveltyFilterConfig(
                similarity_threshold=float(nf.get("similarity_threshold", 0.80)),
                lookback_days=int(nf.get("lookback_days", 5)),
                text_col="text",
                ticker_col=cols["ticker"],
                timestamp_col=cols["timestamp"],
            ),
        )

    label_cfg = cfg["labels"]
    labelled = build_return_labels(
        news,
        returns,
        horizon_days=int(label_cfg["horizon_days"]),
        timestamp_col=cols["timestamp"],
        ticker_col=cols["ticker"],
        date_col=cols["date"],
        return_col=cols["return"],
        market_return_col=cols["market_return"],
        excess_return=bool(label_cfg.get("excess_return", True)),
    )
    labelled.to_csv(out_dir / "news_with_forward_labels.csv", index=False)

    target = f"fwd_{int(label_cfg['horizon_days'])}d"
    reg = predictive_return_regression(
        labelled,
        return_col=target,
        signal_col=cols["sentiment_score"],
        controls=[],
        ticker_col=cols["ticker"],
        date_col="signal_date",
        fixed_effects=["ticker"],
    )
    write_regression_summary(reg, str(out_dir / "predictive_return_regression.txt"))

    signal = daily_sentiment_signal(
        news,
        returns[cols["date"]],
        timestamp_col=cols["timestamp"],
        ticker_col=cols["ticker"],
        score_col=cols["sentiment_score"],
    )
    signal.to_csv(out_dir / "daily_sentiment_signal.csv", index=False)

    bt_cfg = cfg["backtest"]
    bt, weights, summary = run_long_short_backtest(
        signal,
        returns,
        quantile=float(bt_cfg["quantile"]),
        transaction_cost_bps=float(bt_cfg["transaction_cost_bps"]),
        lag_days=int(bt_cfg.get("signal_lag_days", 1)),
        date_col=cols["date"],
        ticker_col=cols["ticker"],
        signal_col="signal",
        return_col=cols["return"],
    )
    bt.to_csv(out_dir / "long_short_returns.csv")
    weights.to_csv(out_dir / "long_short_weights.csv")
    summary.to_csv(out_dir / "long_short_summary.csv", header=["value"])

    wealth = (1 + bt["net_return"].fillna(0)).cumprod()
    ax = wealth.plot(legend=False)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth")
    plt.tight_layout()
    plt.savefig(out_dir / "long_short_wealth.png", dpi=300)
    plt.close()

    print(f"Wrote sentiment-trading outputs to {out_dir}")


if __name__ == "__main__":
    main()
