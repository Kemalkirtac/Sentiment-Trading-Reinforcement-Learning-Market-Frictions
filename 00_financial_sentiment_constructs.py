from __future__ import annotations

import argparse

import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.constructs import (
    add_construct_scores,
    construct_correlation_report,
    construct_table,
    monthly_construct_coverage,
)
from thesis_finllm.data import read_table
from thesis_finllm.sentiment import combine_text_fields


def main() -> None:
    parser = argparse.ArgumentParser(description='Construct-validity diagnostics for financial sentiment text.')
    parser.add_argument('--config', default='configs/financial_sentiment_constructs.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['paths']['output_dir'])

    news = read_table(cfg['paths']['news'], parse_dates=['timestamp'])
    news = combine_text_fields(news, cfg['text']['fields'], output_col='text')
    scored = add_construct_scores(news, text_col='text')
    scored.to_csv(out_dir / 'news_construct_scores.csv', index=False)

    construct_table().to_csv(out_dir / 'financial_sentiment_construct_taxonomy.csv', index=False)
    monthly_construct_coverage(
        scored.assign(date=pd.to_datetime(scored['timestamp']).dt.date),
        date_col='date',
        construct_cols=['media_tone_score', 'uncertainty_score', 'attention_score'],
    ).to_csv(out_dir / 'monthly_construct_coverage.csv', index=False)

    target_col = cfg['targets'].get('target_col')
    if target_col and target_col in scored.columns:
        corr = construct_correlation_report(
            scored,
            construct_cols=['media_tone_score', 'uncertainty_score', 'attention_score', 'word_count'],
            target_cols=[target_col],
        )
        corr.to_csv(out_dir / 'construct_target_correlations.csv', index=False)

    print(f'Wrote construct diagnostics to {out_dir}')


if __name__ == '__main__':
    main()
