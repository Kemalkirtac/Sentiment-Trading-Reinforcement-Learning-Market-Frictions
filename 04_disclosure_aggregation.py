from __future__ import annotations

import argparse

import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import read_table
from thesis_finllm.disclosures import (
    aggregate_model_outputs,
    benchmark_adjusted_returns,
    disclosure_event_returns,
    fit_disclosure_return_model,
)
from thesis_finllm.regressions import write_regression_summary


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM-disagreement aggregation for disclosure abnormal returns.')
    parser.add_argument('--config', default='configs/disclosure_aggregation.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['paths']['output_dir'])

    disclosures = read_table(cfg['paths']['disclosures'], parse_dates=['release_date'])
    returns = read_table(cfg['paths']['returns'], parse_dates=['date'])
    model_cols = cfg['models']['output_columns']

    agg = aggregate_model_outputs(disclosures, model_cols)
    adj_returns = benchmark_adjusted_returns(returns)
    event = disclosure_event_returns(agg, adj_returns, windows=cfg['event_windows'])
    event.to_csv(out_dir / 'disclosure_event_returns.csv', index=False)

    result = fit_disclosure_return_model(
        event,
        outcome_col=cfg['regression']['outcome_col'],
        score_col='llm_mean_score',
        disagreement_col='llm_disagreement',
        controls=cfg['regression'].get('controls', []),
    )
    write_regression_summary(result, str(out_dir / 'disclosure_return_regression.txt'))
    pd.Series({
        'n_disclosures': len(event),
        'mean_llm_score': event['llm_mean_score'].mean(),
        'mean_disagreement': event['llm_disagreement'].mean(),
        'mean_outcome': event[cfg['regression']['outcome_col']].mean(),
    }).to_csv(out_dir / 'disclosure_summary.csv', header=['value'])
    print(f'Wrote disclosure aggregation outputs to {out_dir}')


if __name__ == '__main__':
    main()
