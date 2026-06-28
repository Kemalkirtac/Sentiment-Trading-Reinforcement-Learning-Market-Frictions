from __future__ import annotations

import argparse

import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import read_table
from thesis_finllm.plotting import plot_cumulative_wealth
from thesis_finllm.portfolio_baselines import run_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description='Run portfolio baselines for PPO/SAPPO comparison.')
    parser.add_argument('--config', default='configs/portfolio_baselines.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['paths']['output_dir'])

    returns = read_table(cfg['paths']['asset_returns'], parse_dates=['date']).set_index('date').sort_index()
    sentiment = None
    if cfg['paths'].get('asset_sentiment'):
        sentiment = read_table(cfg['paths']['asset_sentiment'], parse_dates=['date']).set_index('date').sort_index()

    summary, backtests = run_baselines(
        returns,
        sentiment=sentiment,
        transaction_cost_bps=cfg['backtest']['transaction_cost_bps'],
    )
    summary.to_csv(out_dir / 'portfolio_baseline_summary.csv')
    wealth = {}
    for name, bt in backtests.items():
        bt.to_csv(out_dir / f'{name}_returns.csv')
        wealth[name] = bt['net_return']
    plot_cumulative_wealth(pd.DataFrame(wealth), out_dir / 'portfolio_baseline_wealth.png')
    print(f'Wrote portfolio baseline outputs to {out_dir}')


if __name__ == '__main__':
    main()
