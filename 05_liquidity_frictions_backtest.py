from __future__ import annotations

import argparse

import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import read_table
from thesis_finllm.frictions import apply_liquidity_frictions, capacity_mask, liquidity_summary
from thesis_finllm.metrics import summarize_returns
from thesis_finllm.plotting import plot_cumulative_wealth
from thesis_finllm.trading import long_short_weights, make_signal_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description='Liquidity and market-frictions sensitivity for sentiment portfolios.')
    parser.add_argument('--config', default='configs/liquidity_frictions.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg['paths']['output_dir'])

    signal = read_table(cfg['paths']['signal'], parse_dates=['date'])
    returns = read_table(cfg['paths']['returns'], parse_dates=['date'])
    liquidity = read_table(cfg['paths']['liquidity'], parse_dates=['date'])

    sig, rets = make_signal_matrix(
        signal,
        returns,
        signal_col=cfg['columns']['signal_col'],
        lag_days=cfg['backtest']['lag_days'],
    )
    weights = long_short_weights(sig, quantile=cfg['backtest']['quantile'])
    dv = liquidity.pivot_table(index='date', columns='ticker', values=cfg['columns']['dollar_volume_col'], aggfunc='mean')

    bt = apply_liquidity_frictions(
        weights,
        rets,
        dollar_volume=dv,
        transaction_cost_bps=cfg['backtest']['transaction_cost_bps'],
        gross_exposure_usd=cfg['backtest']['gross_exposure_usd'],
        impact_coefficient=cfg['backtest']['impact_coefficient'],
    )
    bt.to_csv(out_dir / 'friction_adjusted_returns.csv')
    weights.to_csv(out_dir / 'friction_adjusted_weights.csv')
    liquidity_summary(bt).to_csv(out_dir / 'liquidity_summary.csv', header=['value'])
    summarize_returns(bt['net_return']).to_csv(out_dir / 'net_return_summary.csv', header=['value'])
    mask = capacity_mask(weights, cfg['backtest']['gross_exposure_usd'], dv, cfg['backtest']['max_participation'])
    pd.Series({
        'share_trade_asset_days_over_capacity': mask.stack().mean(),
        'days_with_any_capacity_violation': mask.any(axis=1).mean(),
    }).to_csv(out_dir / 'capacity_diagnostics.csv', header=['value'])
    plot_cumulative_wealth(bt['net_return'], out_dir / 'friction_adjusted_wealth.png')
    print(f'Wrote liquidity-friction outputs to {out_dir}')


if __name__ == '__main__':
    main()
