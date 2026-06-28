from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    out = Path('data/sample')
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range('2021-01-01', periods=520)
    tickers = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']
    mkt = rng.normal(0.0003, 0.010, len(dates))

    rows = []
    for ticker in tickers:
        beta = rng.uniform(0.8, 1.3)
        alpha = rng.normal(0.0, 0.0002)
        eps = rng.normal(0.0, 0.018, len(dates))
        ret = alpha + beta * mkt + eps
        rows.extend({'date': d, 'ticker': ticker, 'ret': r, 'mkt_ret': m} for d, r, m in zip(dates, ret, mkt))
    daily_returns = pd.DataFrame(rows)
    daily_returns.to_csv(out / 'daily_returns.csv', index=False)

    # SPAC event study panel
    spacs = ['SPAC1', 'SPAC2', 'SPAC3', 'SPAC4', 'SPAC5', 'SPAC6']
    rows = []
    for ticker in spacs:
        beta = rng.uniform(0.6, 1.4)
        ret = beta * mkt + rng.normal(0, 0.025, len(dates))
        rows.extend({'date': d, 'ticker': ticker, 'ret': r, 'mkt_ret': m} for d, r, m in zip(dates, ret, mkt))
    pd.DataFrame(rows).to_csv(out / 'spac_returns.csv', index=False)
    event_rows = []
    for i, ticker in enumerate(spacs):
        for event_type, offset in [('ipo', 120 + i * 20), ('merger', 250 + i * 25)]:
            event_rows.append({
                'event_id': f'{ticker}_{event_type}',
                'ticker': ticker,
                'event_date': dates[min(offset, len(dates)-20)],
                'event_type': event_type,
                'ess': rng.normal(0, 50),
                'css': rng.normal(0, 50),
                'news_count': int(rng.integers(1, 15)),
            })
    pd.DataFrame(event_rows).to_csv(out / 'spac_events.csv', index=False)

    # News sentiment sample
    words_pos = ['beats estimates', 'strong growth', 'raises guidance', 'record profit', 'share buyback']
    words_neg = ['misses estimates', 'downgrade warning', 'weak demand', 'lawsuit probe', 'cuts guidance']
    news_rows = []
    for i in range(900):
        d = dates[int(rng.integers(0, len(dates)-5))]
        ticker = rng.choice(tickers)
        latent = rng.normal()
        phrase = rng.choice(words_pos if latent > 0 else words_neg)
        hour = int(rng.integers(8, 20))
        ts = pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
        news_rows.append({
            'news_id': f'N{i:05d}',
            'timestamp': ts,
            'date': pd.Timestamp(d).date(),
            'ticker': ticker,
            'headline': f'{ticker} {phrase}',
            'body': f'{ticker} reports {phrase}. Analysts cite uncertainty and market attention in coverage update.',
            'sentiment_score': float(np.tanh(latent)),
        })
    pd.DataFrame(news_rows).to_csv(out / 'news.csv', index=False)

    # Portfolio RL data
    asset_returns = daily_returns.pivot(index='date', columns='ticker', values='ret').reset_index()
    asset_returns.to_csv(out / 'asset_returns.csv', index=False)
    asset_sent = asset_returns.copy()
    for t in tickers:
        asset_sent[t] = rng.normal(0, 0.5, len(asset_sent)) + 2.0 * asset_returns[t].shift(-1).fillna(0)
    asset_sent.to_csv(out / 'asset_sentiment.csv', index=False)

    # Liquidity panel
    liquidity_rows = []
    for _, row in daily_returns.iterrows():
        liquidity_rows.append({
            'date': row['date'],
            'ticker': row['ticker'],
            'dollar_volume': float(rng.lognormal(mean=16.0, sigma=0.7)),
            'bid_ask_spread_bps': float(rng.uniform(2, 30)),
        })
    pd.DataFrame(liquidity_rows).to_csv(out / 'liquidity_panel.csv', index=False)

    # Disclosure aggregation sample
    disclosure_rows = []
    form_types = ['8-K', '10-Q', '10-K', 'EARNINGS']
    labels = ['positive', 'neutral', 'negative']
    for i in range(120):
        ticker = rng.choice(tickers)
        d = dates[int(rng.integers(20, len(dates)-10))]
        latent = rng.normal()
        label = 'positive' if latent > 0.4 else 'negative' if latent < -0.4 else 'neutral'
        noisy = lambda: rng.choice(labels, p=[0.65, 0.25, 0.10]) if label == 'positive' else rng.choice(labels, p=[0.10, 0.25, 0.65]) if label == 'negative' else rng.choice(labels, p=[0.2, 0.6, 0.2])
        disclosure_rows.append({
            'disclosure_id': f'D{i:05d}',
            'ticker': ticker,
            'release_date': d,
            'form_type': rng.choice(form_types),
            'text': f'{ticker} disclosure discusses results, uncertainty, guidance, and operational performance.',
            'llm_a_output': f'{{"label":"{noisy()}", "confidence":0.80}}',
            'llm_b_output': f'{{"label":"{noisy()}", "confidence":0.75}}',
            'llm_c_output': f'{{"label":"{noisy()}", "confidence":0.70}}',
            'log_market_cap': float(rng.normal(9.5, 1.0)),
            'disclosure_length': int(rng.integers(500, 5000)),
        })
    pd.DataFrame(disclosure_rows).to_csv(out / 'disclosures.csv', index=False)
    disclosure_returns = daily_returns.rename(columns={'mkt_ret': 'benchmark_ret'}).copy()
    disclosure_returns.to_csv(out / 'disclosure_returns.csv', index=False)

    print(f'Wrote synthetic data to {out}')


if __name__ == '__main__':
    main()
