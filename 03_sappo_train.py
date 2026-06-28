from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import chronological_split, read_table
from thesis_finllm.metrics import summarize_returns
from thesis_finllm.ppo import PPOConfig, evaluate_policy, train_ppo
from thesis_finllm.rl_env import PortfolioEnv, PortfolioEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact SAPPO/PPO portfolio agent.")
    parser.add_argument("--config", default="configs/sappo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    out_dir = ensure_dir(paths["output_dir"])

    returns = read_table(paths["asset_returns"], parse_dates=["date"]).set_index("date").sort_index()
    sentiment = read_table(paths["asset_sentiment"], parse_dates=["date"]).set_index("date").sort_index()

    exp = cfg["experiment"]
    train, valid, test = chronological_split(
        returns.reset_index(),
        date_col="date",
        train_fraction=float(exp["train_fraction"]),
        validation_fraction=float(exp["validation_fraction"]),
    )
    train_returns = train.set_index("date")
    test_returns = test.set_index("date")
    train_sent = sentiment.reindex(train_returns.index).fillna(0.0)
    test_sent = sentiment.reindex(test_returns.index).fillna(0.0)

    env_cfg = PortfolioEnvConfig(
        lookback=int(exp["lookback"]),
        transaction_cost_bps=float(exp["transaction_cost_bps"]),
        sentiment_mode=str(exp.get("sentiment_mode", "state")),
    )
    train_env = PortfolioEnv(train_returns, train_sent, env_cfg)
    ppo_cfg = PPOConfig(seed=int(exp["seed"]), **cfg["ppo"])
    model, logs = train_ppo(train_env, ppo_cfg)

    test_env = PortfolioEnv(test_returns, test_sent, env_cfg)
    rewards, infos = evaluate_policy(test_env, model)
    pd.Series(rewards, name="net_return").to_csv(out_dir / "sappo_test_returns.csv", index=False)
    summarize_returns(pd.Series(rewards)).to_csv(out_dir / "sappo_test_summary.csv", header=["value"])
    pd.DataFrame({"episode_return": logs["episode_returns"]}).to_csv(out_dir / "training_episode_returns.csv", index=False)

    wealth = (1 + pd.Series(rewards)).cumprod()
    ax = wealth.plot(legend=False)
    ax.set_xlabel("Test step")
    ax.set_ylabel("Cumulative wealth")
    plt.tight_layout()
    plt.savefig(out_dir / "sappo_test_wealth.png", dpi=300)
    plt.close()

    print(f"Wrote SAPPO outputs to {out_dir}")


if __name__ == "__main__":
    main()
