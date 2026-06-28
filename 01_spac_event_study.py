from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thesis_finllm.config import ensure_dir, load_config
from thesis_finllm.data import read_table
from thesis_finllm.event_study import EventStudyConfig, compute_event_abnormal_returns, run_event_regression
from thesis_finllm.regressions import write_regression_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPAC media-sentiment event study.")
    parser.add_argument("--config", default="configs/spac_event_study.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    cols = cfg["columns"]
    out_dir = ensure_dir(paths["output_dir"])

    events = read_table(paths["events"], parse_dates=[cols["event_date"]])
    returns = read_table(paths["returns"], parse_dates=[cols["date"]])

    mm = cfg["market_model"]
    es_cfg = EventStudyConfig(
        estimation_window=tuple(mm["estimation_window"]),
        min_estimation_obs=int(mm["min_estimation_obs"]),
        event_windows={k: tuple(v) for k, v in mm["event_windows"].items()},
    )
    car = compute_event_abnormal_returns(events, returns, es_cfg, cols)
    car.to_csv(out_dir / "spac_event_cars.csv", index=False)

    reg_cfg = cfg["regression"]
    result = run_event_regression(
        car_df=car,
        dependent_var=reg_cfg["dependent_var"],
        sentiment_vars=cols["sentiment_vars"],
        controls=reg_cfg.get("controls", []),
        event_type_col=cols.get("event_type", "event_type"),
    )
    write_regression_summary(result, str(out_dir / "spac_event_regression.txt"))

    summary = car.groupby(cols["event_type"])[list(mm["event_windows"].keys())].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "spac_event_summary.csv")

    # Simple black-and-white figure: CAR distribution by event type.
    dep = reg_cfg["dependent_var"]
    ax = car.boxplot(column=dep, by=cols["event_type"], grid=False)
    ax.set_title("")
    ax.get_figure().suptitle("")
    ax.set_xlabel("Event type")
    ax.set_ylabel(dep)
    plt.tight_layout()
    plt.savefig(out_dir / "spac_car_boxplot.png", dpi=300)
    plt.close()

    print(f"Wrote SPAC event-study outputs to {out_dir}")


if __name__ == "__main__":
    main()
