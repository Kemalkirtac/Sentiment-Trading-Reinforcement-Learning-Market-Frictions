from __future__ import annotations

import argparse

from thesis_finllm.config import load_config
from thesis_finllm.reporting import build_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a Markdown report from output folders.')
    parser.add_argument('--config', default='configs/results_report.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = build_markdown_report(cfg['paths']['output_root'], cfg['paths']['report_path'])
    print(f'Wrote report to {report}')


if __name__ == '__main__':
    main()
