#!/usr/bin/env python3
"""Convenience wrapper to run the cluster proxy benchmark from python_analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.cluster_proxy import run_proxy_benchmark  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cluster proxy benchmark wrapper")
    parser.add_argument("--steps", type=int, default=10, help="Training steps per model")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metrics = run_proxy_benchmark(steps=args.steps)
    print(metrics.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
