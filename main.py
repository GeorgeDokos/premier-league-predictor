"""Command-line entry point for the Premier League predictor."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulator import run_2026_27_prediction  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Simulate the 2026/27 Premier League season."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50000,
        help="Number of Monte Carlo season simulations (default: 50000).",
    )
    parser.add_argument(
        "--output",
        default="model_2a_transfers_2026_27.csv",
        help="Output CSV filename inside results/.",
    )
    args = parser.parse_args()

    if args.simulations <= 0:
        parser.error("--simulations must be greater than 0")

    run_2026_27_prediction(
        simulations=args.simulations,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()
