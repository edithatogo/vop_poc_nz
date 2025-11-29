"""
Main Entry Point for Health Economic Analysis Pipeline.

This script orchestrates the full analysis workflow:
1. Runs the core analysis pipeline (CEA, DCEA, VOI, DSA).
2. Runs the reporting pipeline (Figures, Dashboards, Policy Brief).

Usage:
    vop-poc-nz run --output-dir output
    vop-poc-nz report --results-file output/results.pkl
    vop-poc-nz --version
"""

import argparse
import logging
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")

from . import __version__
from .logging_config import setup_logging
from .pipeline.analysis import run_analysis_pipeline
from .pipeline.reporting import run_reporting_pipeline


def cmd_run(args):
    """Run full analysis pipeline (CEA, DCEA, VOI, DSA) + reporting."""
    setup_logging(output_dir=args.output_dir)
    logging.info(f"Starting analysis pipeline... Output directory: {args.output_dir}")

    # 1. Run Analysis
    results = run_analysis_pipeline()

    # Save results to pickle for easier re-running of reporting
    results_path = os.path.join(args.output_dir, "results.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Results saved to {results_path}")

    # 2. Run Reporting
    if not args.skip_reporting:
        run_reporting_pipeline(results, output_dir=args.output_dir)

    logging.info("Pipeline execution completed successfully.")


def cmd_report(args):
    """Generate reports from previously saved results."""
    setup_logging(output_dir=args.output_dir)

    if not os.path.exists(args.results_file):
        logging.error(f"Results file not found: {args.results_file}")
        sys.exit(1)

    logging.info(f"Loading results from {args.results_file}")
    with open(args.results_file, "rb") as f:
        results = pickle.load(f)

    run_reporting_pipeline(results, output_dir=args.output_dir)
    logging.info("Reporting completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        prog="vop-poc-nz",
        description="Distributional Cost-Effectiveness Analysis (DCEA) Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vop-poc-nz run                       Run full analysis pipeline
  vop-poc-nz run --output-dir results  Save outputs to 'results/' directory
  vop-poc-nz run --skip-reporting      Run analysis only, skip report generation
  vop-poc-nz report -r output/results.pkl  Regenerate reports from saved results

For more information, visit: https://github.com/edithatogo/vop_poc_nz
        """,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'run' subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run full analysis pipeline (CEA, DCEA, VOI, DSA, reporting)"
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save outputs (default: output)",
    )
    run_parser.add_argument(
        "--skip-reporting",
        action="store_true",
        help="Skip report generation (only run analysis)",
    )
    run_parser.set_defaults(func=cmd_run)

    # 'report' subcommand
    report_parser = subparsers.add_parser(
        "report", help="Generate reports from saved results"
    )
    report_parser.add_argument(
        "--results-file",
        "-r",
        type=str,
        default="output/results.pkl",
        help="Path to saved results pickle file (default: output/results.pkl)",
    )
    report_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save reports (default: output)",
    )
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
