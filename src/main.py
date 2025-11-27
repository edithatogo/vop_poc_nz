"""
Main Entry Point for Health Economic Analysis Pipeline.

This script orchestrates the full analysis workflow:
1. Runs the core analysis pipeline (CEA, DCEA, VOI, DSA).
2. Runs the reporting pipeline (Figures, Dashboards, Policy Brief).
"""

import argparse
import sys
from pathlib import Path

from .pipeline.analysis import run_analysis_pipeline
from .pipeline.reporting import run_reporting_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Health Economic Analysis Pipeline")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Directory to save outputs"
    )
    args = parser.parse_args()

    print(f"Starting analysis pipeline... Output directory: {args.output_dir}")

    # 1. Run Analysis
    results = run_analysis_pipeline()

    # 2. Run Reporting
    run_reporting_pipeline(results, output_dir=args.output_dir)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
