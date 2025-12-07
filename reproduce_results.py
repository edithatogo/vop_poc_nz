#!/usr/bin/env python3
"""
JSS Replication Script for vop_poc_nz.

This script serves as the standalone replication entry point for the manuscript.
It performs the following steps:
1. Checks for and installs the `vop_poc_nz` package in editable mode.
2. Checks for and installs `snakemake` if missing.
3. Executes the Snakemake workflow to generate all results and figures.

Usage:
    python reproduce_results.py
"""

import subprocess
import sys
import os
import shutil

def check_and_install_package():
    """Ensure vop_poc_nz is installed."""
    print("Checking vop_poc_nz installation...")
    try:
        import vop_poc_nz
        print(f"Found vop_poc_nz version: {vop_poc_nz.__version__}")
    except ImportError:
        print("vop_poc_nz not found. Installing in editable mode...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            print("Successfully installed vop_poc_nz.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing package: {e}")
            sys.exit(1)

def check_and_install_snakemake():
    """Ensure snakemake is installed."""
    print("Checking snakemake installation...")
    if shutil.which("snakemake") is None:
        print("snakemake not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "snakemake"])
            print("Successfully installed snakemake.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing snakemake: {e}")
            sys.exit(1)
    else:
        print("snakemake is available.")

def run_replication():
    """Run the analysis workflow."""
    print("\n" + "="*50)
    print("STARTING REPLICATION WORKFLOW")
    print("="*50 + "\n")
    
    # Define output directory for replication
    output_dir = "output/jss_replication"
    
    # Construct snakemake command
    # -c1: Use 1 core (for reproducibility/simplicity)
    # --config: Override output directory
    cmd = [
        "snakemake", 
        "-c1", 
        "--config", 
        f"output_dir={output_dir}"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "="*50)
        print("REPLICATION COMPLETE")
        print(f"Results are available in: {os.path.abspath(output_dir)}")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"\nError during replication execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_and_install_package()
    check_and_install_snakemake()
    run_replication()
