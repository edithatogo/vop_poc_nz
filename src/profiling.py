"""
Profiling Script for Health Economic Analysis Pipeline.

This script runs the main analysis pipeline under cProfile to identify performance bottlenecks.
It saves the profiling statistics to 'output/profiling_stats.txt'.
"""

import cProfile
import io
import os
import pstats
import sys

from src.main import main


def profile_execution():
    print("Starting profiling run...")

    # Setup output directory
    output_dir = "output/profiling_run"
    os.makedirs(output_dir, exist_ok=True)

    # Mock command line arguments
    sys.argv = ["src.main", "--output-dir", output_dir]

    pr = cProfile.Profile()
    pr.enable()

    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"An error occurred during profiling: {e}")
    finally:
        pr.disable()

        print("\nProfiling completed. Generating report...")
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(50)  # Print top 50 time-consuming functions

        # Save to file
        stats_file = os.path.join(output_dir, "profiling_stats.txt")
        with open(stats_file, "w") as f:
            f.write(s.getvalue())

        print(f"Profiling statistics saved to {stats_file}")


if __name__ == "__main__":
    profile_execution()
