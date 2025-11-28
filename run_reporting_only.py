
import pickle
import logging
from src.pipeline.reporting import run_reporting_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Loading results from output/results.pkl...")
    with open("output/results.pkl", "rb") as f:
        results = pickle.load(f)
    
    print("Running reporting pipeline...")
    run_reporting_pipeline(results, output_dir="output")
    print("Done.")

if __name__ == "__main__":
    main()
