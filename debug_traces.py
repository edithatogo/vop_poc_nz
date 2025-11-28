
import logging

from src.cea_model_core import run_cea

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trace_generation():
    logger.info("Testing trace generation...")

    # Example parameters
    params = {
        "states": ["Healthy", "Sick", "Dead"],
        "transition_matrices": {
            "standard_care": [[0.9, 0.05, 0.05], [0, 0.8, 0.2], [0, 0, 1]],
            "new_treatment": [[0.95, 0.03, 0.02], [0, 0.85, 0.15], [0, 0, 1]],
        },
        "cycles": 10,
        "initial_population": [1000, 0, 0],
        "costs": {
            "health_system": {
                "standard_care": [0, 500, 0],
                "new_treatment": [100, 500, 0],
            },
            "societal": {"standard_care": [0, 2000, 0], "new_treatment": [0, 1000, 0]},
        },
        "qalys": {"standard_care": [1, 0.7, 0], "new_treatment": [1, 0.75, 0]},
    }

    # Run CEA
    results = run_cea(params, perspective="health_system")

    # Check for traces
    if "trace_standard_care" in results:
        trace = results["trace_standard_care"]
        if trace is not None:
            logger.info(f"Trace Standard Care found. Shape: {trace.shape}")
            logger.info(f"First row: {trace[0]}")
            logger.info(f"Last row: {trace[-1]}")
        else:
            logger.error("Trace Standard Care is None")
    else:
        logger.error("trace_standard_care key missing in results")

    if "trace_new_treatment" in results:
        trace = results["trace_new_treatment"]
        if trace is not None:
            logger.info(f"Trace New Treatment found. Shape: {trace.shape}")
        else:
            logger.error("Trace New Treatment is None")
    else:
        logger.error("trace_new_treatment key missing in results")

if __name__ == "__main__":
    test_trace_generation()
