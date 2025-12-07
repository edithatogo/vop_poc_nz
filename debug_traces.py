
import logging
import sys
from vop_poc_nz.pipeline.analysis import run_analysis_pipeline
from vop_poc_nz.pipeline.reporting import plot_markov_trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_traces():
    logger.info("Running analysis pipeline...")
    results = run_analysis_pipeline("src/parameters.yaml")
    
    if "HPV Vaccination" in results["cea_results"]:
        hpv = results["cea_results"]["HPV Vaccination"]
        if "societal" in hpv:
            soc = hpv["societal"]
            if "human_capital" in soc:
                hc = soc["human_capital"]
                if "trace_new_treatment" in hc:
                    trace = hc["trace_new_treatment"]
                    if trace is not None:
                        logger.info(f"Trace found! Shape: {trace.shape}")
                        plot_markov_trace(trace, ["Healthy", "Infected", "Cancer", "Dead"], "Debug Trace", output_dir="output/debug_figures")
                    else:
                        logger.error("Trace is None")
                else:
                    logger.error("trace_new_treatment key missing")
            else:
                logger.error("human_capital missing")
        else:
            logger.error("societal missing")
    else:
        logger.error("HPV Vaccination missing")

if __name__ == "__main__":
    debug_traces()
