"""
Snakemake workflow for Health Economic Analysis Pipeline.

This workflow manages the execution of the analysis, figure generation,
and report compilation. It supports versioning via an output directory parameter.

Usage:
    snakemake -c1  # Run with 1 core
    snakemake -c1 --config output_dir=output/v1.0  # Run with custom output dir
"""

configfile: "src/parameters.yaml"

# Default output directory if not specified
# Output directory configuration
VERSION = config.get("version", "latest")
OUTPUT_DIR = f"output/{VERSION}"

rule all:
    input:
        f"{OUTPUT_DIR}/reports/policy_brief.md",
        f"{OUTPUT_DIR}/combined_report.md",
        f"{OUTPUT_DIR}/figures/dashboard_ce_voi_societal.png",
        f"{OUTPUT_DIR}/figures/discordance_loss.png"

rule run_analysis:
    output:
        directory(f"{OUTPUT_DIR}/figures"),
        directory(f"{OUTPUT_DIR}/reports"),
        f"{OUTPUT_DIR}/reports/policy_brief.md",
        f"{OUTPUT_DIR}/combined_report.md",
        f"{OUTPUT_DIR}/figures/dashboard_ce_voi_societal.png",
        f"{OUTPUT_DIR}/figures/discordance_loss.png"
    params:
        output_dir = OUTPUT_DIR
    shell:
        "python -m src.main --output-dir {params.output_dir}"

rule clean:
    shell:
        "rm -rf {OUTPUT_DIR}"

# Quality checks (inherited from previous Snakefile)
rule lint:
    shell:
        "ruff check src/ tests/"

rule format_check:
    shell:
        "ruff format --check ."

rule test:
    shell:
        "pytest -v --cov=src --cov-report=xml"
