"""
Snakemake workflow for local quality checks.
Targets:
  - lint: ruff check
  - format_check: ruff format --check
  - typecheck: fast syntax/type stub check (compileall placeholder; full mypy deferred)
  - test: pytest with coverage
  - ci: aggregate all of the above
"""

rule all:
    input:
        "lint",
        "format_check",
        "typecheck",
        "test"


rule lint:
    output: "lint"
    shell:
        "ruff check src/ tests/ && touch {output}"


rule format_check:
    output: "format_check"
    shell:
        "ruff format --check . && touch {output}"


rule typecheck:
    output: "typecheck"
    shell:
        # Fast, deterministic check; full mypy/pyright runs are deferred due to runtime/timeout concerns.
        "python -m compileall src && touch {output}"


rule test:
    output: "test"
    shell:
        "pytest -v --cov=src --cov-report=xml && touch {output}"


rule ci:
    input:
        rules.lint.output,
        rules.format_check.output,
        rules.typecheck.output,
        rules.test.output
    output: "ci_done"
    shell:
        "touch {output}"
