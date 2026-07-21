#!/usr/bin/env python
"""Regenerate manuscript analytical results and bind them to a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vop_poc_nz.cea_model_core import calculate_cea
from vop_poc_nz.pipeline.analysis import load_parameters

SCHEMA_VERSION = "1.0.0"
DEFAULT_SEED = 20260721
DEFAULT_DRAWS = 10_000
WTP_DECISION = 20_000.0
WTP_FIGURE = 50_000.0
CASES = {
    "HPV Vaccination": "hpv_vaccination",
    "Smoking Cessation": "smoking_cessation",
    "Hepatitis C Therapy": "hepatitis_c_therapy",
    "Childhood Obesity Prevention": "childhood_obesity_prevention",
    "Housing Insulation": "housing_insulation",
}


@dataclass(frozen=True)
class Draws:
    hs_cost: np.ndarray
    soc_cost: np.ndarray
    qaly: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _lognormal(
    rng: np.random.Generator, cv: float, shape: tuple[int, ...]
) -> np.ndarray:
    sigma = np.sqrt(np.log1p(cv * cv))
    return rng.lognormal(mean=-(sigma * sigma) / 2, sigma=sigma, size=shape)


def _draw_case(
    hs: dict[str, Any], soc: dict[str, Any], rng: np.random.Generator, draws: int
) -> Draws:
    # Independent arm-level multipliers are a transparent synthetic uncertainty
    # model. QALY multipliers are shared across perspectives to preserve the
    # declared perspective boundary.
    cost_hs_mult = _lognormal(rng, 0.10, (draws, 2))
    cost_soc_mult = _lognormal(rng, 0.15, (draws, 2))
    qaly_mult = _lognormal(rng, 0.05, (draws, 2))
    hs_cost = (
        hs["cost_new_treatment"] * cost_hs_mult[:, 1]
        - hs["cost_standard_care"] * cost_hs_mult[:, 0]
    )
    soc_cost = (
        soc["cost_new_treatment"] * cost_soc_mult[:, 1]
        - soc["cost_standard_care"] * cost_soc_mult[:, 0]
    )
    qaly = (
        hs["qalys_new_treatment"] * qaly_mult[:, 1]
        - hs["qalys_standard_care"] * qaly_mult[:, 0]
    )
    return Draws(hs_cost=hs_cost, soc_cost=soc_cost, qaly=qaly)


def _interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(low),
        "upper_95": float(high),
        "mcse": float(np.std(values, ddof=1) / np.sqrt(values.size)),
    }


def _ratio_interval(cost: np.ndarray, qaly: np.ndarray) -> dict[str, float]:
    stable = np.abs(qaly) > 1e-9
    ratios = cost[stable] / qaly[stable]
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "mean": float(np.median(ratios)),
        "lower_95": float(low),
        "upper_95": float(high),
        "mcse": float(np.std(ratios, ddof=1) / np.sqrt(ratios.size)),
        "estimand": "median",
    }


def _probability_interval(events: np.ndarray) -> dict[str, float]:
    probability = float(np.mean(events))
    n = events.size
    z = 1.959963984540054
    denominator = 1 + z * z / n
    centre = (probability + z * z / (2 * n)) / denominator
    half = (
        z
        * np.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n))
        / denominator
    )
    return {
        "mean": probability,
        "lower_95": float(max(0, centre - half)),
        "upper_95": float(min(1, centre + half)),
        "mcse": float(np.sqrt(probability * (1 - probability) / n)),
        "interval_method": "Wilson score",
    }


def _fmt_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}\\${abs(value):,.0f}"


def _fmt_interval(summary: dict[str, float]) -> str:
    return (
        f"{_fmt_currency(summary['mean'])} "
        f"[{_fmt_currency(summary['lower_95'])}, "
        f"{_fmt_currency(summary['upper_95'])}]"
    )


def _write_results_tex(
    results: list[dict[str, Any]], destination: Path, *, draws: int
) -> None:
    rows = []
    decision_rows = []
    for item in results:
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                item["name"],
                _fmt_interval(item["hs_icer"]),
                _fmt_interval(item["soc_icer"]),
                _fmt_interval(item["vop_per_person"]),
            )
        )
        decision_rows.append(
            "{} & {} & {} & {:.1f}\\% [{:.1f}\\%, {:.1f}\\%] & {} \\\\".format(
                item["name"],
                item["hs_decision"],
                item["soc_decision"],
                100 * item["discordance_probability"]["mean"],
                100 * item["discordance_probability"]["lower_95"],
                100 * item["discordance_probability"]["upper_95"],
                _fmt_interval(item["vop_per_person"]),
            )
        )
    by_name = {item["name"]: item for item in results}
    smoking = by_name["Smoking Cessation"]
    housing = by_name["Housing Insulation"]

    def probability_text(item: dict[str, Any]) -> str:
        value = item["discordance_probability"]
        return (
            f"{100 * value['mean']:.1f}\\% "
            f"(95\\% Wilson interval {100 * value['lower_95']:.1f}--"
            f"{100 * value['upper_95']:.1f}\\%)"
        )

    destination.write_text(
        "% Generated by scripts/regenerate_manuscript_results.py; do not edit.\n"
        f"\\newcommand{{\\PublicationDrawCount}}{{{draws:,}}}\n"
        f"\\newcommand{{\\PublicationSmokingVop}}{{{_fmt_interval(smoking['vop_per_person'])}}}\n"
        f"\\newcommand{{\\PublicationHousingVop}}{{{_fmt_interval(housing['vop_per_person'])}}}\n"
        f"\\newcommand{{\\PublicationSmokingDiscordance}}{{{probability_text(smoking)}}}\n"
        f"\\newcommand{{\\PublicationHousingDiscordance}}{{{probability_text(housing)}}}\n"
        "\\newcommand{\\PublicationResultRows}{%\n" + "\n".join(rows) + "\n}\n"
        "\\newcommand{\\PublicationDecisionRows}{%\n"
        + "\n".join(decision_rows)
        + "\n}\n",
        encoding="utf-8",
        newline="\n",
    )


def _plot_outputs(
    draw_map: dict[str, Draws], results: list[dict[str, Any]], figures: Path
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(results)))

    # Comparative cost-effectiveness plane.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for color, item in zip(palette, results, strict=True):
        draws = draw_map[item["name"]]
        idx = np.linspace(0, len(draws.qaly) - 1, 1000, dtype=int)
        axes[0].scatter(
            draws.qaly[idx] / 1000,
            draws.hs_cost[idx] / 1000,
            s=4,
            alpha=0.18,
            color=color,
        )
        axes[1].scatter(
            draws.qaly[idx] / 1000,
            draws.soc_cost[idx] / 1000,
            s=4,
            alpha=0.18,
            color=color,
            label=item["name"],
        )
    for axis, title in zip(axes, ["Health-system", "Societal"], strict=True):
        axis.axhline(0, color="0.3", linewidth=0.7)
        axis.axvline(0, color="0.3", linewidth=0.7)
        axis.set(
            title=title,
            xlabel="Incremental QALYs per person",
            ylabel="Incremental cost (NZ$ per person)",
        )
    axes[1].legend(fontsize=6)
    fig.savefig(figures / "cost_effectiveness_plane_comparative.png", dpi=220)
    plt.close(fig)

    # Perspective NMB difference distributions.
    values = []
    for item in results:
        d = draw_map[item["name"]]
        values.append(
            ((WTP_FIGURE * d.qaly - d.soc_cost) - (WTP_FIGURE * d.qaly - d.hs_cost))
            / 1000
        )
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    parts = ax.violinplot(values, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], palette, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xticks(
        range(1, len(results) + 1),
        [x["name"] for x in results],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Societal minus health-system NMB (NZ$ per person)")
    fig.savefig(figures / "delta_nmb_violin.png", dpi=220)
    plt.close(fig)

    # CEAC with Wilson 95% bands.
    thresholds = np.linspace(0, 100_000, 101)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for color, item in zip(palette, results, strict=True):
        d = draw_map[item["name"]]
        probability = np.array(
            [np.mean(w * d.qaly - d.soc_cost > 0) for w in thresholds]
        )
        n = len(d.qaly)
        z = 1.959963984540054
        centre = (probability + z * z / (2 * n)) / (1 + z * z / n)
        half = (
            z
            * np.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n))
            / (1 + z * z / n)
        )
        ax.plot(thresholds, probability, color=color, label=item["name"])
        ax.fill_between(
            thresholds, centre - half, centre + half, color=color, alpha=0.12
        )
    ax.set(
        xlabel="Willingness-to-pay (NZ$/QALY)",
        ylabel="Probability cost-effective",
        ylim=(0, 1),
    )
    ax.legend(fontsize=6)
    fig.savefig(figures / "ceac_societal.png", dpi=220)
    plt.close(fig)

    # HPV Markov trace is deterministic and comes from the same input model.
    params = load_parameters()[CASES["HPV Vaccination"]]
    trace = calculate_cea(params, perspective="societal")["trace_new_treatment"]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for state, series, color in zip(
        params["states"],
        trace.T / np.sum(params["initial_population"]),
        palette,
        strict=False,
    ):
        ax.plot(np.arange(trace.shape[0]), series, label=state, color=color)
    ax.set(xlabel="Cycle", ylabel="Cohort proportion", ylim=(0, 1))
    ax.legend()
    fig.savefig(figures / "markov_trace_hpv_vaccination_new_treatment.png", dpi=220)
    plt.close(fig)


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def generate(root: Path, *, seed: int, draws: int) -> Path:
    root = root.resolve()
    out = root / "manuscript" / "generated"
    figures = root / "manuscript" / "figures" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    params_path = root / "src" / "vop_poc_nz" / "parameters.yaml"
    script_path = root / "scripts" / "regenerate_manuscript_results.py"
    parameters = load_parameters(str(params_path))
    rng = np.random.default_rng(seed)
    result_rows: list[dict[str, Any]] = []
    draw_map: dict[str, Draws] = {}
    for name, key in CASES.items():
        case = parameters[key]
        hs = calculate_cea(
            case, perspective="health_system", wtp_threshold=WTP_DECISION
        )
        soc = calculate_cea(case, perspective="societal", wtp_threshold=WTP_DECISION)
        sampled = _draw_case(hs, soc, rng, draws)
        population = float(np.sum(case["initial_population"]))
        hs_nmb = WTP_DECISION * sampled.qaly - sampled.hs_cost
        soc_nmb = WTP_DECISION * sampled.qaly - sampled.soc_cost
        hs_accept = hs_nmb > 0
        soc_accept = soc_nmb > 0
        loss = (
            np.where(hs_accept, np.maximum(0, -soc_nmb), np.maximum(0, soc_nmb))
            / population
        )
        discordance = hs_accept != soc_accept
        probability = _probability_interval(discordance)
        row = {
            "name": name,
            "population": int(population),
            "hs_icer": _ratio_interval(sampled.hs_cost, sampled.qaly),
            "soc_icer": _ratio_interval(sampled.soc_cost, sampled.qaly),
            "hs_incremental_nmb_per_person": _interval(hs_nmb / population),
            "soc_incremental_nmb_per_person": _interval(soc_nmb / population),
            "vop_per_person": _interval(loss),
            "discordance_probability": probability,
            "hs_decision": "Accept" if hs["incremental_nmb"] > 0 else "Reject",
            "soc_decision": "Accept" if soc["incremental_nmb"] > 0 else "Reject",
        }
        result_rows.append(row)
        draw_map[name] = sampled

    results_path = out / "publication-results.json"
    results_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "seed": seed,
            "draws": draws,
            "decision_wtp_nzd_per_qaly": WTP_DECISION,
            "figure_wtp_nzd_per_qaly": WTP_FIGURE,
            "interval": "equal-tailed 95% simulation interval",
            "uncertainty_model": {
                "distribution": "mean-one lognormal arm multipliers",
                "health_system_cost_cv": 0.10,
                "societal_cost_cv": 0.15,
                "qaly_cv": 0.05,
                "qaly_draws_shared_across_perspectives": True,
            },
        },
        "results": result_rows,
    }
    results_path.write_bytes(_canonical_json(results_payload))
    tex_path = out / "publication-results.tex"
    _write_results_tex(result_rows, tex_path, draws=draws)
    _plot_outputs(draw_map, result_rows, figures)

    artifacts = [results_path, tex_path, *sorted(figures.glob("*.png"))]
    try:
        package_version = version("vop-poc-nz")
    except PackageNotFoundError:
        package_version = "unknown"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": _git_revision(root),
        "generator": script_path.relative_to(root).as_posix(),
        "generator_sha256": _sha256(script_path),
        "input": params_path.relative_to(root).as_posix(),
        "input_sha256": _sha256(params_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "vop_poc_nz": package_version,
        },
        "analysis": results_payload["analysis"],
        "artifacts": [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for path in artifacts
        ],
    }
    manifest_path = out / "publication-results-manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return manifest_path


def verify(root: Path) -> list[str]:
    root = root.resolve()
    manifest_path = (
        root / "manuscript" / "generated" / "publication-results-manifest.json"
    )
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported manifest schema")
    for key, hash_key in (("generator", "generator_sha256"), ("input", "input_sha256")):
        path = root / payload[key]
        if not path.is_file() or _sha256(path) != payload[hash_key]:
            failures.append(f"stale or missing {key}: {payload[key]}")
    for artifact in payload.get("artifacts", []):
        path = root / artifact["path"]
        if not path.is_file() or _sha256(path) != artifact["sha256"]:
            failures.append(f"stale or missing artifact: {artifact['path']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        failures = verify(args.root)
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print("publication-results: verified")
        return 0
    print(generate(args.root, seed=args.seed, draws=args.draws))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
