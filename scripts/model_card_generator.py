#!/usr/bin/env python3
"""Generate concise model cards from case contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("case contract JSON must be an object")
        return data
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("YAML case contracts require PyYAML; convert to JSON or install PyYAML") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("case contract YAML must be a mapping")
    return data


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [f"{key}: {value[key]}" for key in sorted(value)]
    return [str(value)]


def section(lines: list[str], title: str, values: list[str]) -> None:
    lines.extend([f"## {title}", ""])
    if values:
        lines.extend(f"- {value}" for value in values)
    else:
        lines.append("- Not specified")
    lines.append("")


def model_card(contract: dict[str, Any]) -> str:
    case_id = contract.get("case_id") or contract.get("id") or "unknown_case"
    title = contract.get("title") or str(case_id).replace("_", " ").title()
    lines = [
        f"# Model card: {title}",
        "",
        f"Case ID: `{case_id}`",
        f"Case type: `{contract.get('case_type', 'not specified')}`",
        f"Model family: `{contract.get('model_family', 'not specified')}`",
        "",
    ]
    section(lines, "Decision question", as_list(contract.get("decision_question") or contract.get("question")))
    section(lines, "Population", as_list(contract.get("population")))
    section(lines, "Strategies", as_list(contract.get("decision_strategies") or contract.get("strategies")))
    section(lines, "Perspectives", as_list(contract.get("perspectives")))
    section(lines, "Cost components", as_list(contract.get("cost_components")))
    section(lines, "Time horizon and discounting", as_list(contract.get("time_horizon")) + as_list(contract.get("discounting")))
    section(lines, "Evidence/source grades", as_list(contract.get("source_grade") or contract.get("source_grades")))
    section(lines, "Validation status", as_list(contract.get("validation_status")))
    section(lines, "Important omissions", as_list(contract.get("omissions") or contract.get("limitations")))
    lines.extend(
        [
            "## Publication boundary",
            "",
            "- A `policy_grade` case requires complete evidence ledger, model-structure justification, result manifests, and output reconciliation.",
            "- An `empirical_tutorial` or `synthetic_fixture` case should not be described as a full policy-grade evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def discover_contracts(root: Path) -> list[Path]:
    patterns = ["examples/*.case_contract.yml", "examples/*.case_contract.yaml", "examples/*.case_contract.json", "cases/**/*.case_contract.*"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--contract", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo.resolve()
    contracts = [path.resolve() for path in args.contract] or discover_contracts(root)
    output_dir = args.output_dir.resolve() if args.output_dir else root / "docs" / "model_cards"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for path in contracts:
        contract = load_yaml_or_json(path)
        case_id = str(contract.get("case_id") or contract.get("id") or path.stem).replace(" ", "_")
        output = output_dir / f"{case_id}.model_card.md"
        output.write_text(model_card(contract), encoding="utf-8")
        written.append(str(output))
    local_dir = root / ".conductor" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "model_card_generation.json").write_text(json.dumps({"written": written}, indent=2) + "\n", encoding="utf-8")
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
