#!/usr/bin/env python3
"""Generate a reviewer-response implementation matrix.

The matrix is intended for conductor agents updating the preprint and package.
It keeps reviewer feedback tied to concrete code/docs/manuscript actions instead
of letting the response become another narrative-only plan.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "templates" / "reviewer_response_items.json"


@dataclass(frozen=True)
class ResponseItem:
    id: str
    source: str
    theme: str
    reviewer_comment: str
    implementation_action: str
    manuscript_action: str
    conductor_track: str
    status: str = "todo"
    owner: str = "conductor-agent"


def load_items(path: Path) -> list[ResponseItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("reviewer response input must be a list or object with items[]")
    items: list[ResponseItem] = []
    for record in records:
        items.append(ResponseItem(**record))
    return items


def build_matrix(items: list[ResponseItem]) -> dict[str, object]:
    by_theme: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for item in items:
        by_theme[item.theme] = by_theme.get(item.theme, 0) + 1
        by_status[item.status] = by_status.get(item.status, 0) + 1
    return {
        "schema_version": "1.0",
        "summary": {"items": len(items), "theme_counts": by_theme, "status_counts": by_status},
        "items": [asdict(item) for item in items],
        "recommendations": [
            "Do not submit until every major reviewer concern has either an implemented change or an explicit scoped rationale.",
            "Use this matrix to decide which material belongs in the main manuscript, supplement, repo docs, or future work.",
        ],
    }


def to_markdown(matrix: dict[str, object]) -> str:
    lines = ["# Reviewer response implementation matrix", "", "## Summary", ""]
    for key, value in matrix["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Recommendations", ""])
    for item in matrix.get("recommendations", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| ID | Source | Theme | Reviewer concern | Implementation action | Manuscript action | Track | Status |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in matrix["items"]:
        lines.append(
            f"| `{item['id']}` | {item['source']} | {item['theme']} | {item['reviewer_comment']} | {item['implementation_action']} | {item['manuscript_action']} | `{item['conductor_track']}` | `{item['status']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, nargs="?", default=Path("."), help="Repo root where local matrix should be written")
    parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    items = load_items(args.input)
    matrix = build_matrix(items)
    out_dir = args.repo / ".conductor" / "local"
    output_json = args.output_json or out_dir / "reviewer_response_matrix.json"
    output_md = args.output_md or out_dir / "reviewer_response_matrix.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(to_markdown(matrix), encoding="utf-8")
    print(f"Reviewer response matrix written: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
