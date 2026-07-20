#!/usr/bin/env python3
"""Validate and regenerate Markdown issue files from the canonical backlog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_backlog(pack_root: Path) -> dict[str, Any]:
    return json.loads((pack_root / "issues" / "backlog.json").read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")[:72]


def expected_files(pack_root: Path) -> dict[Path, str]:
    backlog = load_backlog(pack_root)
    root = pack_root / "issues" / "generated"
    expected: dict[Path, str] = {}
    for issue in backlog.get("issues", []):
        out = root / issue["repository"] / f"{issue['id']}_{slugify(issue['title'])}.md"
        labels = ", ".join(f"`{label}`" for label in issue.get("labels", []))
        expected[out] = (
            f"# {issue['title']}\n\n"
            f"- **ID:** `{issue['id']}`\n"
            f"- **Track:** `{issue['track']}`\n"
            f"- **Priority:** `{issue['priority']}`\n"
            f"- **Labels:** {labels}\n\n"
            f"{issue['body']}"
        )
    return expected


def generate(pack_root: Path, *, clean: bool = False) -> list[Path]:
    root = pack_root / "issues" / "generated"
    if clean and root.exists():
        for path in root.rglob("*.md"):
            path.unlink()
    written: list[Path] = []
    for out, text in expected_files(pack_root).items():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written


def check(pack_root: Path) -> list[str]:
    expected = expected_files(pack_root)
    root = pack_root / "issues" / "generated"
    actual = set(root.rglob("*.md")) if root.exists() else set()
    findings: list[str] = []
    for path in sorted(set(expected) - actual):
        findings.append(f"missing: {path.relative_to(pack_root)}")
    for path in sorted(actual - set(expected)):
        findings.append(f"unexpected: {path.relative_to(pack_root)}")
    for path in sorted(set(expected) & actual):
        if path.read_text(encoding="utf-8") != expected[path]:
            findings.append(f"content drift: {path.relative_to(pack_root)}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify generated issue files without modifying them")
    args = parser.parse_args()
    pack_root = args.pack_root.resolve()
    if args.check:
        findings = check(pack_root)
        if findings:
            print("Generated issue registry is out of date:")
            for finding in findings:
                print(f"- {finding}")
            return 2
        print(f"Generated issue registry is current ({len(expected_files(pack_root))} files)")
        return 0
    written = generate(pack_root, clean=args.clean)
    print(f"Generated {len(written)} issue files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
