"""Repository-wide contracts for the reviewed uv workflow frontier."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SETUP_UV = "uses: astral-sh/setup-uv@"
REVIEWED_UV_VERSION = 'version: "0.11.29"'


def _setup_uv_blocks(text: str) -> list[list[str]]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    for index, line in enumerate(lines):
        if SETUP_UV not in line:
            continue
        action_indent = len(line) - len(line.lstrip())
        block = [line]
        for candidate in lines[index + 1 : index + 14]:
            indent = len(candidate) - len(candidate.lstrip())
            if candidate.strip().startswith("- ") and indent <= action_indent:
                break
            block.append(candidate)
        blocks.append(block)
    return blocks


def test_every_setup_uv_lane_pins_the_reviewed_binary_version() -> None:
    """Pinned action code must not download an unreviewed floating uv binary."""
    missing: list[str] = []
    observed = 0
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        for block in _setup_uv_blocks(workflow.read_text(encoding="utf-8")):
            observed += 1
            if not any(REVIEWED_UV_VERSION in line for line in block):
                missing.append(workflow.relative_to(ROOT).as_posix())

    assert observed >= 18
    assert missing == []
