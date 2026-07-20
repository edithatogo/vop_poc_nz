from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
PUBLICATION_GUARD = (
    "if: ${{ github.event_name == 'workflow_dispatch' && inputs.publish == true }}"
)


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n.*?(?=^  [a-z][a-z0-9-]*:\n|\Z)",
        workflow,
    )
    assert match is not None, f"missing release job: {name}"
    return match.group(0)


def test_private_draft_is_the_manual_dispatch_default() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(
        r"(?ms)^      publish:\n"
        r"        description: .*\n"
        r"        required: true\n"
        r"        type: boolean\n"
        r"        default: false$",
        workflow,
    )
    draft = _job(workflow, "draft-release")
    assert "gh release create" in draft
    assert "--draft" in draft
    assert PUBLICATION_GUARD not in draft


def test_every_publication_job_requires_explicit_manual_authorization() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for job_name in (
        "publish-testpypi",
        "publish-pypi",
        "smoke-test",
        "finalize-release",
    ):
        assert PUBLICATION_GUARD in _job(workflow, job_name)


def test_contract_tags_cannot_enter_the_package_release_workflow() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert '- "v[0-9]*.[0-9]*.[0-9]*"' in workflow
    assert '- "v*"' not in workflow
    assert "vop-voiage-contracts" not in _job(workflow, "build")
