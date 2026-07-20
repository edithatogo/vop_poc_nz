#!/usr/bin/env python3
"""Guard mutating conductor operations with explicit Git worktree checks.

Mapping and audit commands remain read-only and may run on any branch. Copying
reference files into a live repository is different: by default this helper
requires a named non-default branch and no tracked or staged modifications.
Untracked local research artifacts do not block the guard because the integration
tools already refuse to overwrite existing paths and classify publication risk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_BRANCH_NAMES = frozenset({"main", "master"})


@dataclass(frozen=True)
class GitSafetyReport:
    repo: str
    is_git_repo: bool
    branch: str | None
    detached: bool
    tracked_dirty: bool
    staged_dirty: bool
    remote_default_branch: str | None
    appears_default_branch: bool
    blockers: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["safe"] = self.safe
        return payload


def _run(repo: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None


def _changed(repo: Path, args: Sequence[str]) -> bool:
    result = _run(repo, args)
    return result is None or result.returncode != 0


def inspect(repo: Path) -> GitSafetyReport:
    repo = repo.resolve()
    inside = _run(repo, ["rev-parse", "--is-inside-work-tree"])
    is_git_repo = bool(inside and inside.returncode == 0 and inside.stdout.strip() == "true")
    if not is_git_repo:
        return GitSafetyReport(
            repo=str(repo),
            is_git_repo=False,
            branch=None,
            detached=False,
            tracked_dirty=False,
            staged_dirty=False,
            remote_default_branch=None,
            appears_default_branch=False,
            blockers=("target is not a Git worktree",),
        )

    branch_result = _run(repo, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    branch = branch_result.stdout.strip() if branch_result and branch_result.returncode == 0 else None
    detached = branch is None
    tracked_dirty = _changed(repo, ["diff", "--quiet", "--"])
    staged_dirty = _changed(repo, ["diff", "--cached", "--quiet", "--"])

    remote_result = _run(repo, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"])
    remote_default = None
    if remote_result and remote_result.returncode == 0:
        value = remote_result.stdout.strip()
        remote_default = value.rsplit("/", 1)[-1] if value else None
    appears_default = bool(branch and (branch in DEFAULT_BRANCH_NAMES or branch == remote_default))

    blockers: list[str] = []
    if detached:
        blockers.append("HEAD is detached; create a named integration branch or worktree")
    if appears_default:
        blockers.append(f"current branch {branch!r} appears to be the default branch")
    if tracked_dirty or staged_dirty:
        blockers.append("tracked or staged changes are present; checkpoint or use a dedicated worktree")
    return GitSafetyReport(
        repo=str(repo),
        is_git_repo=True,
        branch=branch,
        detached=detached,
        tracked_dirty=tracked_dirty,
        staged_dirty=staged_dirty,
        remote_default_branch=remote_default,
        appears_default_branch=appears_default,
        blockers=tuple(blockers),
    )


def require_safe(
    repo: Path,
    *,
    allow_dirty: bool = False,
    allow_default_branch: bool = False,
    allow_detached: bool = False,
) -> GitSafetyReport:
    report = inspect(repo)
    remaining: list[str] = []
    for blocker in report.blockers:
        if "tracked or staged" in blocker and allow_dirty:
            continue
        if "default branch" in blocker and allow_default_branch:
            continue
        if "detached" in blocker and allow_detached:
            continue
        remaining.append(blocker)
    if remaining:
        joined = "; ".join(remaining)
        raise RuntimeError(f"Refusing mutating integration in {repo.resolve()}: {joined}")
    return report


def to_markdown(report: GitSafetyReport) -> str:
    lines = [
        "# Git integration safety",
        "",
        f"- Repository: `{report.repo}`",
        f"- Git worktree: **{report.is_git_repo}**",
        f"- Branch: `{report.branch}`",
        f"- Detached HEAD: **{report.detached}**",
        f"- Tracked changes: **{report.tracked_dirty}**",
        f"- Staged changes: **{report.staged_dirty}**",
        f"- Remote default branch: `{report.remote_default_branch}`",
        f"- Appears to be default branch: **{report.appears_default_branch}**",
        f"- Safe for mutating integration: **{report.safe}**",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in report.blockers) if report.blockers else lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--strict", action="store_true", help="return non-zero when the worktree is unsafe")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = inspect(repo)
    default_dir = repo / ".conductor" / "local"
    output_json = args.output_json or default_dir / "git_safety.json"
    output_md = args.output_md or default_dir / "git_safety.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(to_markdown(report), encoding="utf-8")
    print(output_md)
    return 2 if args.strict and not report.safe else 0


if __name__ == "__main__":
    raise SystemExit(main())
