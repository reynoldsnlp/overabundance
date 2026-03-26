#!/usr/bin/env python3
"""Rewrite git history so docs/ has no prior history.

This script is intentionally destructive. It removes the chosen docs directory
from every existing commit, then restores the current working-tree snapshot of
that directory as one new commit so GitHub Pages can still publish it.

What it does:
1. Verifies that the repo is clean outside the docs directory.
2. Creates a full-repo backup bundle outside the repository.
3. Copies the current docs snapshot to a temporary backup.
4. Runs `git filter-repo --path <docs_dir> --invert-paths --force`.
5. Restores the docs snapshot and commits it once.

Requirements:
- `git filter-repo` must be installed and available as `git filter-repo`.

Typical usage:
    python scripts/rewrite_docs_history.py --yes
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(cmd: list[str], *, cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def repo_root() -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), capture_output=True)
    return Path(proc.stdout.strip()).resolve()


def posix_rel(path: Path, *, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def is_under_docs(path_text: str, docs_rel: str) -> bool:
    path_text = path_text.strip()
    if not path_text:
        return False
    return path_text == docs_rel or path_text.startswith(docs_rel + "/")


def non_docs_status_lines(root: Path, docs_rel: str) -> list[str]:
    proc = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
    )
    bad: list[str] = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line:
            continue
        path_part = raw_line[3:]
        candidates = path_part.split(" -> ") if " -> " in path_part else [path_part]
        if all(is_under_docs(candidate, docs_rel) for candidate in candidates):
            continue
        bad.append(raw_line)
    return bad


def ensure_filter_repo(root: Path) -> None:
    try:
        run(["git", "filter-repo", "--version"], cwd=root, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "git filter-repo is required but was not found.\n"
            "Install it first, for example with `brew install git-filter-repo`."
        ) from exc


def backup_bundle_path(root: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = root.parent / f"{root.name}-history-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir / f"{root.name}-pre-docs-rewrite-{timestamp}.bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Repository-relative docs directory whose history should be collapsed.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add current docs snapshot after stripping docs history",
        help="Commit message for the restored docs snapshot.",
    )
    parser.add_argument(
        "--backup-bundle",
        default="",
        help="Optional path for the pre-rewrite git bundle backup.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    ensure_filter_repo(root)

    docs_path = (root / args.docs_dir).resolve()
    if not docs_path.exists() or not docs_path.is_dir():
        raise SystemExit(f"Docs directory not found: {docs_path}")
    try:
        docs_rel = posix_rel(docs_path, start=root)
    except ValueError as exc:
        raise SystemExit(f"Docs directory must live inside the repository: {docs_path}") from exc

    dirty = non_docs_status_lines(root, docs_rel)
    if dirty:
        details = "\n".join(dirty[:20])
        extra = "" if len(dirty) <= 20 else f"\n... plus {len(dirty) - 20} more"
        raise SystemExit(
            "Refusing to rewrite history with non-docs working-tree changes.\n"
            f"Clean or stash these first:\n{details}{extra}"
        )

    bundle_path = Path(args.backup_bundle).expanduser().resolve() if args.backup_bundle else backup_bundle_path(root)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {root}")
    print(f"Docs directory: {docs_rel}")
    print(f"Backup bundle: {bundle_path}")
    print("This will rewrite git history for every branch and tag.")
    print("After it finishes, you will need to force-push rewritten refs.")

    if not args.yes:
        prompt = input("Type REWRITE to continue: ").strip()
        if prompt != "REWRITE":
            raise SystemExit("Aborted.")

    with tempfile.TemporaryDirectory(prefix="rewrite-docs-history-") as tmp_dir:
        snapshot_dir = Path(tmp_dir) / "docs_snapshot"
        shutil.copytree(docs_path, snapshot_dir, symlinks=True)

        run(["git", "bundle", "create", str(bundle_path), "--all"], cwd=root)
        print(f"Created backup bundle: {bundle_path}")

        run(
            ["git", "filter-repo", "--path", docs_rel, "--invert-paths", "--force"],
            cwd=root,
        )
        print(f"Removed {docs_rel} from prior history.")

        restored_docs = root / docs_rel
        if restored_docs.exists():
            shutil.rmtree(restored_docs)
        restored_docs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot_dir, restored_docs, symlinks=True)

        run(["git", "add", "--all", "--", docs_rel], cwd=root)
        run(["git", "commit", "-m", args.commit_message], cwd=root)

    print("Done.")
    print(f"Backup bundle kept at: {bundle_path}")
    print("Next steps:")
    print("  1. Verify the rewritten history and the restored docs snapshot.")
    print("  2. Force-push the rewritten refs, for example:")
    print("     git push --force-with-lease --all")
    print("     git push --force-with-lease --tags")
    print("  3. Tell collaborators to re-clone or hard-reset to the rewritten history.")


if __name__ == "__main__":
    main()
