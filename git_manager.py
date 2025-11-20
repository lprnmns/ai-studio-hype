from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(command: list[str]) -> None:
    """Run a git command and surface errors immediately."""
    result = subprocess.run(command, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def build_authenticated_url(repo_url: str, token: str) -> str:
    if not repo_url.startswith("https://"):
        raise ValueError("GITHUB_REPO_URL must use https://")
    return repo_url.replace("https://", f"https://{token}@", 1)


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Automate git add/commit/push")
    parser.add_argument("message", nargs="+", help="Commit message")
    args = parser.parse_args()
    return " ".join(args.message)


def main() -> int:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    commit_message = parse_args()
    github_token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO_URL")

    if not github_token:
        print("Missing GITHUB_TOKEN in .env", file=sys.stderr)
        return 1
    if not repo_url:
        print("Missing GITHUB_REPO_URL in .env", file=sys.stderr)
        return 1

    auth_url = build_authenticated_url(repo_url, github_token)

    try:
        run_cmd(["git", "add", "."])
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=REPO_ROOT,
            text=True,
        )
        if commit_result.returncode != 0:
            print("git commit failed; aborting push.", file=sys.stderr)
            return commit_result.returncode
        run_cmd(["git", "push", auth_url, "HEAD"])
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

