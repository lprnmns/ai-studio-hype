from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SERVER_IP = "24.144.105.173"
SERVER_USER = "root"
SSH_KEY = "ssh.key"
ARCHIVE_NAME = "hyperliquid_bot.zip"


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True)


def ensure_key_exists(key_path: Path) -> None:
    if not key_path.exists():
        print(f"❌ SSH key not found at {key_path}")
        sys.exit(1)


def create_archive(project_root: Path) -> None:
    print("📦 ZIPPING PROJECT...")
    archive_path = project_root / ARCHIVE_NAME
    if archive_path.exists():
        archive_path.unlink()

    include_paths = ["src", "tests", "tools", "requirements.txt", ".env"]
    exclude_dirs = {".git", "logs", "venv", "__pycache__"}
    exclude_suffixes = {".pyc"}

    def should_skip(path: Path) -> bool:
        parts = set(path.parts)
        if parts & exclude_dirs:
            return True
        if path.suffix in exclude_suffixes:
            return True
        return False

    with ZipFile(archive_path, "w", ZIP_DEFLATED) as zipf:
        for item in include_paths:
            full_path = project_root / item
            if not full_path.exists():
                continue
            if full_path.is_file():
                if not should_skip(full_path):
                    zipf.write(full_path, full_path.relative_to(project_root))
                continue
            for file_path in full_path.rglob("*"):
                if file_path.is_dir():
                    continue
                if should_skip(file_path):
                    continue
                zipf.write(file_path, file_path.relative_to(project_root))
    print(f"✅ Archive created at {archive_path}")


def upload_archive(project_root: Path, key_path: Path) -> None:
    print("🚀 UPLOADING ARCHIVE...")
    cmd = [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        str(project_root / ARCHIVE_NAME),
        f"{SERVER_USER}@{SERVER_IP}:~/",
    ]
    run_command(cmd)
    print("✅ Upload complete.")


def run_remote_commands(key_path: Path) -> None:
    print("☁️ RUNNING REMOTE TEST...")
    remote_cmd = " && ".join(
        [
            "sudo apt-get update",
            "sudo apt-get install -y unzip python3-pip python3-venv",
            "unzip -o hyperliquid_bot.zip",
            "python3 -m venv hyperliquid_env",
            "hyperliquid_env/bin/pip install --upgrade pip",
            "hyperliquid_env/bin/pip install -r requirements.txt",
            "hyperliquid_env/bin/pip install uvloop pytest",
            "hyperliquid_env/bin/python -m pytest tests/test_latency_breakdown.py -s",
        ]
    )
    ssh_cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        f"{SERVER_USER}@{SERVER_IP}",
        remote_cmd,
    ]
    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print("❌ Remote command failed.")
        sys.exit(process.returncode)
    print("✅ Remote latency test completed.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    key_path = project_root / SSH_KEY
    ensure_key_exists(key_path)
    create_archive(project_root)
    upload_archive(project_root, key_path)
    run_remote_commands(key_path)


if __name__ == "__main__":
    main()

