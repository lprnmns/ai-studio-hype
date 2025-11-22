from __future__ import annotations

import subprocess
from typing import Tuple


def run_command(cmd: list[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 1, f"Command not found: {' '.join(cmd)}"


def parse_ping(output: str) -> str:
    for line in output.splitlines():
        if "Minimum" in line and "Maximum" in line and "Average" in line:
            return line.strip()
    return "Ping summary not found."


def ping_test() -> None:
    print("=== ICMP PING TEST (Windows) ===")
    code, output = run_command(["ping", "-n", "4", "api.hyperliquid.xyz"])
    if code != 0:
        print("Ping failed:")
        print(output)
        return
    print(output)
    summary = parse_ping(output)
    print(f"Summary: {summary}\n")


def curl_timing() -> None:
    print("=== HTTPS LATENCY (curl) ===")
    format_str = (
        r"time_namelookup: %{time_namelookup}\n"
        r"time_connect: %{time_connect}\n"
        r"time_appconnect: %{time_appconnect}\n"
        r"time_starttransfer: %{time_starttransfer}\n"
        r"time_total: %{time_total}\n"
    )
    code, output = run_command(
        [
            "curl",
            "-o",
            "NUL",
            "-s",
            "-w",
            format_str,
            "https://api.hyperliquid.xyz/info",
        ]
    )
    if code != 0:
        print("curl failed:")
        print(output)
        return
    print(output)


def cpu_load() -> None:
    print("=== CPU LOAD (Windows) ===")
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage (1s avg): {cpu_percent:.2f}%\n")
    except ImportError:
        code, output = run_command(
            ["wmic", "cpu", "get", "loadpercentage", "/value"]
        )
        if code != 0:
            print("Unable to determine CPU load.")
            print(output + "\n")
        else:
            print(output.strip() + "\n")


def main() -> None:
    ping_test()
    curl_timing()
    cpu_load()


if __name__ == "__main__":
    main()

