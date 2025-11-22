from __future__ import annotations

import os
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
        if "min/avg/max" in line or "min/avg/max/mdev" in line:
            return line.strip()
    return "Ping summary not found."


def ping_test() -> None:
    print("=== ICMP PING TEST ===")
    code, output = run_command(["ping", "-c", "4", "api.hyperliquid.xyz"])
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
            "/dev/null",
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
    print("=== CPU LOAD ===")
    try:
        load = os.getloadavg()
        print(f"Load averages (1m, 5m, 15m): {load}\n")
    except (AttributeError, OSError):
        print("getloadavg not available on this system.\n")


def main() -> None:
    ping_test()
    curl_timing()
    cpu_load()


if __name__ == "__main__":
    main()

