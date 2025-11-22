from __future__ import annotations

import os
import statistics
import time

from eth_account import Account
from hyperliquid.utils.signing import sign_l1_action


def benchmark_signing(iterations: int = 100) -> None:
    private_key = os.getenv("HL_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("HL_PRIVATE_KEY must be set in the environment for benchmarking.")

    wallet = Account.from_key(private_key)
    dummy_action = {
        "type": "order",
        "orders": [
            {
                "coin": "HYPE",
                "is_buy": True,
                "limit_px": 35.0,
                "sz": 0.3,
                "order_type": {"limit": {"tif": "Ioc"}},
            }
        ],
    }

    timings_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        sign_l1_action(
            wallet,
            dummy_action,
            None,  # vault_address
            int(time.time() * 1000),
            None,  # expires_after
            True,
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        timings_ms.append(elapsed_ms)

    print("=== SIGNING BENCHMARK ===")
    print(f"Iterations: {iterations}")
    print(f"Min: {min(timings_ms):.4f} ms")
    print(f"Max: {max(timings_ms):.4f} ms")
    print(f"Avg: {statistics.mean(timings_ms):.4f} ms")


if __name__ == "__main__":
    benchmark_signing()

