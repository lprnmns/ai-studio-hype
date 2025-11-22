from __future__ import annotations

import time

import pytest

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


class WarmTimedExecutionManager(ExecutionManager):
    def __init__(self, connector: HyperliquidConnector):
        super().__init__(connector)
        self.last_report: dict[str, float] = {}

    def execute_entry_ioc(
        self,
        size: float,
        spot_price: float,
        perp_price: float,
        spot_asset_id: str,
    ) -> bool:
        timestamps: dict[str, float] = {}

        def mark(label: str) -> None:
            timestamps[label] = time.perf_counter_ns()

        mark("T0")
        request_size = self._round_size_down(size)
        spot_limit = self._round_to_sig_figs(spot_price * 1.02)
        perp_limit = self._round_to_sig_figs(perp_price * 0.98)
        mark("T1")

        mark("T2")
        spot_resp = self._submit_order(
            coin=spot_asset_id,
            is_buy=True,
            size=request_size,
            price=spot_limit,
            tif="Ioc",
            reduce_only=False,
        )
        mark("T3")

        filled_size = self._extract_filled_size(spot_resp)
        if filled_size <= 0:
            self.last_report = {k: (v - timestamps["T0"]) / 1e6 for k, v in timestamps.items()}
            return False

        hedge_size = self._round_size_down(filled_size)
        if hedge_size <= 0:
            self.last_report = {k: (v - timestamps["T0"]) / 1e6 for k, v in timestamps.items()}
            return False

        mark("T4")
        perp_resp = self._submit_order(
            coin=self.symbol,
            is_buy=False,
            size=hedge_size,
            price=perp_limit,
            tif="Ioc",
            reduce_only=False,
        )
        mark("T5")

        self.last_report = {k: (v - timestamps["T0"]) / 1e6 for k, v in timestamps.items()}
        return self._is_success(perp_resp)


@pytest.mark.integration
def test_warm_latency():
    connector = HyperliquidConnector()
    execution = WarmTimedExecutionManager(connector)

    print("🔥 Warming up connection...")
    warm_times = []
    for i in range(5):
        start = time.perf_counter()
        data = connector.info.l2_snapshot("HYPE")
        elapsed = (time.perf_counter() - start) * 1000
        warm_times.append(elapsed)
        print(f"Warm-up {i + 1}: {elapsed:.2f} ms (levels: {len(data.get('levels', []))})")

    spot_asset_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    try:
        success = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
        assert success, "Hot entry failed."
    finally:
        execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")

    report = execution.last_report
    if not report:
        pytest.fail("Missing latency report.")

    def delta(a: str, b: str) -> float:
        return report[b] - report[a]

    logic_ms = report["T1"] - report["T0"]
    spot_rtt = delta("T2", "T3")
    processing_gap = report["T4"] - report["T3"]
    perp_rtt = delta("T4", "T5")
    total_time = report["T5"]

    print("\n=== WARM-UP SUMMARY ===")
    print(f"First fetch: {warm_times[0]:.2f} ms")
    print(f"Last fetch : {warm_times[-1]:.2f} ms")

    print("\n=== HOT LATENCY BREAKDOWN ===")
    print(f"Logic/Calc Overhead : {logic_ms:8.2f} ms")
    print(f"Spot Order RTT      : {spot_rtt:8.2f} ms")
    print(f"Processing Gap      : {processing_gap:8.2f} ms")
    print(f"Perp Order RTT      : {perp_rtt:8.2f} ms")
    print("---------------------------------")
    print(f"TOTAL EXEC TIME     : {total_time:8.2f} ms")

