from __future__ import annotations

import time

import pytest

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


class TimedExecutionManager(ExecutionManager):
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
def test_latency_breakdown():
    connector = HyperliquidConnector()
    execution = TimedExecutionManager(connector)

    spot_asset_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    try:
        success = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
        assert success, "Entry failed during latency test."
    finally:
        execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")

    report = execution.last_report
    if not report:
        pytest.fail("Latency report missing.")

    def delta(ms_a: str, ms_b: str) -> float:
        return report[ms_b] - report[ms_a]

    logic_ms = report["T1"] - report["T0"]
    spot_rtt = delta("T2", "T3")
    processing_gap = report["T4"] - report["T3"]
    perp_rtt = delta("T4", "T5")
    total_time = report["T5"]  # already offset from T0

    print("\n=== LATENCY BREAKDOWN ===")
    print(f"1. Logic/Calc Overhead:  {logic_ms:8.2f} ms")
    print(f"2. Spot Order RTT:       {spot_rtt:8.2f} ms  (Network)")
    print(f"3. Processing Gap:       {processing_gap:8.2f} ms")
    print(f"4. Perp Order RTT:       {perp_rtt:8.2f} ms  (Network)")
    print("---------------------------------")
    print(f"TOTAL EXECUTION TIME:    {total_time:8.2f} ms")
    print("==================================")

