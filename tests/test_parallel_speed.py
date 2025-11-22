from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pytest

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def _extract_oid(response: Any) -> Optional[int]:
    if not isinstance(response, dict):
        return None
    payload = response.get("response") or response
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    statuses = data.get("statuses", [])
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


@pytest.mark.integration
def test_parallel_speed():
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    local_now = int(time.time() * 1000)
    server_time = None
    if hasattr(connector.info, "time"):
        try:
            server_time = connector.info.time()
        except Exception:
            server_time = None
    print("\n=== CLOCK SYNC ===")
    print(f"Client epoch ms : {local_now}")
    if server_time:
        print(f"Server epoch ms : {server_time}")
        print(f"Drift           : {server_time - local_now} ms")
    else:
        print("Server time API unavailable.")

    mids = connector.info.all_mids()
    spot_asset_id = connector.get_spot_asset_id("HYPE")
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(11.0 / spot_price, 2)

    t_send_start = int(time.time() * 1000)
    entry_ok = execution.execute_entry_parallel(size, spot_price, perp_price, spot_asset_id)
    t_send_end = int(time.time() * 1000)
    assert entry_ok, "Parallel entry failed."

    spot_resp = execution.last_responses.get("spot")
    perp_resp = execution.last_responses.get("perp")
    spot_oid = _extract_oid(spot_resp)
    perp_oid = _extract_oid(perp_resp)
    assert spot_oid and perp_oid, "Missing order IDs for latency analysis."

    user = connector.master_address
    spot_details = connector.info.query_order_by_oid(user, spot_oid)
    perp_details = connector.info.query_order_by_oid(user, perp_oid)
    spot_server_time = spot_details.get("time")
    perp_server_time = perp_details.get("time")
    assert spot_server_time and perp_server_time, "Missing server timestamps."

    gap_ms = abs(spot_server_time - perp_server_time)
    spot_one_way = spot_server_time - t_send_start
    perp_one_way = perp_server_time - t_send_start

    print("\n=== SPEED REPORT ===")
    print(f"Client Send Time : {t_send_start}")
    print(f"Spot Server Time : {spot_server_time}")
    print(f"Perp Server Time : {perp_server_time}")
    print("------------------------------------")
    print(f"🚀 EXECUTION GAP : {gap_ms} ms")
    print(f"📡 Spot One-Way  : {spot_one_way} ms")
    print(f"📡 Perp One-Way  : {perp_one_way} ms")
    print(f"Client RTT (Total): {(t_send_end - t_send_start)} ms")
    print("------------------------------------")

    execution.execute_exit_parallel(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")

