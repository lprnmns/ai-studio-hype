from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def extract_oid(response: Any) -> Optional[int]:
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


def extract_server_time(details: Dict[str, Any]) -> Optional[int]:
    if not details:
        return None
    if "time" in details:
        return details["time"]
    order_root = details.get("order")
    if isinstance(order_root, dict):
        if "time" in order_root:
            return order_root["time"]
        inner = order_root.get("order")
        if isinstance(inner, dict) and "timestamp" in inner:
            return inner["timestamp"]
    return None


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    spot_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_px = float(mids[spot_id])
    perp_px = float(mids["HYPE"])
    size = round(11.0 / spot_px, 2)

    print(f"\n🔥 Parallel Latency Test: {size} HYPE (≈${size * spot_px:.2f})")
    t_client_start_ns = time.time_ns()
    success = execution.execute_entry_parallel(size, spot_px, perp_px, spot_id)
    t_client_end_ns = time.time_ns()

    if not success:
        print("⚠️ Entry reported mismatch; continuing with timestamp analysis...")

    spot_resp = execution.last_responses.get("spot", {})
    perp_resp = execution.last_responses.get("perp", {})
    spot_oid = extract_oid(spot_resp)
    perp_oid = extract_oid(perp_resp)

    if not spot_oid or not perp_oid:
        print("❌ Could not extract order IDs from responses.")
        print("Spot resp:", spot_resp)
        print("Perp resp:", perp_resp)
        return

    user = connector.master_address
    spot_details = connector.info.query_order_by_oid(user, spot_oid)
    perp_details = connector.info.query_order_by_oid(user, perp_oid)
    spot_server_time = extract_server_time(spot_details)
    perp_server_time = extract_server_time(perp_details)

    if spot_server_time is None or perp_server_time is None:
        print("❌ Could not extract server timestamps.")
        print("Spot details:", spot_details)
        print("Perp details:", perp_details)
        return

    t_client_start_ms = t_client_start_ns / 1_000_000
    spot_one_way = spot_server_time - t_client_start_ms
    perp_one_way = perp_server_time - t_client_start_ms
    gap_ms = abs(spot_server_time - perp_server_time)

    print("\n=== PARALLEL LATENCY REPORT ===")
    print(f"Client send ms  : {t_client_start_ms:.3f}")
    print(f"Spot server ms  : {spot_server_time}")
    print(f"Perp server ms  : {perp_server_time}")
    print("--------------------------------")
    print(f"📡 Spot one-way  : {spot_one_way:.2f} ms")
    print(f"📡 Perp one-way  : {perp_one_way:.2f} ms")
    print(f"🚀 Execution gap : {gap_ms} ms")
    print(f"Client RTT       : {(t_client_end_ns - t_client_start_ns)/1_000_000:.2f} ms")
    print("================================")

    print("🧹 Closing positions...")
    execution.execute_exit_parallel(size, spot_px, perp_px, spot_id, "HYPE")


if __name__ == "__main__":
    main()

