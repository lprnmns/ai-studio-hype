from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.connector import HyperliquidConnector
def _round_to_sig_figs(value: float, sig_figs: int = 5) -> float:
    if value == 0:
        return 0.0
    return float(f"{value:.{sig_figs}g}")


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
    exchange = connector.exchange

    spot_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_px = float(mids[spot_id])
    min_usd_value = 10.8  # min 10 USDC + small buffer
    usdc_balance = 0.0
    spot_state = connector.info.spot_user_state(connector.master_address)
    for bal in spot_state.get("balances", []):
        if bal.get("coin") == "USDC":
            usdc_balance = float(bal.get("total", 0.0))
            break
    max_affordable = usdc_balance * 0.8 / spot_px if spot_px > 0 else 0
    desired = 11.0 / spot_px
    min_size = min_usd_value / spot_px
    raw_size = min(max_affordable, desired)
    size = round(max(raw_size, min_size), 2)
    if size * spot_px < min_usd_value:
        print("❌ Not enough USDC balance (need at least ~10 USDC).")
        return

    print(f"🔥 Spot Latency Test: {size} HYPE (≈${size * spot_px:.2f})")
    t_client_start_ns = time.time_ns()
    response = exchange.order(
        name=spot_id,
        is_buy=True,
        sz=size,
        limit_px=_round_to_sig_figs(spot_px * 1.05),
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=False,
    )
    t_client_end_ns = time.time_ns()

    oid = extract_oid(response)
    if not oid:
        print("❌ Could not extract OID from response.")
        print(response)
        return

    server_time = None
    try:
        details = connector.info.query_order_by_oid(connector.master_address, oid)
        server_time = extract_server_time(details)
    except Exception:
        details = None

    if server_time is None:
        print("⚠️ query_order_by_oid returned no timestamp; falling back to fills.")
        time.sleep(1)
        fills = connector.info.user_fills(connector.master_address)
        for fill in fills:
            if float(fill.get("sz", 0)) == size and fill.get("coin") == spot_id:
                server_time = fill.get("time")
                break
        if server_time is None:
            print("❌ Could not extract server timestamp even from fills.")
            print("details:", details)
            return

    t_client_start_ms = t_client_start_ns / 1_000_000
    spot_one_way = server_time - t_client_start_ms

    print("\n=== SPOT LATENCY REPORT ===")
    print(f"Client send ms : {t_client_start_ms:.3f}")
    print(f"Exchange time  : {server_time}")
    print(f"One-way latency: {spot_one_way:.2f} ms")
    print(f"Client RTT     : {(t_client_end_ns - t_client_start_ns)/1_000_000:.2f} ms")


if __name__ == "__main__":
    main()

