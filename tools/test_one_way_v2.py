from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict, List, Optional

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def next_full_minute() -> dt.datetime:
    now = dt.datetime.utcnow()
    return now.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)


def busy_wait(target_ts: float) -> None:
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            break
        if remaining > 0.01:
            time.sleep(remaining - 0.005)
        else:
            pass


def extract_oid(response: Dict[str, Any]) -> Optional[int]:
    statuses: List[Dict[str, Any]] = response.get("statuses", [])
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled") or status.get("resting")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    target_dt = next_full_minute()
    target_ts = target_dt.timestamp()
    print(f"⏳ Waiting for target time: {target_dt.strftime('%H:%M:%S.000')} UTC")
    busy_wait(target_ts)

    mids = connector.info.all_mids()
    spot_asset_id = connector.get_spot_asset_id("HYPE")
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    fire_time = time.time()
    entry_response = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
    if not entry_response:
        print("Entry failed.")
        return

    if isinstance(entry_response, dict):
        oid = extract_oid(entry_response)
    else:
        oid = None

    if not oid:
        print("Could not extract order id from response.")
        print(entry_response)
        return

    user = connector.master_address
    order_details = connector.info.query_order_by_oid(user, oid)
    exchange_ts_ms = order_details.get("time")

    if exchange_ts_ms is None:
        print("Missing exchange timestamp.")
    else:
        diff_ms = exchange_ts_ms - int(target_ts * 1000)
        print(f"Local fire time: {fire_time:.3f}")
        print(f"Exchange timestamp: {exchange_ts_ms}")
        print(f"One-way latency: {diff_ms} ms")

    execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")


if __name__ == "__main__":
    main()

