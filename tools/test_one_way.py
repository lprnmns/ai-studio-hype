from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def next_full_minute() -> dt.datetime:
    now = dt.datetime.utcnow()
    target = (now.replace(second=0, microsecond=0) + dt.timedelta(minutes=1))
    return target


def busy_wait(target_ts: float) -> None:
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            break
        if remaining > 0.01:
            time.sleep(remaining - 0.005)
        else:
            pass  # busy wait last milliseconds


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    spot_asset_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    target = next_full_minute()
    target_ts = target.timestamp()
    print(f"⏳ Waiting for target time: {target.strftime('%H:%M:%S.000')} UTC")
    busy_wait(target_ts)

    fire_time = time.time()
    entry_ok = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
    if not entry_ok:
        print("Entry failed.")
        return

    report = execution.last_report if hasattr(execution, "last_report") else {}
    order_ids = []
    for status in report.get("statuses", []):
        if status.get("status") == "ok" and "oid" in status:
            order_ids.append(status["oid"])

    if not order_ids:
        print("Could not retrieve order ids from response.")
        return

    user = connector.master_address
    oid = order_ids[0]
    print(f"Order OID: {oid}")
    order_details = connector.info.query_order_by_oid(user, oid)
    exchange_ts_ms = order_details.get("time")
    if exchange_ts_ms is None:
        print(f"Order details: {order_details}")
        print("Missing exchange timestamp.")
        return

    diff_ms = exchange_ts_ms - int(target_ts * 1000)
    print(f"Local fire time: {fire_time:.3f}")
    print(f"Exchange timestamp: {exchange_ts_ms}")
    print(f"One-way latency (target -> exchange): {diff_ms} ms")

    execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")


if __name__ == "__main__":
    main()

