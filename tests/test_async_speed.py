from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def _extract_oid(response: Any) -> Optional[int]:
    if not isinstance(response, dict):
        return None
    payload = response.get("response") or response
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    statuses = data.get("statuses")
    if not isinstance(statuses, list):
        return None
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


async def _warm_manager_session(execution: ExecutionManager, coin: str) -> None:
    await execution._ensure_session()
    session = execution._session
    if session is None:
        raise RuntimeError("Execution session is not initialized.")
    url = f"{execution.base_url.rstrip('/')}/info"
    payload = {"type": "l2Book", "coin": coin.upper()}
    headers = {"Content-Type": "application/json"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            await resp.text()
    except Exception as exc:  # pragma: no cover - network warm-up best effort
        print(f"⚠️ Warm-up request failed: {exc}")


@pytest.mark.integration
def test_async_speed() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    # Use _run_async for the warm-up coroutine
    execution._run_async(_warm_manager_session(execution, "HYPE"))
    print("🔥 Connection Warmed Up.")

    mids = connector.info.all_mids()
    spot_asset_id = connector.get_spot_asset_id("HYPE")
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = max(round(11.0 / spot_price, 2), 0.01)

    client_fire_epoch_ms = int(time.time() * 1000)
    t0 = time.perf_counter()
    
    # execute_entry_parallel is async, wrap it
    entry_ok = execution._run_async(
        execution.execute_entry_parallel(size, spot_price, perp_price, spot_asset_id)
    )
    
    t1 = time.perf_counter()
    assert entry_ok, "Parallel entry failed."

    client_execution_ms = (t1 - t0) * 1000
    print(f"Client Execution Time: {client_execution_ms:.2f} ms")

    spot_resp = execution.last_responses.get("spot")
    perp_resp = execution.last_responses.get("perp")
    spot_oid = _extract_oid(spot_resp)
    perp_oid = _extract_oid(perp_resp)

    if spot_oid and perp_oid:
        try:
            user = connector.master_address
            spot_details = connector.info.query_order_by_oid(user, spot_oid)
            perp_details = connector.info.query_order_by_oid(user, perp_oid)
            spot_server_time = spot_details.get("time")
            perp_server_time = perp_details.get("time")
            if spot_server_time and perp_server_time:
                spot_one_way = spot_server_time - client_fire_epoch_ms
                perp_one_way = perp_server_time - client_fire_epoch_ms
                print("=== SERVER TIMESTAMPS ===")
                print(f"Spot Order  OID {spot_oid}: {spot_server_time} ms (Δ {spot_one_way} ms)")
                print(f"Perp Order  OID {perp_oid}: {perp_server_time} ms (Δ {perp_one_way} ms)")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"⚠️ Unable to fetch server timestamps: {exc}")

    # execute_exit_parallel is async, wrap it
    execution._run_async(
        execution.execute_exit_parallel(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")
    )
