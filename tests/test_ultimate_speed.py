from __future__ import annotations

import time

import pytest

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


@pytest.mark.integration
def test_ultimate_speed():
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    execution.setup_account()
    spot_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_px = float(mids[spot_id])
    perp_px = float(mids["HYPE"])
    size = round(11.0 / spot_px, 2)

    print(f"\n🔥 TEST BAŞLIYOR: {size} HYPE (Approx ${size*spot_px:.2f})")
    print(f"📍 Lokasyon: DigitalOcean -> Hyperliquid")

    t_client_start_ns = time.time_ns()
    print("🚀 Emirler Gönderiliyor...")
    success = execution.execute_entry_parallel(size, spot_px, perp_px, spot_id)
    t_client_end_ns = time.time_ns()

    print("🔍 API'den Gerçek Zamanlar Sorgulanıyor...")
    user = connector.master_address
    time.sleep(1)
    spot_resp = execution.last_responses.get("spot", {})
    perp_resp = execution.last_responses.get("perp", {})

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

    spot_oid = _extract_oid(spot_resp)
    perp_oid = _extract_oid(perp_resp)
    if not spot_oid or not perp_oid:
        print("❌ Order ID'leri alınamadı.")
        return

    def _extract_server_time(details: Dict[str, Any]) -> Optional[int]:
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

    spot_details = connector.info.query_order_by_oid(user, spot_oid)
    perp_details = connector.info.query_order_by_oid(user, perp_oid)
    t_spot = _extract_server_time(spot_details)
    t_perp = _extract_server_time(perp_details)
    if t_spot is None or t_perp is None:
        print("❌ Sunucu timestamp'leri alınamadı.")
        print(f"Spot details: {spot_details}")
        print(f"Perp details: {perp_details}")
        return
    t_client_start_ms = t_client_start_ns / 1_000_000

    lat_spot = t_spot - t_client_start_ms
    lat_perp = t_perp - t_client_start_ms
    gap = abs(t_spot - t_perp)

    print("\n" + "=" * 40)
    print("📊 HIZ RAPORU (ONE-WAY LATENCY)")
    print("=" * 40)
    print(f"⏱️  Client Start:  {t_client_start_ms:.3f}")
    print(f"🏁 Spot Server:   {t_spot}")
    print(f"🏁 Perp Server:   {t_perp}")
    print("-" * 40)
    print(f"📡 SPOT Gecikmesi: {lat_spot:.2f} ms")
    print(f"📡 PERP Gecikmesi: {lat_perp:.2f} ms")
    print("-" * 40)
    print(f"🚀 EXECUTION GAP:  {gap} ms")
    print("=" * 40)

    print("🧹 Pozisyonlar Kapatılıyor...")
    execution.execute_exit_parallel(size, spot_px, perp_px, spot_id, "HYPE")

