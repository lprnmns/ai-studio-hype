# Project Tests

This document contains the full source code for the test suite (`tests/`).

### `tests/benchmark_performance.py`

```python
from __future__ import annotations

import asyncio
import contextlib
import time

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector


async def run_benchmark(duration_seconds: int = 300) -> None:
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector)

    ws_task = asyncio.create_task(bot.run())
    start = time.time()
    print(f"Benchmark started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    try:
        await asyncio.sleep(duration_seconds)
    finally:
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task
    end = time.time()

    total_ticks = bot.total_ticks
    total_spreads = bot.total_spread_calculations
    elapsed = end - start if end > start else 1
    tps = total_ticks / elapsed

    print(f"Benchmark ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Duration: {elapsed:.2f}s")
    print(f"Total ticks processed: {total_ticks}")
    print(f"Total spread calculations: {total_spreads}")
    print(f"Ticks per second (TPS): {tps:.2f}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

### `tests/test_async_speed.py`

```python
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
```

### `tests/test_latency_breakdown.py`

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

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
```

### `tests/test_parallel_speed.py`

```python
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
```

### `tests/test_step2_connection.py`

```python
from __future__ import annotations

from src.connector import HyperliquidConnector


def test_step2_connection():
    connector = HyperliquidConnector()

    print(f"✅ Agent Address: {connector.agent_address}")
    print(f"✅ Master Address: {connector.master_address}")

    hype_asset_id = connector.get_spot_asset_id("HYPE")
    print(f"✅ HYPE Asset ID: {hype_asset_id}")
    assert hype_asset_id is not None

    balances = connector.get_account_balance()
    print(f"✅ Total Equity: {balances['total_equity']}")
    print(f"✅ Spot USDC Balance: {balances['spot_usdc']}")
```

### `tests/test_step3_stream.py`

```python
from __future__ import annotations

import asyncio
import contextlib

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector


async def _wait_for_prices(bot: ArbitrageBot, timeout: float = 15.0) -> bool:
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < timeout:
        if bot.spot_best_ask > 0 and bot.perp_best_bid > 0:
            return True
        await asyncio.sleep(1)
    return False


def test_step3_stream():
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector)

    async def runner():
        task = asyncio.create_task(bot.run())
        try:
            assert await _wait_for_prices(bot), "Did not receive live L2 data for spot/perp"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(runner())
```

### `tests/test_step4_live_trade.py`

```python
from __future__ import annotations

import asyncio
import os
import time

import pytest
from dotenv import load_dotenv

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


load_dotenv()


def _require_env_vars():
    if not os.getenv("HL_PRIVATE_KEY") or not os.getenv("HL_MASTER_ADDRESS"):
        pytest.skip("Skipping live trade test: Missing HL_PRIVATE_KEY or HL_MASTER_ADDRESS")


def _calc_size(target_usd_size, price):
    size = target_usd_size / price
    return round(size, 2)


def test_step4_live_trade():
    _require_env_vars()
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    execution.setup_account()

    SPOT_ID_HARDCODED = "@107"

    print("📡 Fetching L2 Snapshot for HYPE (Lightweight)...")

    perp_snapshot = connector.info.l2_snapshot(name="HYPE")
    perp_price = float(perp_snapshot["levels"][0][0]["px"])

    spot_price = perp_price

    TEST_SIZE_USD = 11.0
    size = _calc_size(TEST_SIZE_USD, spot_price)
    print(f"📐 Target size: {size} HYPE (Price ~{spot_price})")

    entry_ok = execution.execute_entry_ioc(size, spot_price, perp_price, SPOT_ID_HARDCODED)
    assert entry_ok, "Entry IOC leg failed."

    print("✅ Entry Successful. Waiting 5 seconds...")
    time.sleep(5)

    exit_ok = execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, SPOT_ID_HARDCODED, symbol="HYPE")
    assert exit_ok, "Exit leg failed."
```

### `tests/test_ultimate_speed.py`

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

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
```

### `tests/test_warm_latency.py`

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

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
```

