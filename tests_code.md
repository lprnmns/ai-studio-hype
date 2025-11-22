# Hyperliquid Arbitrage Bot - Tests

This document contains the test suite for verifying the bot's functionality, connectivity, and performance.

## 1. `tests/test_step2_connection.py`
Verifies basic API connectivity and account access.

```python
import pytest
import os
from dotenv import load_dotenv
from src.connector import HyperliquidConnector

load_dotenv()

def test_env_vars():
    assert os.getenv("HL_PRIVATE_KEY") is not None
    assert os.getenv("HL_MASTER_ADDRESS") is not None

def test_connection():
    connector = HyperliquidConnector()
    assert connector.agent_account is not None
    print(f"Connected as: {connector.agent_account.address}")

def test_spot_asset_resolution():
    connector = HyperliquidConnector()
    asset_id = connector.get_spot_asset_id("HYPE")
    assert asset_id.startswith("@")
    print(f"HYPE Asset ID: {asset_id}")
```

## 2. `tests/test_step3_stream.py`
Verifies WebSocket data streaming.

```python
import pytest
import asyncio
from src.connector import HyperliquidConnector
from src.bot import ArbitrageBot

@pytest.mark.asyncio
async def test_market_data_stream():
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector, symbol="HYPE")
    
    # Start the bot in a task
    task = asyncio.create_task(bot.run())
    
    # Wait for data
    for _ in range(10):
        await asyncio.sleep(1)
        if bot.spot_price > 0 and bot.perp_price > 0:
            break
            
    print(f"Spot: {bot.spot_price}, Perp: {bot.perp_price}")
    
    assert bot.spot_price > 0
    assert bot.perp_price > 0
    
    bot.should_run = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

## 3. `tests/test_step4_live_trade.py`
Performs a real (small size) entry and exit to verify execution logic.

```python
import pytest
import os
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

@pytest.mark.asyncio
async def test_step4_live_trade():
    _require_env_vars()
    
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)
    execution.setup_account()
    
    # 1. Get Data
    symbol = "HYPE"
    spot_asset_id = connector.get_spot_asset_id(symbol)
    snapshot = connector.info.l2_snapshot(name=symbol)
    spot_price = float(snapshot["levels"][0][0]["px"])
    perp_price = spot_price
    
    # 2. Calc Size
    TEST_SIZE_USD = 11.0
    size = _calc_size(TEST_SIZE_USD, spot_price)
    print(f"Target size: {size} {symbol} (Price ~{spot_price})")
    
    # 3. ENTRY (Parallel Async)
    print("Executing Entry...")
    entry_ok = await execution.execute_entry_parallel(size, spot_price, perp_price, spot_asset_id)
    assert entry_ok, "Entry failed"
    
    print("Entry Successful. Waiting 5 seconds...")
    await asyncio.sleep(5)
    
    # 4. EXIT (Parallel Async)
    print("Executing Exit...")
    exit_ok = await execution.execute_exit_parallel(size, spot_price, perp_price, spot_asset_id, symbol)
    assert exit_ok, "Exit failed"
    
    print("Test Complete.")
```

## 4. `tests/benchmark_performance.py`
Measures the bot's "Ticks Per Second" (TPS) processing capability.

```python
from __future__ import annotations

import asyncio
import contextlib
import time
import sys

# Fix path
sys.path.append(".")

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector

async def run_benchmark(duration_seconds: int = 60) -> None:
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector)

    print(f"🚀 Benchmark Started ({duration_seconds}s)...")
    ws_task = asyncio.create_task(bot.run())
    
    while bot.total_spread_calculations == 0:
        await asyncio.sleep(0.1)
        
    print("✅ Stream active!")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and elapsed > 1:
                tps = bot.total_ticks / elapsed
                print(f"   Elapsed: {int(elapsed)}s | TPS: {tps:.2f}")
            await asyncio.sleep(1)
            
    finally:
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task

    end_time = time.time()
    elapsed = end_time - start_time
    tps = bot.total_ticks / elapsed if elapsed > 0 else 0
    
    print("\n📊 RESULTS")
    print(f"   Total Time: {elapsed:.2f}s")
    print(f"   Total Ticks: {bot.total_ticks}")
    print(f"   TPS: {tps:.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmark(300))
```

## 5. `tools/test_profiling.py`
Detailed breakdown of latency components (Logic, CPU, Network RTT).

```python
from __future__ import annotations
import asyncio
import time
import traceback
import sys
import aiohttp
from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

class ProfilingExecutionManager(ExecutionManager):
    def __init__(self, connector):
        super().__init__(connector)
        self.timings = {}

    async def run_heartbeat(self) -> None:
        """Aggressive 0.5s heartbeat keeping 5 connections warm."""
        print("💓 Heartbeat started (0.5s interval, 5 conns)...")
        while True:
            await asyncio.sleep(0.5)
            if self._session and not self._session.closed:
                try:
                    t0 = time.perf_counter()
                    url = "/info"
                    payload = {"type": "clearinghouseState", "user": self.master_address}
                    tasks = [self._session.post(url, json=payload) for _ in range(5)]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    for resp in responses:
                        if not isinstance(resp, Exception):
                            await resp.read()
                    t1 = time.perf_counter()
                except Exception as e:
                    print(f"⚠️ Heartbeat failed: {e}")

    async def execute_entry_parallel(self, size, spot_price, perp_price, spot_asset_id):
        self.timings = {}
        t0 = time.perf_counter()
        
        # Logic
        request_size = self._round_size_down(size)
        spot_coin = self._spot_coin_from_asset_id(spot_asset_id)
        spot_limit = self._round_to_sig_figs(spot_price * 1.05)
        perp_limit = self._round_to_sig_figs(perp_price * 0.95)
        
        t1 = time.perf_counter()
        self.timings["Logic/Rounding"] = (t1 - t0) * 1000

        # Payload
        t2_start = time.perf_counter()
        spot_payload = self._sdk_build_payload(coin_name=spot_coin, is_buy=True, size=request_size, price=spot_limit, reduce_only=False)
        t2_end = time.perf_counter()
        self.timings["Build Payload (Spot)"] = (t2_end - t2_start) * 1000

        t3_start = time.perf_counter()
        perp_payload = self._sdk_build_payload(coin_name=self.symbol, is_buy=False, size=request_size, price=perp_limit, reduce_only=False)
        t3_end = time.perf_counter()
        self.timings["Build Payload (Perp)"] = (t3_end - t3_start) * 1000

        # Network
        t4_start = time.perf_counter()
        responses = await self._post_parallel_payloads([("spot", spot_payload), ("perp", perp_payload)])
        t4_end = time.perf_counter()
        self.timings["Network (POST RTT)"] = (t4_end - t4_start) * 1000
        
        spot_ok = self._is_success(responses["spot"])
        perp_ok = self._is_success(responses["perp"])
        return spot_ok and perp_ok

# ... (LatencyTestBot class and main omitted for brevity, see file) ...
```
