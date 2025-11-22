from __future__ import annotations

import asyncio
import time
import sys
import logging

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

# Configuration
TEST_SIZE_USD = 11.0  # Safe size for your balance
SYMBOL = "HYPE"
WAIT_TIME_SECONDS = 10

class InstrumentedExecutionManager(ExecutionManager):
    """
    Custom ExecutionManager that measures individual request latencies
    without changing the core logic.
    """
    async def _post_parallel_payloads(self, labeled_payloads):
        # We override this to inject timing wrappers around the tasks
        await self._ensure_session()
        
        tasks = {}
        for label, payload in labeled_payloads:
            # Wrap the coroutine with a timer
            tasks[label] = asyncio.create_task(self._timed_request(label, payload))
            
        results = {}
        try:
            # Wait for all
            for label, task in tasks.items():
                results[label] = await task
                if not self._is_success(results[label]):
                    print(f"   ❌ {label.upper()} Failed: {results[label]}")
        except Exception as e:
            await self._reset_session()
            raise e
            
        self.last_responses = results
        return results

    async def _timed_request(self, label, payload):
        t0 = time.perf_counter()
        # Call the original _post_payload from the parent class
        res = await self._post_payload(payload)
        t1 = time.perf_counter()
        
        duration_ms = (t1 - t0) * 1000
        # Log immediately clearly
        print(f"   ⏱️  {label.upper():<4} Leg Latency: {duration_ms:.2f} ms")
        return res

async def main():
    print(f"🎬 LIVE SCENARIO TEST: Entry -> Wait {WAIT_TIME_SECONDS}s -> Exit")
    print("=============================================================")
    
    # 1. Initialize
    connector = HyperliquidConnector()
    execution = InstrumentedExecutionManager(connector, SYMBOL)
    
    print("🛠️ Setup & Warmup...")
    execution.setup_account()
    await execution._ensure_session()
    
    # Warmup connections
    url = f"{execution.base_url}/info"
    payload = {"type": "meta"}
    tasks = [execution._session.post(url, json=payload) for _ in range(3)]
    responses = await asyncio.gather(*tasks)
    for r in responses: await r.read()
    print("✅ Connections Warm.\n")

    # 2. Get Market Data
    snapshot = connector.info.l2_snapshot(name=SYMBOL)
    spot_price = float(snapshot["levels"][0][0]["px"])
    perp_price = spot_price # Approx
    spot_asset_id = connector.get_spot_asset_id(SYMBOL)
    
    size = round(TEST_SIZE_USD / spot_price, 2)
    print(f"📊 Market Price: ${spot_price} | Trade Size: {size} {SYMBOL}")

    # 3. ENTRY SIGNAL
    print("\n🚀 [T=0] SPREAD DETECTED! FIRING ENTRY...")
    print("   ---------------------------------------")
    
    t_start = time.perf_counter()
    entry_success = await execution.execute_entry_parallel(
        size=size,
        spot_price=spot_price,
        perp_price=perp_price,
        spot_asset_id=spot_asset_id
    )
    t_end = time.perf_counter()
    
    if entry_success:
        print(f"   ✅ ENTRY SUCCESSFUL. (Total logic+net: {(t_end-t_start)*1000:.1f} ms)")
    else:
        print("   ❌ ENTRY FAILED. Aborting test.")
        return

    # 4. WAIT (Simulate Strategy)
    print(f"\n⏳ Holding position for {WAIT_TIME_SECONDS} seconds (Simulating spread duration)...")
    for i in range(WAIT_TIME_SECONDS, 0, -1):
        sys.stdout.write(f"\r   {i}...")
        sys.stdout.flush()
        await asyncio.sleep(1)
    print("\r   Done waiting.   ")

    # 5. EXIT SIGNAL
    print("\n📉 [T=10] SPREAD CLOSED! FIRING EXIT...")
    print("   --------------------------------------")
    
    # Update price slightly for exit logic (though IOC doesn't care much)
    exit_success = await execution.execute_exit_parallel(
        size=size,
        spot_price=spot_price,
        perp_price=perp_price,
        spot_asset_id=spot_asset_id,
        symbol=SYMBOL
    )
    
    if exit_success:
        print("   ✅ EXIT SUCCESSFUL. Position Closed.")
    else:
        print("   ⚠️ EXIT FAILED/PARTIAL. Check balances manually.")

    print("\n🏁 TEST COMPLETE.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

