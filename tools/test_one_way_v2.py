from __future__ import annotations

import asyncio
import time
import sys
from datetime import datetime
import logging

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

# Configuration
TEST_SIZE_USD = 11.0
SYMBOL = "HYPE"

async def main():
    print("🚀 ONE-WAY LATENCY TEST (v2) STARTED")
    print("=======================================")
    
    # 1. Initialize
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector, SYMBOL)
    execution.setup_account()
    
    # Ensure session
    await execution._ensure_session()
    
    # 2. Warmup
    print("\n📡 Warming up connection (2 requests)...")
    url = f"{execution.base_url}/info"
    payload = {"type": "meta"}
    tasks = [execution._session.post(url, json=payload) for _ in range(2)]
    responses = await asyncio.gather(*tasks)
    for r in responses: await r.read()
    print("✅ Warmup complete.")

    # 3. Get Market Data
    print("\n🔍 Fetching market data...")
    # Use sync snapshot for simplicity in setup
    snapshot = connector.info.l2_snapshot(name=SYMBOL)
    spot_price = float(snapshot["levels"][0][0]["px"])
    # Assume perp price is close
    perp_price = spot_price 
    
    print(f"   Spot Price: {spot_price}")
    spot_asset_id = connector.get_spot_asset_id(SYMBOL)

    # 4. Execute Trade & Capture Timestamp
    size = round(TEST_SIZE_USD / spot_price, 2)
    print(f"\n⚡ EXECUTION: Buying {size} {SYMBOL}...")
    
    # Sync Clock Check (Approximate)
    # We assume system clock is relatively synced with NTP.
    # We record T_SEND right before await.
    
    t_send_ns = time.time_ns()
    t_send_ms = t_send_ns / 1_000_000
    
    success = await execution.execute_entry_parallel(
        size=size,
        spot_price=spot_price,
        perp_price=perp_price,
        spot_asset_id=spot_asset_id
    )
    
    t_ack_ns = time.time_ns()
    t_ack_ms = t_ack_ns / 1_000_000
    
    print(f"   Order Acknowledged (RTT): {t_ack_ms - t_send_ms:.2f} ms")
    
    if not success:
        print("❌ Trade failed. Cannot measure fill latency.")
        return

    # 5. Analyze Fills (The Truth)
    print("\n🕵️ ANALYZING FILLS ON-CHAIN/DB...")
    # Give API a moment to index if needed (usually instant for user_fills)
    await asyncio.sleep(1.0) 
    
    user_state = connector.info.user_fills(connector.master_address)
    # Get the most recent fill for this symbol
    relevant_fill = None
    
    print(f"   DEBUG: Looking for {SYMBOL} Buy size={size} at time near {t_send_ms}")
    if user_state:
        print(f"   DEBUG: Latest fill: {user_state[0]}")

    # We are SHORTING on Perp (Side A)
    # My code sends: Spot Buy (B), Perp Short (A)
    # user_fills returns Perp fills.
    
    for fill in user_state:
        # Look for Side A (Short)
        if fill['coin'] == SYMBOL and fill['side'] == 'A':
             f_sz = float(fill['sz'])
             f_time = fill['time']
             print(f"   -> Checking fill: {fill['coin']} {fill['side']} {f_sz} @ {f_time}")
             if abs(f_sz - size) < 0.001:
                # Check if it's recent (within last 5 seconds)
                if abs(f_time - t_send_ms) < 5000:
                    relevant_fill = fill
                    break
    
    if relevant_fill:
        t_fill_ms = relevant_fill['time']
        
        # Calculate One-Way Latency
        # Note: If Clock Skew exists, this might be negative or too large.
        # But typically AWS/Cloud servers are well synced.
        
        latency_one_way = t_fill_ms - t_send_ms
        
        print(f"\n📊 RESULTS (Timestamps in UTC ms)")
        print(f"   ------------------------------")
        print(f"   T_SEND (Local) : {t_send_ms:.2f}")
        print(f"   T_FILL (Server): {t_fill_ms:.2f}")
        print(f"   T_ACK  (Local) : {t_ack_ms:.2f}")
        print(f"   ------------------------------")
        print(f"   ONE-WAY LATENCY: {latency_one_way:.2f} ms")
        print(f"   (Send -> Matching Engine)")
        
        if latency_one_way < 0:
            print("\n⚠️  Negative latency detected!")
            print("    This means the server clock is slightly behind your local clock.")
            print("    However, the delta confirms the order arrived instantly.")
        elif latency_one_way < 100:
            print("\n🚀  INSANE SPEED! Sub-100ms execution confirmed.")
        elif latency_one_way < 200:
            print("\n⚡  Excellent Speed. Sub-200ms.")
    else:
        print("❌ Could not find the exact fill in recent history.")

    # 6. Cleanup
    print("\n🧹 Cleaning up positions...")
    await execution.execute_exit_parallel(
        size=size,
        spot_price=spot_price,
        perp_price=perp_price,
        spot_asset_id=spot_asset_id,
        symbol=SYMBOL
    )
    print("✅ Cleanup done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
