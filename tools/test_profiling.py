from __future__ import annotations

import asyncio
import logging
import time
import traceback
import sys
import aiohttp
import socket

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

TEST_SIZE_USD = 11.0
MOCK_ENTRY_BPS_THRESHOLD = -5000

# ExecutionManager'ı "Enstrümante" Ediyoruz (Ölçüm yeteneği ekliyoruz)
class ProfilingExecutionManager(ExecutionManager):
    def __init__(self, connector):
        super().__init__(connector)
        self.timings = {}

    async def run_heartbeat(self) -> None:
        """Aggressive 0.5s heartbeat keeping 2 connections warm."""
        print("💓 Heartbeat started (0.5s interval, 2 conns)...")
        while True:
            await asyncio.sleep(0.5)
            if self._session and not self._session.closed:
                try:
                    t0 = time.perf_counter()
                    url = "/info"
                    payload = {"type": "clearinghouseState", "user": self.master_address}
                    # Keep 2 connections alive
                    tasks = [self._session.post(url, json=payload) for _ in range(2)]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    for resp in responses:
                        if not isinstance(resp, Exception):
                            await resp.read()
                    t1 = time.perf_counter()
                    # print(f"💓 Heartbeat RTT: {(t1-t0)*1000:.1f} ms")
                except Exception as e:
                    print(f"⚠️ Heartbeat failed: {e}")

    async def execute_entry_parallel(self, size, spot_price, perp_price, spot_asset_id):
        self.timings = {} # Reset
        
        t0 = time.perf_counter()
        
        # 1. Hazırlık ve Hesaplama
        request_size = self._round_size_down(size)
        spot_coin = self._spot_coin_from_asset_id(spot_asset_id)
        spot_limit = self._round_to_sig_figs(spot_price * 1.05)
        perp_limit = self._round_to_sig_figs(perp_price * 0.95)
        
        t1 = time.perf_counter()
        self.timings["Logic/Rounding"] = (t1 - t0) * 1000

        # 2. Payload Oluşturma (SDK Monkey Patch Kısmı)
        # Spot Payload
        t2_start = time.perf_counter()
        spot_payload = self._sdk_build_payload(
            coin_name=spot_coin, is_buy=True, size=request_size, price=spot_limit, reduce_only=False
        )
        t2_end = time.perf_counter()
        self.timings["Build Payload (Spot)"] = (t2_end - t2_start) * 1000

        # Perp Payload
        t3_start = time.perf_counter()
        perp_payload = self._sdk_build_payload(
            coin_name=self.symbol, is_buy=False, size=request_size, price=perp_limit, reduce_only=False
        )
        t3_end = time.perf_counter()
        self.timings["Build Payload (Perp)"] = (t3_end - t3_start) * 1000

        # 3. Ağ İsteği (Network)
        t4_start = time.perf_counter()
        responses = await self._post_parallel_payloads(
            [("spot", spot_payload), ("perp", perp_payload)]
        )
        t4_end = time.perf_counter()
        self.timings["Network (POST RTT)"] = (t4_end - t4_start) * 1000
        
        # Analiz
        spot_resp = responses["spot"]
        perp_resp = responses["perp"]
        spot_ok = self._is_success(spot_resp)
        perp_ok = self._is_success(perp_resp)
        
        # Temel mantığı koru (Test için basitleştirilmiş return)
        return spot_ok and perp_ok

class LatencyTestBot(ArbitrageBot):
    def __init__(self, connector, execution):
        super().__init__(connector)
        self.execution = execution
        self.test_done = False
        self.connector = connector

    async def _stream_books(self) -> None:
        await super()._stream_books()

    def _calculate_and_log_spread(self) -> None:
        super()._calculate_and_log_spread()
        
        if self.test_done: return
        if self.spot_price <= 0 or self.perp_price <= 0: return

        print(f"⚡ SİNYAL ALGILANDI! Spot: {self.spot_price} | Perp: {self.perp_price}")
        asyncio.create_task(self._execute_test_trade())
        self.test_done = True

    async def _execute_test_trade(self):
        try:
            size = round(TEST_SIZE_USD / self.spot_price, 2)
            spot_asset_id = self.connector.get_spot_asset_id("HYPE")
            
            print(f"🚀 Emirler Gönderiliyor... (Size: {size})")
            
            # ProfilingManager kullanıyoruz
            success = await self.execution.execute_entry_parallel(
                size, self.spot_price, self.perp_price, spot_asset_id
            )
            
            timings = self.execution.timings
            total_measured = sum(timings.values())
            
            print("\n" + "="*50)
            print("🔬 DETAYLI PERFORMANS ANALİZİ")
            print("="*50)
            
            for step, ms in timings.items():
                bar = "█" * int(ms / 50)  # Görsel bar
                print(f"{step:<20} : {ms:8.2f} ms  {bar}")
            
            print("-" * 50)
            print(f"TOPLAM ÖLÇÜLEN SÜRE  : {total_measured:8.2f} ms")
            print("="*50)
            
            if success:
                print("🧹 Temizlik yapılıyor...")
                await asyncio.sleep(1)
                await self.execution.execute_exit_parallel(
                    size, self.spot_price, self.perp_price, spot_asset_id, symbol="HYPE"
                )
            
            print("Test tamamlandı, çıkılıyor...")
            sys.exit(0)

        except Exception as e:
            print(f"❌ Test hatası: {e}")
            traceback.print_exc()
            sys.exit(1)

async def main():
    print("🔥 Profiling Testi Başlatılıyor...")
    connector = HyperliquidConnector()
    # Normal ExecutionManager yerine Profiling versiyonunu kullan
    execution = ProfilingExecutionManager(connector)
    
    execution.setup_account()
    _ = connector.get_spot_asset_id("HYPE")
    
    await execution._ensure_session()
    
    # Start aggressive heartbeat
    asyncio.create_task(execution.run_heartbeat())

    # Isınma (Warmup - MULTI-CONNECTION)
    try:
        ip = socket.gethostbyname("api.hyperliquid.xyz")
        print(f"🔍 Resolved IP: {ip}")
    except:
        pass

    print("📡 Kanal Isıtılıyor (2 Bağlantı)...")
    try:
        url = f"{execution.base_url}/info"
        headers = {"Content-Type": "application/json"}
        payload = {"type": "meta"}
        
        # Send 2 requests in parallel
        tasks = []
        for _ in range(2):
             tasks.append(execution._session.post(url, json=payload, headers=headers))
        
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            await resp.read()
            
        print(f"✅ Kanal Hazır ({len(responses)} bağlantı ısıtıldı)")
        
        # Wait a bit to settle
        await asyncio.sleep(1)
    except Exception as e:
        print(f"⚠️ Warmup Failed: {e}")
    
    bot = LatencyTestBot(connector, execution)
    
    try:
        await bot.run()
    except SystemExit:
        pass
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

