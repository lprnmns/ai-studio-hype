from __future__ import annotations

import asyncio
import logging
import time
import traceback
import sys
import aiohttp # aiohttp modülünü ekledik

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

# Test için özel ayarlar
TEST_SIZE_USD = 11.0
MOCK_ENTRY_BPS_THRESHOLD = -5000  # Her zaman tetiklenir

class LatencyTestBot(ArbitrageBot):
    """Test için özelleştirilmiş Bot: İlk fırsatta emri basar ve süreyi ölçer."""
    
    def __init__(self, connector: HyperliquidConnector, execution: ExecutionManager):
        super().__init__(connector)
        self.execution = execution
        self.test_done = False
        self.connector = connector

    async def _stream_books(self) -> None:
        await super()._stream_books()

    def _calculate_and_log_spread(self) -> None:
        super()._calculate_and_log_spread()
        
        if self.test_done:
            return
        if self.spot_price <= 0 or self.perp_price <= 0:
            return

        print(f"⚡ SİNYAL ALGILANDI! Spot: {self.spot_price} | Perp: {self.perp_price}")
        asyncio.create_task(self._execute_test_trade())
        self.test_done = True

    async def _execute_test_trade(self):
        try:
            t0 = time.perf_counter()
            
            size = round(TEST_SIZE_USD / self.spot_price, 2)
            spot_asset_id = self.connector.get_spot_asset_id("HYPE")
            
            print(f"🚀 Emirler Gönderiliyor... (Size: {size})")
            
            success = await self.execution.execute_entry_parallel(
                size, 
                self.spot_price, 
                self.perp_price, 
                spot_asset_id
            )
                
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            
            print("\n" + "="*40)
            print("🏁 LATENCY SONUCU (Hot Path)")
            print("="*40)
            print(f"⏱️  Client-Side Execution Time: {elapsed_ms:.2f} ms")
            print(f"✅ İşlem Başarılı mı?: {success}")
            print("-" * 40)
            
            if success:
                print("🧹 Temizlik: Pozisyon kapatılıyor...")
                await asyncio.sleep(1)
                await self.execution.execute_exit_parallel(
                    size, 
                    self.spot_price, 
                    self.perp_price, 
                    spot_asset_id, 
                    symbol="HYPE"
                )
            
            print("Test tamamlandı, çıkılıyor...")
            sys.exit(0)

        except Exception as e:
            print(f"❌ Test hatası: {e}")
            traceback.print_exc()
            sys.exit(1)

async def main():
    print("🔥 Isınma Turu (Warm-up)...")
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)
    
    execution.setup_account()
    _ = connector.get_spot_asset_id("HYPE")
    
    # 1. Session Başlat
    await execution._ensure_session()
    
    # 2. GERÇEK ISINMA: Kanalı açmak için boş bir istek atalım
    print("📡 Bağlantı kanalı ısıtılıyor (Keep-Alive)...")
    try:
        # Borsanın 'exchange' endpoint'ine (işlem yapılan yer) geçerli ama işlem yapmayan bir istek
        # veya 'info' endpoint'ine bir istek atarak TCP/SSL el sıkışmasını tamamlayalım.
        # execution._session nesnesine eriştik (public değil ama test için kullanıyoruz)
        
        # Yöntem A: Info isteği (Hızlı ve güvenli)
        url = f"{execution.base_url}/info"
        headers = {"Content-Type": "application/json"}
        payload = {"type": "meta"}
        async with execution._session.post(url, json=payload, headers=headers) as resp:
            await resp.text()
            print(f"✅ Kanal Isındı! (HTTP {resp.status})")
            
    except Exception as e:
        print(f"⚠️ Isınma hatası (önemsiz): {e}")
    
    print("✅ Sistem Tam Hazır. Sinyal Bekleniyor...")
    
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
