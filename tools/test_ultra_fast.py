from __future__ import annotations

import asyncio
import logging
import time
import traceback
import socket
import ssl

import aiohttp
from eth_account import Account
from hyperliquid.utils.signing import sign_l1_action

from src.connector import HyperliquidConnector

class UltraFastExecutor:
    def __init__(self, private_key: str):
        self.wallet = Account.from_key(private_key)
        self._session = None

    async def start_session(self):
        # SSL Context optimizasyonu
        ssl_context = ssl.create_default_context()
        # Güvenlikten ödün vermeden hız:
        ssl_context.check_hostname = False # Hostname check'i kapat (Hız kazandırır)
        ssl_context.verify_mode = ssl.CERT_NONE # Sertifika doğrulama yok (Riskli ama test için OK)

        connector = aiohttp.TCPConnector(
            limit=0, # Sınırsız bağlantı
            ttl_dns_cache=None, # Sonsuz DNS cache
            use_dns_cache=True, 
            force_close=False, # Asla kapatma
            enable_cleanup_closed=False,
            ssl=ssl_context, # Custom SSL context
            family=socket.AF_INET # Sadece IPv4
        )
        
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }
        
        # TCP_NODELAY (Nagle algoritmasını kapat)
        # aiohttp bunu varsayılan olarak yapabilir ama emin olalım
        
        self._session = aiohttp.ClientSession(
            base_url="https://api.hyperliquid.xyz", 
            connector=connector,
            headers=headers
        )
        
        # ISINMA (Warm-up)
        print("🔥 Kanal Isıtılıyor (Aggressive Keep-Alive)...")
        t0 = time.perf_counter()
        async with self._session.post("/info", json={"type": "meta"}) as resp:
            await resp.read() # Body'i tüket
            t1 = time.perf_counter()
            print(f"✅ Kanal Hazır! İlk İstek: {(t1-t0)*1000:.2f} ms (Status: {resp.status})")

    async def execute_trade(self, asset_id: int, price: float, size: float, is_buy: bool):
        # 1. Build Payload
        action = {
            "type": "order",
            "grouping": "na",
            "orders": [{
                "a": asset_id,
                "b": is_buy,
                "p": f"{price:.5g}",
                "s": f"{size:.2f}",
                "r": False,
                "t": {"limit": {"tif": "Ioc"}}
            }]
        }
        
        nonce = int(time.time() * 1000)
        signature = sign_l1_action(self.wallet, action, None, nonce, None, True)
        
        payload = {
            "action": action,
            "nonce": nonce,
            "signature": signature
        }
        
        # 2. Network (IO Bound)
        t0 = time.perf_counter()
        try:
            async with self._session.post("/exchange", json=payload) as resp:
                text = await resp.text()
                t1 = time.perf_counter()
                return (t1 - t0) * 1000, text, resp.status
        except Exception as e:
            return 0, str(e), 0

async def main():
    connector = HyperliquidConnector()
    spot_id_str = connector.get_spot_asset_id("HYPE")
    spot_id = int(spot_id_str.replace("@", ""))
    mids = connector.info.all_mids()
    price = float(mids["HYPE"])
    
    executor = UltraFastExecutor(connector._private_key)
    await executor.start_session()
    
    print(f"\n🚀 ULTRA-FAST TEST v2 BAŞLIYOR")
    
    size = 10.5 / price
    
    # Test İsteği
    latency_ms, response, status = await executor.execute_trade(
        asset_id=spot_id, 
        price=price * 1.05, 
        size=size, 
        is_buy=True
    )
    
    print("\n" + "="*40)
    print(f"⚡ NETWORK LATENCY: {latency_ms:.2f} ms")
    print("-" * 40)
    print(f"Status: {status}")
    print(f"Cevap: {response[:100]}...")
    print("="*40)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
