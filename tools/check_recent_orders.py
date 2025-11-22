from __future__ import annotations

import datetime
from src.connector import HyperliquidConnector

def main():
    connector = HyperliquidConnector()
    user = connector.master_address
    
    print(f"🔍 Son emirler sorgulanıyor: {user}")
    
    # Son 50 emri çekelim (fills + open orders history yok, fills daha kesin)
    # Not: Hyperliquid API'de 'user_fills' kesinleşmiş işlemleri gösterir.
    fills = connector.info.user_fills(user)
    
    if not fills:
        print("❌ Hiç işlem bulunamadı.")
        return

    print(f"\n{'ZAMAN (UTC)':<25} | {'TÜR':<5} | {'COIN':<5} | {'FİYAT':<8} | {'OID':<12} | {'LATENCY ANALİZİ'}")
    print("-" * 90)
    
    # Fills listesi genellikle en yeniden eskiye sıralıdır
    # Gruplamak için basit bir mantık: Zaman farkı < 50ms olanları "Aynı Emir Grubu" sayalım
    
    last_time = 0
    group_start_time = 0
    
    for fill in fills[:20]: # Son 20 işlem
        ts = fill['time']
        dt_object = datetime.datetime.fromtimestamp(ts / 1000.0)
        time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        coin = fill['coin']
        side = "BUY" if fill['side'] == 'B' else "SELL"
        px = float(fill['px'])
        oid = fill['oid']
        
        # Gap Analizi
        gap_msg = ""
        if last_time > 0:
            diff = abs(last_time - ts)
            if diff < 1000: # 1 saniyeden kısa süre önce işlem olmuş
                gap_msg = f"⚡ {diff} ms fark (Önceki işlemle)"
        
        print(f"{time_str:<25} | {side:<5} | {coin:<5} | {px:<8.2f} | {oid:<12} | {gap_msg}")
        
        last_time = ts

if __name__ == "__main__":
    main()

