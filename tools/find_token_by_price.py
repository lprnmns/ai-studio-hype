from __future__ import annotations

from typing import Any, Dict, Optional

from hyperliquid.info import Info


def find_token_index(meta: Dict[str, Any], coin_name: str) -> Optional[int]:
    for token in meta.get("tokens", []):
        if token.get("name") == coin_name:
            return token.get("index")
    for entry in meta.get("universe", []):
        if entry.get("name") == coin_name:
            return entry.get("index")
    return None


def main() -> None:
    info = Info(skip_ws=True)
    prices: Dict[str, str] = info.all_mids()
    spot_meta = info.spot_meta()

    matches = []
    for coin, price_str in prices.items():
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            continue
        if 30.0 <= price <= 45.0:
            index = find_token_index(spot_meta, coin)
            matches.append((coin, index, price))

    if not matches:
        print("No spot assets found with price between $30 and $45.")
        return

    for coin, index, price in matches:
        print(f"🎯 MATCH FOUND: Name: {coin} | Index: @{index if index is not None else '?'} | Price: ${price:.2f}")


if __name__ == "__main__":
    main()

