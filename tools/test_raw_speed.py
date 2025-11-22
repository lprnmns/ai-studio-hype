from __future__ import annotations

import os
import time

import requests
from eth_account import Account
from hyperliquid.utils.signing import sign_l1_action


BASE_URL = "https://api.hyperliquid.xyz"


def fetch_asset_id() -> int:
    response = requests.post(
        f"{BASE_URL}/info",
        json={"type": "meta"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    for idx, asset in enumerate(data.get("universe", [])):
        if asset.get("name") == "HYPE":
            return idx
    raise RuntimeError("HYPE asset not found in meta response.")


def main() -> None:
    private_key = os.getenv("HL_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("HL_PRIVATE_KEY must be set.")

    asset_id = fetch_asset_id()
    wallet = Account.from_key(private_key)

    action = {
        "type": "order",
        "orders": [
            {
                "a": asset_id,
                "b": True,
                "p": "30.5000",
                "s": "0.1000",
                "r": False,
                "t": {"limit": {"tif": "Ioc"}},
            }
        ],
        "grouping": "na",
    }

    timestamp = int(time.time() * 1000)
    signature = sign_l1_action(wallet, action, None, timestamp, None, True)
    payload = {"action": action, "signature": signature, "nonce": timestamp}

    t1 = time.perf_counter()
    response = requests.post(f"{BASE_URL}/exchange", json=payload, timeout=10)
    t2 = time.perf_counter()

    print(f"Raw HTTP Request Time: {(t2 - t1) * 1000:.2f} ms")
    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()

