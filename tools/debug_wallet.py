from __future__ import annotations

import sys

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    try:
        main_wallet = execution.main_wallet_address
        print(f"Main Wallet Address : {main_wallet}")

        spot_asset_id = connector.get_spot_asset_id("HYPE")
        spot_asset_int = execution._spot_asset_to_int(spot_asset_id)

        action = execution._build_order(
            asset_id=spot_asset_int,
            is_buy=True,
            size=0.01,
            price=1.0,
            reduce_only=False,
        )
        payload = execution._sign_action(action)
        print(f"Signed Payload Nonce : {payload['nonce']}")

        worker_wallet = execution.wallet.address
        print(f"Signer Wallet Address: {worker_wallet}")

        if worker_wallet != main_wallet:
            raise RuntimeError("Wallet mismatch detected! Signing logic is inconsistent.")

        print("✅ Wallet verification passed. Signing logic is consistent.")
    finally:
        execution.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"❌ {exc}")
        sys.exit(1)

