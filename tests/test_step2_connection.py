from __future__ import annotations

from src.connector import HyperliquidConnector


def test_step2_connection():
    connector = HyperliquidConnector()

    print(f"✅ Agent Address: {connector.agent_address}")
    print(f"✅ Master Address: {connector.master_address}")

    hype_asset_id = connector.get_spot_asset_id("HYPE")
    print(f"✅ HYPE Asset ID: {hype_asset_id}")
    assert hype_asset_id is not None

    balances = connector.get_account_balance()
    print(f"✅ Total Equity: {balances['total_equity']}")
    print(f"✅ Spot USDC Balance: {balances['spot_usdc']}")

