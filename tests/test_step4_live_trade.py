from __future__ import annotations

import asyncio
import os
import time

import pytest
from dotenv import load_dotenv

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


load_dotenv()


def _require_env_vars():
    if not os.getenv("HL_PRIVATE_KEY") or not os.getenv("HL_MASTER_ADDRESS"):
        pytest.skip("Skipping live trade test: Missing HL_PRIVATE_KEY or HL_MASTER_ADDRESS")


def _calc_size(target_usd_size, price):
    size = target_usd_size / price
    return round(size, 2)


def test_step4_live_trade():
    _require_env_vars()
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    execution.setup_account()

    SPOT_ID_HARDCODED = "@107"

    print("📡 Fetching L2 Snapshot for HYPE (Lightweight)...")

    perp_snapshot = connector.info.l2_snapshot(name="HYPE")
    perp_price = float(perp_snapshot["levels"][0][0]["px"])

    spot_price = perp_price

    TEST_SIZE_USD = 11.0
    size = _calc_size(TEST_SIZE_USD, spot_price)
    print(f"📐 Target size: {size} HYPE (Price ~{spot_price})")

    entry_ok = execution.execute_entry_ioc(size, spot_price, perp_price, SPOT_ID_HARDCODED)
    assert entry_ok, "Entry IOC leg failed."

    print("✅ Entry Successful. Waiting 5 seconds...")
    time.sleep(5)

    exit_ok = execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, SPOT_ID_HARDCODED, symbol="HYPE")
    assert exit_ok, "Exit leg failed."

