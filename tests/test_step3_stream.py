from __future__ import annotations

import asyncio
import contextlib

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector


async def _wait_for_prices(bot: ArbitrageBot, timeout: float = 15.0) -> bool:
    loop = asyncio.get_running_loop()
    start = loop.time()
    while loop.time() - start < timeout:
        if bot.spot_best_ask > 0 and bot.perp_best_bid > 0:
            return True
        await asyncio.sleep(1)
    return False


def test_step3_stream():
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector)

    async def runner():
        task = asyncio.create_task(bot.run())
        try:
            assert await _wait_for_prices(bot), "Did not receive live L2 data for spot/perp"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(runner())

