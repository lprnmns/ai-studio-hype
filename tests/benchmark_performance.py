from __future__ import annotations

import asyncio
import contextlib
import time

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector


async def run_benchmark(duration_seconds: int = 300) -> None:
    connector = HyperliquidConnector()
    bot = ArbitrageBot(connector)

    ws_task = asyncio.create_task(bot.run())
    start = time.time()
    print(f"Benchmark started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    try:
        await asyncio.sleep(duration_seconds)
    finally:
        ws_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_task
    end = time.time()

    total_ticks = bot.total_ticks
    total_spreads = bot.total_spread_calculations
    elapsed = end - start if end > start else 1
    tps = total_ticks / elapsed

    print(f"Benchmark ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Duration: {elapsed:.2f}s")
    print(f"Total ticks processed: {total_ticks}")
    print(f"Total spread calculations: {total_spreads}")
    print(f"Ticks per second (TPS): {tps:.2f}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

