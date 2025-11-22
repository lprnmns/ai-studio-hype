from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
import time
from typing import Optional

import colorama

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

try:
    import uvloop  # type: ignore
except Exception:  # pragma: no cover
    uvloop = None
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


ENTRY_BPS = 30  # 0.30%
EXIT_BPS = 5   # 0.05%
TRADE_SIZE_USD = 12.0
COOLDOWN_SECONDS = 10


class DisplayManager:
    def __init__(self) -> None:
        self.log_lines = 0
        self.log_start_line = 0

    def render(self, bot: ArbitrageBot, state: str, balance_tokens: float, last_latency_ms: float = 0.0) -> None:
        spot = bot.spot_price
        perp = bot.perp_price
        spread = bot.latest_spread_bps
        updated_ms = (
            f"{int((time.monotonic() - bot.last_update_time) * 1000)} ms ago"
            if bot.last_update_time
            else "waiting..."
        )
        balance_usd = balance_tokens * spot if spot > 0 else 0.0
        
        latency_display = f"{last_latency_ms:.1f} ms" if last_latency_ms > 0 else "--"

        lines = [
            "================ HYPERLIQUID ARBITRAGE BOT ================",
            f"STATUS: {state:<14} BALANCE: ${balance_usd:>10.2f}",
            "-----------------------------------------------------------",
            "SPOT (@107)   | PERP (HYPE)   | SPREAD      | UPDATED",
            f"{spot:>11.4f}   | {perp:>11.4f}   | {spread:>5.0f} bps   | {updated_ms}",
            "-----------------------------------------------------------",
            f"LAST EXECUTION LATENCY: {latency_display}",
            "===========================================================",
            "[LOGS AREA - Trade executions will appear below]",
            "",
        ]

        self.log_start_line = len(lines) + 1
        sys.stdout.write("\033[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def log(self, message: str) -> None:
        if self.log_start_line == 0:
            print(message)
            return
        line_no = self.log_start_line + self.log_lines
        sys.stdout.write(f"\033[{line_no};1H{message}\033[K\n")
        sys.stdout.flush()
        self.log_lines += 1


class MainController:
    def __init__(self) -> None:
        colorama.init(autoreset=True)
        self.connector = HyperliquidConnector()
        self.bot = ArbitrageBot(self.connector)
        self.execution = ExecutionManager(self.connector)
        self.display = DisplayManager()

        self.spot_asset_id = self.connector.get_spot_asset_id("HYPE")
        self.current_balance = 0.0
        self.in_position = self._detect_existing_position()
        self.ws_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        self.last_latency_ms = 0.0

    def _detect_existing_position(self) -> bool:
        self.current_balance = self.execution._get_spot_balance(symbol="HYPE", asset_id=self.spot_asset_id)
        return self.current_balance > 0.1

    def _refresh_balance(self) -> None:
        self.current_balance = self.execution._get_spot_balance(symbol="HYPE", asset_id=self.spot_asset_id)

    def _calc_trade_size(self, spot_price: float) -> float:
        if spot_price <= 0:
            return 0.0
        return round(TRADE_SIZE_USD / spot_price, 2)

    async def run(self) -> None:
        self.execution.setup_account()
        await self.execution._ensure_session() # Ensure session is created
        
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _handle_stop(*_: object) -> None:
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_stop)
            except NotImplementedError:
                pass

        self.ws_task = asyncio.create_task(self.bot.run())
        self.heartbeat_task = asyncio.create_task(self.execution.run_heartbeat()) # Start heartbeat

        try:
            await self._trade_loop(stop_event)
        finally:
            if self.ws_task:
                self.ws_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.ws_task
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.heartbeat_task

    async def _trade_loop(self, stop_event: asyncio.Event) -> None:
        state_label = "IN POSITION" if self.in_position else "SEARCHING"
        while not stop_event.is_set():
            spread_bps = self.bot.latest_spread_bps
            spot_price = self.bot.spot_price
            perp_price = self.bot.perp_price

            self.display.render(self.bot, state_label, self.current_balance, self.last_latency_ms)

            if (
                not self.in_position
                and spread_bps >= ENTRY_BPS
                and spot_price > 0
                and perp_price > 0
            ):
                self.display.log(colorama.Fore.GREEN + "🚀 ENTRY SIGNAL: Executing hedge...")
                size = self._calc_trade_size(spot_price)
                if size > 0:
                    t0 = time.perf_counter()
                    entry_ok = await self.execution.execute_entry_parallel(
                        size=size,
                        spot_price=spot_price,
                        perp_price=perp_price,
                        spot_asset_id=self.spot_asset_id,
                    )
                    t1 = time.perf_counter()
                    self.last_latency_ms = (t1 - t0) * 1000
                    
                    if entry_ok:
                        self.in_position = True
                        state_label = "IN POSITION"
                        self._refresh_balance()
                        self.display.log(colorama.Fore.GREEN + f"✅ Entry filled. (Latency: {self.last_latency_ms:.1f}ms)")
                        await asyncio.sleep(COOLDOWN_SECONDS)
                        continue
                    else:
                        self.display.log(colorama.Fore.RED + f"❌ Entry failed. (Latency: {self.last_latency_ms:.1f}ms)")

            if (
                self.in_position
                and spread_bps <= EXIT_BPS
                and spot_price > 0
                and perp_price > 0
            ):
                self.display.log(colorama.Fore.YELLOW + "💰 EXIT SIGNAL: Closing hedge...")
                t0 = time.perf_counter()
                exit_ok = await self.execution.execute_exit_parallel(
                    size=self._calc_trade_size(spot_price),
                    spot_price=spot_price,
                    perp_price=perp_price,
                    spot_asset_id=self.spot_asset_id,
                    symbol="HYPE",
                )
                t1 = time.perf_counter()
                self.last_latency_ms = (t1 - t0) * 1000
                
                if exit_ok:
                    self.in_position = False
                    state_label = "SEARCHING"
                    self._refresh_balance()
                    self.display.log(colorama.Fore.YELLOW + f"✅ Exit completed. (Latency: {self.last_latency_ms:.1f}ms)")
                    await asyncio.sleep(COOLDOWN_SECONDS)
                    continue
                else:
                    self.display.log(colorama.Fore.RED + f"❌ Exit failed. (Latency: {self.last_latency_ms:.1f}ms)")

            await asyncio.sleep(0.1)


async def main() -> None:
    controller = MainController()
    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
