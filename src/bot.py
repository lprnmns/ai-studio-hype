from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Optional

import aiohttp

from src.connector import HyperliquidConnector

try:
    import uvloop
except Exception:  # pragma: no cover - uvloop unavailable on Windows
    uvloop = None  # type: ignore
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ArbitrageBot:
    """Streams spot/perp order books and calculates the live spread."""

    def __init__(self, connector: HyperliquidConnector, symbol: str = "HYPE") -> None:
        self.connector = connector
        self.symbol = symbol.upper()
        self._spot_coin = self.connector.get_spot_asset_id(self.symbol)
        self._perp_coin = self.symbol
        self._ws_url = self._build_ws_url(self.connector.exchange.base_url)

        self.spot_best_ask: float = 0.0
        self.perp_best_bid: float = 0.0
        self.spot_price: float = 0.0
        self.perp_price: float = 0.0
        self.latest_spread_bps: float = 0.0
        self.last_update_time: float = 0.0
        self.total_ticks: int = 0
        self.total_spread_calculations: int = 0

    @staticmethod
    def _build_ws_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :] + "/ws"
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :] + "/ws"
        raise ValueError(f"Unsupported base url format: {base_url}")

    async def run(self) -> None:
        """Main loop that maintains a websocket connection and processes data."""
        while True:
            try:
                await self._stream_books()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ArbitrageBot] websocket error: {exc}")
                await asyncio.sleep(2)

    async def _stream_books(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self._ws_url, heartbeat=25) as ws:
                await self._subscribe(ws)
                ping_task = asyncio.create_task(self._send_ping(ws))
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_message(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                finally:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task

    async def _send_ping(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send Hyperliquid-specific ping messages to keep the socket alive."""
        while True:
            await asyncio.sleep(45)
            await ws.send_str(json.dumps({"method": "ping"}))

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        subscriptions = [
            {"type": "l2Book", "coin": self._perp_coin},
            {"type": "l2Book", "coin": self._spot_coin},
        ]
        for sub in subscriptions:
            await ws.send_str(json.dumps({"method": "subscribe", "subscription": sub}))

    def _handle_message(self, message: str) -> None:
        if message == "Websocket connection established.":
            return

        payload = json.loads(message)
        if payload.get("channel") != "l2Book":
            return

        self.total_ticks += 1
        data = payload.get("data", {})
        coin = data.get("coin")
        levels = data.get("levels", [])
        if not levels or len(levels) < 2:
            return

        if coin and coin.lower() == self._perp_coin.lower():
            best_bid = self._extract_price(levels[0])
            if best_bid > 0:
                self.perp_best_bid = best_bid
                self.perp_price = best_bid
                self._calculate_and_log_spread()
        elif coin and coin.lower() == self._spot_coin.lower():
            best_ask = self._extract_price(levels[1])
            if best_ask > 0:
                self.spot_best_ask = best_ask
                self.spot_price = best_ask
                self._calculate_and_log_spread()

    @staticmethod
    def _extract_price(levels: list[dict]) -> float:
        if not levels:
            return 0.0
        top = levels[0]
        try:
            return float(top["px"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _calculate_and_log_spread(self) -> None:
        if self.spot_best_ask <= 0 or self.perp_best_bid <= 0:
            return

        spread_bps = (self.perp_best_bid - self.spot_best_ask) / self.spot_best_ask * 10000
        self.latest_spread_bps = spread_bps
        self.last_update_time = time.monotonic()
        self.total_spread_calculations += 1

