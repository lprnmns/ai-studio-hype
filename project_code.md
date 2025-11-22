# Hyperliquid Arbitrage Bot Project

This document contains the full source code for the Hyperliquid Arbitrage Bot.
The system consists of:
1.  `src/connector.py`: Connection handling and basic API wrapper.
2.  `src/bot.py`: Real-time data streaming and spread calculation.
3.  `src/execution.py`: Optimized trade execution, signing, and connection management.
4.  `src/main.py`: The main entry point orchestrating the strategy.
5.  `requirements.txt`: Project dependencies.

## 1. `src/connector.py`

```python
import logging
import os
from typing import Any

from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HyperliquidConnector:
    def __init__(self) -> None:
        load_dotenv()

        self.private_key = os.getenv("HL_PRIVATE_KEY")
        self.master_address = os.getenv("HL_MASTER_ADDRESS")
        self.wallet_address = os.getenv("HL_WALLET_ADDRESS")

        if not self.private_key:
            raise EnvironmentError("HL_PRIVATE_KEY not found in .env")
        if not self.master_address:
            raise EnvironmentError("HL_MASTER_ADDRESS not found in .env")

        # Initialize Hyperliquid SDK components
        try:
            # "base_url" is optional, defaults to production API
            self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
            self.agent_account: LocalAccount = self._setup_account()
            
            # Initialize Exchange with the agent account
            self.exchange = Exchange(
                self.agent_account, 
                constants.MAINNET_API_URL,
                account_address=self.master_address
            )
            
            logger.info(f"✅ Connected to Hyperliquid. Master: {self.master_address}")

        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            raise

    def _setup_account(self) -> LocalAccount:
        """Initializes the LocalAccount from the private key."""
        import eth_account

        account: LocalAccount = eth_account.Account.from_key(self.private_key)
        logger.info(f"🔑 Agent Wallet loaded: {account.address}")
        return account

    def get_spot_asset_id(self, symbol: str) -> str:
        """
        Resolves a spot token symbol (e.g., 'HYPE') to its asset ID (e.g., '@107').
        Useful for Spot trading which requires specific asset IDs.
        """
        try:
            spot_meta = self.info.spot_meta()
            universe = spot_meta.get("universe", [])
            
            for token in universe:
                if token["name"] == symbol:
                    index = token["index"]
                    asset_id = f"@{index}"
                    logger.info(f"🔍 Resolved {symbol} to Asset ID: {asset_id}")
                    return asset_id
            
            raise ValueError(f"Token {symbol} not found in Spot Universe.")
        except Exception as e:
            logger.error(f"❌ Failed to resolve spot asset ID: {e}")
            raise
```

## 2. `src/bot.py`

```python
import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from hyperliquid.utils import constants
import websocket

from src.connector import HyperliquidConnector

logger = logging.getLogger(__name__)


class ArbitrageBot:
    def __init__(self, connector: HyperliquidConnector, symbol: str = "HYPE") -> None:
        self.connector = connector
        self.symbol = symbol
        self.spot_asset_id = connector.get_spot_asset_id(symbol)
        self.spot_id_int = int(self.spot_asset_id.replace("@", ""))

        # Market Data State
        self.spot_price: float = 0.0
        self.perp_price: float = 0.0
        self.latest_spread_bps: float = 0.0
        self.last_update_time: float = 0.0
        
        # Performance Metrics
        self.total_ticks: int = 0
        self.total_spread_calculations: int = 0

        # WebSocket Management
        self.ws: Optional[websocket.WebSocketApp] = None
        self.loop = asyncio.get_event_loop()
        self.should_run = True

    async def run(self) -> None:
        """Starts the WebSocket listener in the asyncio loop."""
        logger.info(f"🤖 Bot starting for {self.symbol}...")
        await self._stream_books()

    async def _stream_books(self) -> None:
        """Connects to Hyperliquid WebSocket and subscribes to L1 books."""
        url = "wss://api.hyperliquid.xyz/ws"

        def on_message(ws: Any, message: str) -> None:
            try:
                data = json.loads(message)
                self._handle_data(data)
            except Exception as e:
                logger.error(f"WS Message Error: {e}")

        def on_error(ws: Any, error: str) -> None:
            logger.error(f"WS Error: {error}")

        def on_close(ws: Any, close_status_code: int, close_msg: str) -> None:
            logger.warning("WS Connection Closed")

        def on_open(ws: Any) -> None:
            logger.info("✅ WebSocket Connected")
            # Subscribe to Spot and Perp L1 books (most lightweight)
            subscriptions = [
                {"type": "l2Book", "coin": self.symbol},  # Perp
                {"type": "l2Book", "coin": self.spot_asset_id},  # Spot
            ]
            
            for sub in subscriptions:
                msg = json.dumps({"method": "subscribe", "subscription": sub})
                ws.send(msg)
                # logger.info(f"📡 Subscribed to: {sub}")

        self.ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Run blocking WebSocket in a separate thread executor
        await self.loop.run_in_executor(None, self.ws.run_forever)

    def _handle_data(self, data: Dict[str, Any]) -> None:
        """Parses incoming WebSocket messages."""
        channel = data.get("channel")
        
        if channel == "l2Book":
            self.total_ticks += 1
            msg_data = data.get("data", {})
            coin = msg_data.get("coin")
            levels = msg_data.get("levels", [[], []])
            
            # Extract Best Bid/Ask
            # L1 Book format: [ [bid_px, bid_sz], [ask_px, ask_sz] ]
            if not levels or len(levels) < 2:
                return
                
            # If there are bids/asks
            best_bid = float(levels[0]["px"]) if levels[0] else 0.0
            best_ask = float(levels[1]["px"]) if levels[1] else 0.0
            mid_price = (best_bid + best_ask) / 2

            # Update State
            if coin == self.symbol:  # Perp
                self.perp_price = mid_price
            elif coin == self.spot_asset_id or coin == f"@{self.spot_id_int}":  # Spot
                self.spot_price = mid_price
            
            # Calculate Spread if we have both prices
            self._calculate_and_log_spread()

    def _calculate_and_log_spread(self) -> None:
        if self.spot_price > 0 and self.perp_price > 0:
            self.total_spread_calculations += 1
            
            # Formula: (Perp Bid - Spot Ask) / Spot Ask
            # But using mid-prices for simpler monitoring:
            # Spread BPS = ((Perp - Spot) / Spot) * 10000
            
            diff = self.perp_price - self.spot_price
            spread_bps = (diff / self.spot_price) * 10000
            
            self.latest_spread_bps = spread_bps
            self.last_update_time = time.time()
```

## 3. `src/execution.py`

```python
from __future__ import annotations

import asyncio
import logging
import math
import ssl
import socket
import threading
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - performance optimization
    import ujson as json  # type: ignore
except ImportError:  # pragma: no cover - fallback
    import json

import aiohttp
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange

from src.connector import HyperliquidConnector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionManager:
    """Async Hyperliquid execution engine with optimized networking (IPv4 Force + No SSL Verify)."""

    def __init__(self, connector: HyperliquidConnector, symbol: str = "HYPE") -> None:
        self.connector = connector
        self.symbol = symbol.upper()
        self.info = connector.info
        self.sdk_exchange: Exchange = connector.exchange
        self.wallet: LocalAccount = connector.agent_account
        if not isinstance(self.wallet, LocalAccount):
            raise TypeError("HyperliquidConnector.agent_account must be a LocalAccount.")
        self.main_wallet_address = self.wallet.address
        self.master_address = connector.master_address
        
        # Original base URL
        self.base_url = getattr(
            self.sdk_exchange,
            "base_url",
            "https://api.hyperliquid.xyz",
        ) or "https://api.hyperliquid.xyz"
        
        # Extract Domain (just for logging/debug if needed)
        self.domain = self.base_url.split("//")[-1].split("/")[0]

        self._perp_meta = self.info.meta()
        self._spot_meta = self.info.spot_meta()
        self._spot_index_to_name = {
            entry.get("index"): entry.get("name")
            for entry in self._spot_meta.get("universe", [])
            if entry.get("index") is not None
        }

        # Create a dedicated exchange instance for payload generation
        self.payload_exchange = Exchange(
            self.wallet,
            self.base_url,
            self._perp_meta,
            self._spot_meta,
        )
        self._patch_exchange_post(self.payload_exchange)

        self.last_responses: Dict[str, Any] = {}
        
        # Async session management
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Public sync wrappers
    # ------------------------------------------------------------------
    def setup_account(self) -> Dict[str, Any]:
        print("🛠️ Setting HYPE leverage to 3x")
        response = self.sdk_exchange.update_leverage(leverage=3, name=self.symbol)
        print(f"✅ Leverage response: {response}")
        return response

    def close(self) -> None:
        """Close aiohttp session + loop (best effort)."""
        try:
            self._run_async(self._close_session())
        finally:
            pass

    def execute_entry_ioc(self, size: float, spot_price: float, perp_price: float, spot_asset_id: str) -> bool:
        return self._run_async(self.execute_entry_parallel(size, spot_price, perp_price, spot_asset_id))

    def execute_exit_alo_or_ioc(
        self,
        size: float,
        spot_price: float,
        perp_price: float,
        spot_asset_id: str,
        symbol: str = "HYPE",
    ) -> bool:
        return self._run_async(self.execute_exit_parallel(size, spot_price, perp_price, spot_asset_id, symbol))

    # ------------------------------------------------------------------
    # Core async logic
    # ------------------------------------------------------------------
    async def execute_entry_parallel(
        self,
        size: float,
        spot_price: float,
        perp_price: float,
        spot_asset_id: str,
    ) -> bool:
        request_size = self._round_size_down(size)
        if request_size <= 0:
            raise ValueError("Order size must be positive.")

        spot_coin = self._spot_coin_from_asset_id(spot_asset_id)
        spot_limit = self._round_to_sig_figs(spot_price * 1.05)
        perp_limit = self._round_to_sig_figs(perp_price * 0.95)

        spot_payload = self._sdk_build_payload(
            coin_name=spot_coin,
            is_buy=True,
            size=request_size,
            price=spot_limit,
            reduce_only=False,
        )
        perp_payload = self._sdk_build_payload(
            coin_name=self.symbol,
            is_buy=False,
            size=request_size,
            price=perp_limit,
            reduce_only=False,
        )

        print(f"🚀 Firing ENTRY: Spot Buy {request_size} @ {spot_limit} | Perp Short {request_size} @ {perp_limit}")
        responses = await self._post_parallel_payloads(
            [("spot", spot_payload), ("perp", perp_payload)]
        )
        spot_resp = responses["spot"]
        perp_resp = responses["perp"]

        spot_fill = self._extract_filled_size(spot_resp)
        perp_fill = self._extract_filled_size(perp_resp)
        spot_ok = self._is_success(spot_resp)
        perp_ok = self._is_success(perp_resp)

        if spot_ok and perp_ok and self._fills_match(spot_fill, perp_fill, request_size):
            return True

        if not spot_ok and not perp_ok:
            print("❌ Entry failed on both legs.")
            return False

        print("⚠️ Mismatch detected! Closing all positions...")
        if spot_fill > 0:
            await self._panic_close_spot(spot_coin, spot_fill, spot_price)
        if perp_fill > 0:
            await self._panic_close_perp(perp_fill, perp_price)
        return False

    async def execute_exit_parallel(
        self,
        size: float,
        spot_price: float,
        perp_price: float,
        spot_asset_id: str,
        symbol: str,
    ) -> bool:
        spot_balance = self._floor_size(self._get_spot_balance(symbol=symbol, asset_id=spot_asset_id))
        if spot_balance <= 0:
            print("⚠️ No spot balance detected; skipping exit.")
            return False

        perp_size = self._round_size_down(size)
        
        spot_coin = self._spot_coin_from_asset_id(spot_asset_id)
        spot_limit = self._round_to_sig_figs(spot_price * 0.99)
        perp_limit = self._round_to_sig_figs(perp_price * 1.01)

        spot_payload = self._sdk_build_payload(
            coin_name=spot_coin,
            is_buy=False,
            size=spot_balance,
            price=spot_limit,
            reduce_only=False,
        )
        perp_payload = self._sdk_build_payload(
            coin_name=symbol,
            is_buy=True,
            size=perp_size,
            price=perp_limit,
            reduce_only=True,
        )

        print(f"🚀 Firing EXIT: Spot Sell {spot_balance} @ {spot_limit} | Perp Buy {perp_size} @ {perp_limit}")
        responses = await self._post_parallel_payloads(
            [("spot", spot_payload), ("perp", perp_payload)]
        )
        spot_resp = responses["spot"]
        perp_resp = responses["perp"]

        if self._is_success(spot_resp) and self._is_success(perp_resp):
            return True

        if not self._is_success(spot_resp) and not self._is_success(perp_resp):
            print("❌ Exit failed on both legs.")
            return False

        print("⚠️ Exit encountered partial failure; please retry.")
        return False

    # ------------------------------------------------------------------
    # Networking / signing (OPTIMIZED - NO CUSTOM RESOLVER)
    # ------------------------------------------------------------------
    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            # 1. Optimized SSL Context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 2. Optimized TCP Connector (Matches test_ultra_fast.py exactly)
            connector = aiohttp.TCPConnector(
                limit=0, 
                ttl_dns_cache=None, # Infinite DNS Cache
                use_dns_cache=True,
                force_close=False,
                enable_cleanup_closed=False,
                ssl=ssl_context,
                family=socket.AF_INET # Force IPv4
            )
            
            # 3. Create Session
            timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_read=2)
            headers = {
                "Content-Type": "application/json", 
                "Connection": "keep-alive"
            }
            
            self._session = aiohttp.ClientSession(
                base_url=self.base_url, 
                timeout=timeout, 
                connector=connector,
                headers=headers,
                json_serialize=json.dumps
            )

    async def _close_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _reset_session(self) -> None:
        await self._close_session()
        print("⚠️ Connection reset. Re-establishing tunnel...")
        await self._ensure_session()

    async def _post_parallel_payloads(
        self,
        labeled_payloads: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        await self._ensure_session()
        tasks = {
            label: asyncio.create_task(self._post_payload(payload))
            for label, payload in labeled_payloads
        }
        results: Dict[str, Any] = {}
        try:
            for label, task in tasks.items():
                results[label] = await task
                if not self._is_success(results[label]):
                    print(f"📬 {label.upper()} FAIL: {results[label]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._reset_session()
            raise exc

        self.last_responses = results
        return results

    async def _post_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        url = "/exchange"
        async with self._session.post(url, json=payload) as resp:
            text = await resp.text()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"status": "err", "error": text}

    async def _panic_close_spot(self, spot_coin: str, size: float, spot_price: float) -> None:
        size = self._round_size_down(size)
        if size <= 0:
            return
        payload = self._sdk_build_payload(
            coin_name=spot_coin,
            is_buy=False,
            size=size,
            price=self._round_to_sig_figs(spot_price * 0.8), # Aggressive
            reduce_only=False,
        )
        try:
            resp = (await self._post_parallel_payloads([("spot_panic", payload)]))["spot_panic"]
            print(f"🆘 Panic Spot: {resp}")
        except Exception as e:
            print(f"🆘 Panic Spot Failed: {e}")

    async def _panic_close_perp(self, size: float, perp_price: float) -> None:
        size = self._round_size_down(size)
        if size <= 0:
            return
        payload = self._sdk_build_payload(
            coin_name=self.symbol,
            is_buy=True,
            size=size,
            price=self._round_to_sig_figs(perp_price * 1.2), # Aggressive
            reduce_only=True,
        )
        try:
            resp = (await self._post_parallel_payloads([("perp_panic", payload)]))["perp_panic"]
            print(f"🆘 Panic Perp: {resp}")
        except Exception as e:
            print(f"🆘 Panic Perp Failed: {e}")

    # ------------------------------------------------------------------
    # Utilities & Monkey Patching
    # ------------------------------------------------------------------
    def _sdk_build_payload(
        self,
        *,
        coin_name: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool,
        tif: str = "Ioc",
    ) -> Dict[str, Any]:
        try:
            return self.payload_exchange.order(
                name=coin_name,
                is_buy=is_buy,
                sz=size,
                limit_px=price,
                order_type={"limit": {"tif": tif}},
                reduce_only=reduce_only,
            )
        except Exception as exc:
            raise RuntimeError(f"SDK payload build failed for {coin_name}: {exc}") from exc

    def _spot_coin_from_asset_id(self, spot_asset_id: str) -> str:
        idx = self._spot_asset_to_int(spot_asset_id)
        coin = self._spot_index_to_name.get(idx)
        if not coin:
            raise ValueError(f"Spot asset '{spot_asset_id}' not found in metadata.")
        return coin

    @staticmethod
    def _spot_asset_to_int(spot_asset_id: str) -> int:
        if spot_asset_id.startswith("@"):
            return int(spot_asset_id[1:])
        if spot_asset_id.isdigit():
            return int(spot_asset_id)
        raise ValueError(f"Unexpected spot asset id format: {spot_asset_id}")

    @staticmethod
    def _patch_exchange_post(exchange: Exchange) -> None:
        def _return_payload(self, path, payload=None, **kwargs):
            body = payload
            if body is None:
                body = kwargs.get("json")
            return body
        exchange.post = MethodType(_return_payload, exchange)

    def _extract_filled_size(self, response: Any) -> float:
        statuses = self._extract_statuses(response)
        if not statuses:
            return 0.0
        for status in statuses:
            if self._status_has_error(status):
                return 0.0
            filled = status.get("filled")
            if filled and filled.get("totalSz"):
                try:
                    return float(filled["totalSz"])
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _extract_statuses(self, response: Any) -> List[Dict[str, Any]]:
        if not isinstance(response, dict):
            return []
        payload = response.get("response") or response
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        statuses = data.get("statuses")
        return statuses if isinstance(statuses, list) else []

    @staticmethod
    def _status_has_error(status: Dict[str, Any]) -> bool:
        if not isinstance(status, dict):
            return True
        state = str(status.get("status", "")).lower()
        if state in {"error", "rejected"}:
            return True
        return "error" in status

    @staticmethod
    def _round_size_down(value: float) -> float:
        return math.floor(value * 100) / 100 if value > 0 else 0.0

    @staticmethod
    def _floor_size(value: float) -> float:
        return math.floor(value * 100) / 100 if value > 0 else 0.0

    @staticmethod
    def _round_to_sig_figs(value: float, sig_figs: int = 5) -> float:
        if value == 0:
            return 0.0
        return float(f"{value:.{sig_figs}g}")

    @staticmethod
    def _is_success(response: Any) -> bool:
        return isinstance(response, dict) and response.get("status") == "ok"

    def _get_spot_balance(self, symbol: str, asset_id: Optional[str] = None) -> float:
        state = self.info.spot_user_state(self.master_address)
        balances = state.get("balances", [])
        symbol_upper = symbol.upper()
        for balance in balances:
            coin_key = balance.get("coin")
            if not coin_key:
                continue
            try:
                amount = float(balance.get("total", 0.0))
            except (TypeError, ValueError):
                amount = 0.0
            if coin_key.upper() == symbol_upper:
                return amount
            if asset_id and coin_key == asset_id:
                return amount
        return 0.0

    @staticmethod
    def _fills_match(spot_filled: float, perp_filled: float, target: float, tolerance: float = 0.02) -> bool:
        min_fill = target * (1 - tolerance)
        return spot_filled >= min_fill and perp_filled >= min_fill

    def _run_async(self, coro):
        """Helper to run async code from sync context."""
        if self._loop.is_running():
             raise RuntimeError("Cannot call sync wrapper from inside an event loop. Use await async_method() instead.")
        return self._loop.run_until_complete(coro)

    async def run_heartbeat(self) -> None:
        """Periodically sends lightweight requests to keep the connection pool warm."""
        logger.info("💓 Aggressive heartbeat started (0.5s interval, 2 connections)...")
        while True:
            await asyncio.sleep(0.5)
            if self._session and not self._session.closed:
                try:
                    # Use user state as it touches the matching engine backend
                    url = "/info"
                    payload = {"type": "clearinghouseState", "user": self.master_address}
                    
                    # Fire 2 concurrent requests to keep multiple sockets open in the pool
                    tasks = [self._session.post(url, json=payload) for _ in range(2)]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for resp in responses:
                        if not isinstance(resp, Exception):
                            await resp.read()
                            
                except Exception as e:
                    logger.warning(f"⚠️ Heartbeat failed: {e}")
```

## 4. `src/main.py`

```python
import asyncio
import logging
import sys
import time

# Optional: use uvloop for faster asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import colorama
from colorama import Fore, Style

from src.bot import ArbitrageBot
from src.connector import HyperliquidConnector
from src.execution import ExecutionManager

# Initialize colorama for cross-platform ANSI
colorama.init()

# --- Configuration ---
ENTRY_BPS = 30       # Enter if spread > 0.30% (30 bps)
EXIT_BPS = 5         # Exit if spread < 0.05% (5 bps)
TRADE_SIZE_USD = 11.0
COOLDOWN_SECONDS = 10.0

# --- UI / Display ---
class DisplayManager:
    def __init__(self):
        self.logs = []
        self.max_logs = 5
        
    def log(self, msg: str, color: str = Fore.WHITE):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL} {color}{msg}{Style.RESET_ALL}"
        self.logs.append(formatted)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def render(self, bot: ArbitrageBot, state: str, balance_tokens: float, last_latency_ms: float = 0.0) -> None:
        # Clear screen or move to home
        sys.stdout.write("\033[H")
        
        # Header
        print(f"{Fore.BLUE}================= HYPERLIQUID ARBITRAGE BOT ================={Style.RESET_ALL}")
        
        # Status Row
        status_color = Fore.GREEN if state == "SEARCHING" else Fore.YELLOW
        print(f"STATUS: {status_color}{state:<10}{Style.RESET_ALL} | BALANCE: {Fore.GREEN}{balance_tokens:.2f} HYPE{Style.RESET_ALL}")
        print("-" * 61)
        
        # Market Data Row
        spot = bot.spot_price
        perp = bot.perp_price
        bps = bot.latest_spread_bps
        updated_ms = (time.time() - bot.last_update_time) * 1000 if bot.last_update_time else 0
        
        bps_color = Fore.GREEN if bps >= ENTRY_BPS else (Fore.RED if bps <= EXIT_BPS else Fore.WHITE)
        latency_display = f"{last_latency_ms:.1f} ms" if last_latency_ms > 0 else "--"
        
        print(f"SPOT (@107)    | PERP (HYPE)    | SPREAD      | UPDATED")
        print(f"${spot:<13.4f} | ${perp:<13.4f} | {bps_color}{bps:>4.0f} bps{Style.RESET_ALL}    | {updated_ms:>4.0f}ms ago")
        print("-" * 61)
        print(f"LAST EXECUTION LATENCY: {latency_display}")
        print("-" * 61)
        
        # Recent Logs
        print(f"{Fore.MAGENTA}[LOGS AREA]{Style.RESET_ALL}")
        for log_line in self.logs:
            print(log_line)
            
        # Clear remaining lines to avoid artifacts
        print("\033[J", end="")
        sys.stdout.flush()

# --- Main Controller ---
class MainController:
    def __init__(self):
        self.connector = HyperliquidConnector()
        self.bot = ArbitrageBot(self.connector)
        self.execution = ExecutionManager(self.connector)
        self.display = DisplayManager()
        
        self.in_position = False
        self.last_latency_ms = 0.0
        
    async def start(self):
        # 1. Setup
        self.display.log("Initializing...", Fore.YELLOW)
        self.execution.setup_account()
        
        # Check initial state
        spot_balance = self.execution._get_spot_balance("HYPE")
        if spot_balance > 0.1:
            self.in_position = True
            self.display.log(f"Detected existing position: {spot_balance:.2f} HYPE", Fore.YELLOW)
        else:
            self.in_position = False
            self.display.log("No existing position detected.", Fore.GREEN)

        # Warmup connection
        await self.execution._ensure_session()
        
        # Start Aggressive Heartbeat in Background
        asyncio.create_task(self.execution.run_heartbeat())

        # Start Data Stream
        asyncio.create_task(self.bot.run())
        
        # Wait for data
        while self.bot.spot_price == 0:
            self.display.render(self.bot, "STARTING", spot_balance)
            await asyncio.sleep(0.1)
            
        self.display.log("Data stream active! Starting trade loop.", Fore.GREEN)
        
        # Run Trade Loop
        await self._trade_loop()

    async def _trade_loop(self):
        spot_asset_id = self.connector.get_spot_asset_id("HYPE")
        
        while True:
            try:
                spot_price = self.bot.spot_price
                perp_price = self.bot.perp_price
                spread_bps = self.bot.latest_spread_bps
                
                # Determine State
                state = "HOLDING" if self.in_position else "SEARCHING"
                balance = self.execution._get_spot_balance("HYPE")
                
                self.display.render(self.bot, state, balance, self.last_latency_ms)
                
                # ENTRY LOGIC
                if not self.in_position and spread_bps >= ENTRY_BPS and spot_price > 0:
                    self.display.log(f"🚀 Opportunity: {spread_bps:.0f} bps. Entering...", Fore.GREEN)
                    
                    # Calculate size fresh
                    size = TRADE_SIZE_USD / spot_price
                    
                    t0 = time.perf_counter()
                    success = await self.execution.execute_entry_parallel(
                        size, spot_price, perp_price, spot_asset_id
                    )
                    t1 = time.perf_counter()
                    self.last_latency_ms = (t1 - t0) * 1000
                    
                    if success:
                        self.in_position = True
                        self.display.log("✅ Entry Successful!", Fore.GREEN)
                        await asyncio.sleep(COOLDOWN_SECONDS)
                    else:
                        self.display.log("❌ Entry Failed.", Fore.RED)
                        await asyncio.sleep(1) # Short retry wait

                # EXIT LOGIC
                elif self.in_position and spread_bps <= EXIT_BPS and spot_price > 0:
                    self.display.log(f"📉 Spread closed: {spread_bps:.0f} bps. Exiting...", Fore.CYAN)
                    
                    size = TRADE_SIZE_USD / spot_price # Dummy size, logic uses balance
                    
                    t0 = time.perf_counter()
                    success = await self.execution.execute_exit_parallel(
                        size, spot_price, perp_price, spot_asset_id, symbol="HYPE"
                    )
                    t1 = time.perf_counter()
                    self.last_latency_ms = (t1 - t0) * 1000
                    
                    if success:
                        self.in_position = False
                        self.display.log("✅ Exit Successful!", Fore.GREEN)
                        await asyncio.sleep(COOLDOWN_SECONDS)
                    else:
                        self.display.log("⚠️ Exit Failed/Partial.", Fore.RED)
                        await asyncio.sleep(1)

                await asyncio.sleep(0.01) # Fast tick
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.display.log(f"Error: {e}", Fore.RED)
                await asyncio.sleep(1)

async def main():
    controller = MainController()
    await controller.start()

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
```

## 5. `requirements.txt`

```text
hyperliquid-python-sdk
python-dotenv
eth-account
aiohttp
uvloop; sys_platform != 'win32'
colorama
websocket-client
```
