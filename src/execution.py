from __future__ import annotations

import asyncio
import logging
import math
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
    """Async Hyperliquid execution engine that reuses SDK payloads but fires via aiohttp."""

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
        self.base_url = getattr(
            self.sdk_exchange,
            "base_url",
            "https://api.hyperliquid.xyz",
        ) or "https://api.hyperliquid.xyz"

        self._perp_meta = self.info.meta()
        self._spot_meta = self.info.spot_meta()
        self._spot_index_to_name = {
            entry.get("index"): entry.get("name")
            for entry in self._spot_meta.get("universe", [])
            if entry.get("index") is not None
        }

        # Create a dedicated exchange instance for payload generation
        self.payload_exchange = Exchange(
            wallet=self.wallet,
            base_url=self.base_url,
            meta=self._perp_meta,
            spot_meta=self._spot_meta,
        )
        # Apply the monkey patch immediately
        self._patch_exchange_post(self.payload_exchange)

        self.last_responses: Dict[str, Any] = {}
        
        # Async session management
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Public sync wrappers (for compatibility)
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
            # self._loop.stop() # Don't stop loop if main runs it, but here we own it
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
        # SDK handles significant figures automatically via internal logic if we pass floats
        # But we double check our rounding logic.
        spot_limit = self._round_to_sig_figs(spot_price * 1.05)
        perp_limit = self._round_to_sig_figs(perp_price * 0.95)

        # Generate payloads using the SDK (Guaranteed Correctness)
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

        # Strict hedge check
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
    # Networking / signing
    # ------------------------------------------------------------------
    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=5)
            connector = aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                force_close=False,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

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
                # Minimal logging for speed
                if not self._is_success(results[label]):
                    print(f"📬 {label.upper()} FAIL: {results[label]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            await self._reset_session()
            raise exc

        self.last_responses = results
        return results

    async def _post_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        url = f"{self.base_url.rstrip('/')}/exchange"
        headers = {"Content-Type": "application/json"}
        async with self._session.post(url, json=payload, headers=headers) as resp:
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
            # This calls the SDK's order method, which calls our patched 'post'
            # Our patched 'post' returns the payload dictionary directly.
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
            # The SDK calls post(path, json=payload)
            body = payload
            if body is None:
                body = kwargs.get("json")
            return body

        # Bind the new method to the instance
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
        return self._loop.run_until_complete(coro)

    async def run_heartbeat(self) -> None:
        """Periodically sends a lightweight request to keep the connection warm."""
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            if self._session and not self._session.closed:
                try:
                    url = f"{self.base_url}/info"
                    headers = {"Content-Type": "application/json"}
                    payload = {"type": "meta"} # Lightweight request
                    async with self._session.post(url, json=payload, headers=headers) as resp:
                         await resp.text() # Consume response
                         # logger.info(f"💓 Heartbeat sent. Status: {resp.status}") # Optional logging
                except Exception as e:
                    logger.warning(f"⚠️ Heartbeat failed: {e}")
