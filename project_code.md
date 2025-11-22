# Project Codebase - Source & Tools

This document contains the full source code for the Hyperliquid Arbitrage Bot, including the main application logic (`src/`) and various utility/diagnostic tools (`tools/`).

## Root Scripts

### `deploy_and_test.py`

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SERVER_IP = "24.144.105.173"
SERVER_USER = "root"
SSH_KEY = "ssh.key"
ARCHIVE_NAME = "hyperliquid_bot.zip"


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True)


def ensure_key_exists(key_path: Path) -> None:
    if not key_path.exists():
        print(f"❌ SSH key not found at {key_path}")
        sys.exit(1)


def create_archive(project_root: Path) -> None:
    print("📦 ZIPPING PROJECT...")
    archive_path = project_root / ARCHIVE_NAME
    if archive_path.exists():
        archive_path.unlink()

    include_paths = ["src", "tests", "tools", "requirements.txt", ".env"]
    exclude_dirs = {".git", "logs", "venv", "__pycache__"}
    exclude_suffixes = {".pyc"}

    def should_skip(path: Path) -> bool:
        parts = set(path.parts)
        if parts & exclude_dirs:
            return True
        if path.suffix in exclude_suffixes:
            return True
        return False

    with ZipFile(archive_path, "w", ZIP_DEFLATED) as zipf:
        for item in include_paths:
            full_path = project_root / item
            if not full_path.exists():
                continue
            if full_path.is_file():
                if not should_skip(full_path):
                    zipf.write(full_path, full_path.relative_to(project_root))
                continue
            for file_path in full_path.rglob("*"):
                if file_path.is_dir():
                    continue
                if should_skip(file_path):
                    continue
                zipf.write(file_path, file_path.relative_to(project_root))
    print(f"✅ Archive created at {archive_path}")


def upload_archive(project_root: Path, key_path: Path) -> None:
    print("🚀 UPLOADING ARCHIVE...")
    cmd = [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        str(project_root / ARCHIVE_NAME),
        f"{SERVER_USER}@{SERVER_IP}:~/",
    ]
    run_command(cmd)
    print("✅ Upload complete.")


def run_remote_commands(key_path: Path) -> None:
    print("☁️ RUNNING REMOTE TEST...")
    remote_cmd = " && ".join(
        [
            "sudo apt-get update",
            "sudo apt-get install -y unzip python3-pip python3-venv",
            "unzip -o hyperliquid_bot.zip",
            "python3 -m venv hyperliquid_env",
            "hyperliquid_env/bin/pip install --upgrade pip",
            "hyperliquid_env/bin/pip install -r requirements.txt",
            "hyperliquid_env/bin/pip install uvloop pytest",
            "hyperliquid_env/bin/python -m pytest tests/test_latency_breakdown.py -s",
        ]
    )
    ssh_cmd = [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        f"{SERVER_USER}@{SERVER_IP}",
        remote_cmd,
    ]
    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print("❌ Remote command failed.")
        sys.exit(process.returncode)
    print("✅ Remote latency test completed.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    key_path = project_root / SSH_KEY
    ensure_key_exists(key_path)
    create_archive(project_root)
    upload_archive(project_root, key_path)
    run_remote_commands(key_path)


if __name__ == "__main__":
    main()
```

### `git_manager.py`

```python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(command: list[str]) -> None:
    """Run a git command and surface errors immediately."""
    result = subprocess.run(command, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def build_authenticated_url(repo_url: str, token: str) -> str:
    if not repo_url.startswith("https://"):
        raise ValueError("GITHUB_REPO_URL must use https://")
    return repo_url.replace("https://", f"https://{token}@", 1)


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Automate git add/commit/push")
    parser.add_argument("message", nargs="+", help="Commit message")
    args = parser.parse_args()
    return " ".join(args.message)


def main() -> int:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    commit_message = parse_args()
    github_token = os.getenv("GITHUB_TOKEN")
    repo_url = os.getenv("GITHUB_REPO_URL")

    if not github_token:
        print("Missing GITHUB_TOKEN in .env", file=sys.stderr)
        return 1
    if not repo_url:
        print("Missing GITHUB_REPO_URL in .env", file=sys.stderr)
        return 1

    auth_url = build_authenticated_url(repo_url, github_token)

    try:
        run_cmd(["git", "add", "."])
        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=REPO_ROOT,
            text=True,
        )
        if commit_result.returncode != 0:
            print("git commit failed; aborting push.", file=sys.stderr)
            return commit_result.returncode
        run_cmd(["git", "push", auth_url, "HEAD"])
    except subprocess.CalledProcessError as exc:
        print(f"Command failed: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `requirements.txt`

```text
hyperliquid-python-sdk
python-dotenv
eth-account
aiohttp
uvloop
```

## Source Code (`src/`)

### `src/bot.py`

```python
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
```

### `src/connector.py`

```python
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_utils import to_checksum_address
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info


class HyperliquidConnector:
    """Wrapper around the Hyperliquid SDK to separate agent vs master wallet logic."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        load_dotenv()
        self._private_key = self._require_env("HL_PRIVATE_KEY")
        self.master_address = self._normalize_address(self._require_env("HL_MASTER_ADDRESS"))

        self.agent_account: LocalAccount = Account.from_key(self._private_key)
        self.agent_address: str = self.agent_account.address

        self.exchange = Exchange(wallet=self.agent_account, base_url=base_url)
        self.info: Info = self.exchange.info

    @staticmethod
    def _require_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise RuntimeError(f"Environment variable '{key}' is missing.")
        return value

    @staticmethod
    def _normalize_address(address: str) -> str:
        address = address.strip()
        try:
            return to_checksum_address(address)
        except ValueError as exc:
            raise ValueError(f"Invalid Ethereum address provided for '{address}'.") from exc

    @staticmethod
    def _token_identifier(token_meta: Dict[str, Any]) -> str:
        index = token_meta.get("index")
        if index is not None:
            return f"@{index}"
        token_id = token_meta.get("tokenId")
        if token_id:
            return token_id
        raise ValueError("Token metadata missing both 'index' and 'tokenId'.")

    @staticmethod
    def _format_asset_index(index: Optional[int]) -> str:
        if index is None:
            raise ValueError("Spot meta entry missing 'index'.")
        return f"@{index}"

    def get_spot_asset_id(self, symbol: str) -> str:
        """Resolve the tradable spot pair index for a given symbol."""
        symbol_upper = symbol.upper()
        meta = self.info.spot_meta()
        tokens: List[Dict[str, Any]] = meta.get("tokens", [])
        universe: List[Dict[str, Any]] = meta.get("universe", [])

        base_token_index: Optional[int] = None
        for token_meta in tokens:
            if token_meta.get("name", "").upper() == symbol_upper:
                base_token_index = token_meta.get("index")
                break

        if base_token_index is not None:
            for universe_entry in universe:
                pair_tokens = universe_entry.get("tokens", [])
                if not pair_tokens:
                    continue
                base_idx = pair_tokens[0]
                if base_idx == base_token_index:
                    return self._format_asset_index(universe_entry.get("index"))

        # Fallback: support callers passing explicit pair names (e.g., HYPE/USDC or @107)
        for universe_entry in universe:
            pair_name = universe_entry.get("name", "")
            if pair_name.upper() == symbol_upper:
                return self._format_asset_index(universe_entry.get("index"))

        # Absolute last resort: return token identifier (should rarely happen)
        for token_meta in tokens:
            if token_meta.get("name", "").upper() == symbol_upper:
                return self._token_identifier(token_meta)

        raise ValueError(f"Spot symbol '{symbol}' not found in Hyperliquid spot metadata.")

    def get_account_balance(self) -> Dict[str, float]:
        """Return total equity and spot USDC balance for the master wallet."""
        clearing_state = self.info.user_state(self.master_address)
        total_equity = float(clearing_state["marginSummary"]["accountValue"])

        spot_state = self.info.spot_user_state(self.master_address)
        usdc_balance = 0.0
        for balance in spot_state.get("balances", []):
            if balance.get("coin", "").upper() == "USDC":
                usdc_balance = float(balance.get("total", 0.0))
                break

        return {"total_equity": total_equity, "spot_usdc": usdc_balance}
```

### `src/execution.py`

> **Note**: This file uses a "Monkey Patch" approach on the `hyperliquid.exchange.Exchange` class. We create a dedicated `payload_exchange` instance and replace its internal `post` method with a function that simply returns the payload instead of sending it. This ensures that we get the exact, correctly-signed, byte-perfect payload that the official SDK would generate, avoiding "User does not exist" errors caused by manual JSON formatting mismatches. We then use `aiohttp` to send these payloads in parallel for maximum speed.

```python
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
            self._session = aiohttp.ClientSession(timeout=timeout)

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
```

### `src/main.py`

```python
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

    def render(self, bot: ArbitrageBot, state: str, balance_tokens: float) -> None:
        spot = bot.spot_price
        perp = bot.perp_price
        spread = bot.latest_spread_bps
        updated_ms = (
            f"{int((time.monotonic() - bot.last_update_time) * 1000)} ms ago"
            if bot.last_update_time
            else "waiting..."
        )
        balance_usd = balance_tokens * spot if spot > 0 else 0.0

        lines = [
            "================ HYPERLIQUID ARBITRAGE BOT ================",
            f"STATUS: {state:<14} BALANCE: ${balance_usd:>10.2f}",
            "-----------------------------------------------------------",
            "SPOT (@107)   | PERP (HYPE)   | SPREAD      | UPDATED",
            f"{spot:>11.4f}   | {perp:>11.4f}   | {spread:>5.0f} bps   | {updated_ms}",
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

        try:
            await self._trade_loop(stop_event)
        finally:
            if self.ws_task:
                self.ws_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.ws_task

    async def _trade_loop(self, stop_event: asyncio.Event) -> None:
        state_label = "IN POSITION" if self.in_position else "SEARCHING"
        while not stop_event.is_set():
            spread_bps = self.bot.latest_spread_bps
            spot_price = self.bot.spot_price
            perp_price = self.bot.perp_price

            self.display.render(self.bot, state_label, self.current_balance)

            if (
                not self.in_position
                and spread_bps >= ENTRY_BPS
                and spot_price > 0
                and perp_price > 0
            ):
                self.display.log(colorama.Fore.GREEN + "🚀 ENTRY SIGNAL: Executing hedge...")
                size = self._calc_trade_size(spot_price)
                if size > 0:
                    entry_ok = self.execution.execute_entry_ioc(
                        size=size,
                        spot_price=spot_price,
                        perp_price=perp_price,
                        spot_asset_id=self.spot_asset_id,
                    )
                    if entry_ok:
                        self.in_position = True
                        state_label = "IN POSITION"
                        self._refresh_balance()
                        self.display.log(colorama.Fore.GREEN + "✅ Entry filled.")
                        await asyncio.sleep(COOLDOWN_SECONDS)
                        continue
                    else:
                        self.display.log(colorama.Fore.RED + "❌ Entry failed.")

            if (
                self.in_position
                and spread_bps <= EXIT_BPS
                and spot_price > 0
                and perp_price > 0
            ):
                self.display.log(colorama.Fore.YELLOW + "💰 EXIT SIGNAL: Closing hedge...")
                exit_ok = self.execution.execute_exit_alo_or_ioc(
                    size=self._calc_trade_size(spot_price),
                    spot_price=spot_price,
                    perp_price=perp_price,
                    spot_asset_id=self.spot_asset_id,
                    symbol="HYPE",
                )
                if exit_ok:
                    self.in_position = False
                    state_label = "SEARCHING"
                    self._refresh_balance()
                    self.display.log(colorama.Fore.YELLOW + "✅ Exit completed.")
                    await asyncio.sleep(COOLDOWN_SECONDS)
                    continue
                else:
                    self.display.log(colorama.Fore.RED + "❌ Exit failed.")

            await asyncio.sleep(0.1)


async def main() -> None:
    controller = MainController()
    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())
```

## Tools (`tools/`)

### `tools/benchmark_signing.py`

```python
from __future__ import annotations

import os
import statistics
import time

from eth_account import Account
from hyperliquid.utils.signing import sign_l1_action


def benchmark_signing(iterations: int = 100) -> None:
    private_key = os.getenv("HL_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("HL_PRIVATE_KEY must be set in the environment for benchmarking.")

    wallet = Account.from_key(private_key)
    dummy_action = {
        "type": "order",
        "orders": [
            {
                "coin": "HYPE",
                "is_buy": True,
                "limit_px": 35.0,
                "sz": 0.3,
                "order_type": {"limit": {"tif": "Ioc"}},
            }
        ],
    }

    timings_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        sign_l1_action(
            wallet,
            dummy_action,
            None,  # vault_address
            int(time.time() * 1000),
            None,  # expires_after
            True,
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        timings_ms.append(elapsed_ms)

    print("=== SIGNING BENCHMARK ===")
    print(f"Iterations: {iterations}")
    print(f"Min: {min(timings_ms):.4f} ms")
    print(f"Max: {max(timings_ms):.4f} ms")
    print(f"Avg: {statistics.mean(timings_ms):.4f} ms")


if __name__ == "__main__":
    benchmark_signing()
```

### `tools/check_gap.py`

```python
from src.connector import HyperliquidConnector


def main():
    SPOT_OID = 244435299155
    PERP_OID = 244435302607

    print(f"🔍 Sorgulanıyor... Spot OID: {SPOT_OID}, Perp OID: {PERP_OID}")

    connector = HyperliquidConnector()
    spot_res = connector.info.query_order_by_oid(connector.master_address, SPOT_OID)
    perp_res = connector.info.query_order_by_oid(connector.master_address, PERP_OID)

    t_spot = spot_res["order"]["order"]["timestamp"]
    t_perp = perp_res["order"]["order"]["timestamp"]

    print("\n" + "=" * 30)
    print(f"🕒 Spot Server Time: {t_spot}")
    print(f"🕒 Perp Server Time: {t_perp}")
    print("-" * 30)
    print(f"🚀 EXECUTION GAP: {abs(t_spot - t_perp)} ms")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
```

### `tools/check_price.py`

```python
from __future__ import annotations

import json

from hyperliquid.info import Info
from hyperliquid.utils import constants


def main() -> None:
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    print("\n--- SPOT @150 RAW DATA ---")
    spot_l2 = info.post("/info", {"type": "l2Book", "coin": "@150"})
    print(json.dumps(spot_l2, indent=2)[:500])

    print("\n--- PERP HYPE RAW DATA ---")
    perp_l2 = info.l2_snapshot(name="HYPE")
    print(json.dumps(perp_l2, indent=2)[:500])

    print("\n--- SPOT META (HYPE) ---")
    spot_meta = info.spot_meta()
    for token in spot_meta["tokens"]:
        if token["index"] == 150:
            print(json.dumps(token, indent=2))
            break


if __name__ == "__main__":
    main()
```

### `tools/debug_network_win.py`

```python
from __future__ import annotations

import subprocess
from typing import Tuple


def run_command(cmd: list[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 1, f"Command not found: {' '.join(cmd)}"


def parse_ping(output: str) -> str:
    for line in output.splitlines():
        if "Minimum" in line and "Maximum" in line and "Average" in line:
            return line.strip()
    return "Ping summary not found."


def ping_test() -> None:
    print("=== ICMP PING TEST (Windows) ===")
    code, output = run_command(["ping", "-n", "4", "api.hyperliquid.xyz"])
    if code != 0:
        print("Ping failed:")
        print(output)
        return
    print(output)
    summary = parse_ping(output)
    print(f"Summary: {summary}\n")


def curl_timing() -> None:
    print("=== HTTPS LATENCY (curl) ===")
    format_str = (
        r"time_namelookup: %{time_namelookup}\n"
        r"time_connect: %{time_connect}\n"
        r"time_appconnect: %{time_appconnect}\n"
        r"time_starttransfer: %{time_starttransfer}\n"
        r"time_total: %{time_total}\n"
    )
    code, output = run_command(
        [
            "curl",
            "-o",
            "NUL",
            "-s",
            "-w",
            format_str,
            "https://api.hyperliquid.xyz/info",
        ]
    )
    if code != 0:
        print("curl failed:")
        print(output)
        return
    print(output)


def cpu_load() -> None:
    print("=== CPU LOAD (Windows) ===")
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage (1s avg): {cpu_percent:.2f}%\n")
    except ImportError:
        code, output = run_command(
            ["wmic", "cpu", "get", "loadpercentage", "/value"]
        )
        if code != 0:
            print("Unable to determine CPU load.")
            print(output + "\n")
        else:
            print(output.strip() + "\n")


def main() -> None:
    ping_test()
    curl_timing()
    cpu_load()


if __name__ == "__main__":
    main()
```

### `tools/debug_network.py`

```python
from __future__ import annotations

import os
import subprocess
from typing import Tuple


def run_command(cmd: list[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 1, f"Command not found: {' '.join(cmd)}"


def parse_ping(output: str) -> str:
    for line in output.splitlines():
        if "min/avg/max" in line or "min/avg/max/mdev" in line:
            return line.strip()
    return "Ping summary not found."


def ping_test() -> None:
    print("=== ICMP PING TEST ===")
    code, output = run_command(["ping", "-c", "4", "api.hyperliquid.xyz"])
    if code != 0:
        print("Ping failed:")
        print(output)
        return
    print(output)
    summary = parse_ping(output)
    print(f"Summary: {summary}\n")


def curl_timing() -> None:
    print("=== HTTPS LATENCY (curl) ===")
    format_str = (
        r"time_namelookup: %{time_namelookup}\n"
        r"time_connect: %{time_connect}\n"
        r"time_appconnect: %{time_appconnect}\n"
        r"time_starttransfer: %{time_starttransfer}\n"
        r"time_total: %{time_total}\n"
    )
    code, output = run_command(
        [
            "curl",
            "-o",
            "/dev/null",
            "-s",
            "-w",
            format_str,
            "https://api.hyperliquid.xyz/info",
        ]
    )
    if code != 0:
        print("curl failed:")
        print(output)
        return
    print(output)


def cpu_load() -> None:
    print("=== CPU LOAD ===")
    try:
        load = os.getloadavg()
        print(f"Load averages (1m, 5m, 15m): {load}\n")
    except (AttributeError, OSError):
        print("getloadavg not available on this system.\n")


def main() -> None:
    ping_test()
    curl_timing()
    cpu_load()


if __name__ == "__main__":
    main()
```

### `tools/debug_signature_mismatch.py`

```python
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

import msgpack
from eth_account import Account

from hyperliquid.utils.signing import (
    order_request_to_order_wire,
    order_wires_to_order_action,
    recover_agent_or_user_from_l1_action,
    sign_l1_action,
)

from src.connector import HyperliquidConnector


def _hex_dump(action: Dict[str, Any]) -> str:
    packed = msgpack.packb(action, use_bin_type=True)
    return packed.hex()


def _manual_action(asset_id: int, is_buy: bool, price: float, size: float) -> Dict[str, Any]:
    return {
        "type": "order",
        "grouping": "na",
        "orders": [
            {
                "a": asset_id,
                "b": is_buy,
                "p": f"{price:.8f}",
                "s": f"{size:.4f}",
                "r": False,
                "t": {"limit": {"tif": "Ioc"}},
            }
        ],
    }


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _diff_structures(a: Any, b: Any, path: str = "") -> List[str]:
    diffs: List[str] = []
    if type(a) != type(b):
        diffs.append(f"{path or 'root'} type mismatch: {type(a).__name__} vs {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for key in sorted(keys):
            new_path = f"{path}.{key}" if path else key
            if key not in a:
                diffs.append(f"{new_path} missing in SDK action")
            elif key not in b:
                diffs.append(f"{new_path} missing in Manual action")
            else:
                diffs.extend(_diff_structures(a[key], b[key], new_path))
        return diffs

    if isinstance(a, list):
        max_len = max(len(a), len(b))
        for idx in range(max_len):
            new_path = f"{path}[{idx}]"
            if idx >= len(a):
                diffs.append(f"{new_path} missing in SDK action")
            elif idx >= len(b):
                diffs.append(f"{new_path} missing in Manual action")
            else:
                diffs.extend(_diff_structures(a[idx], b[idx], new_path))
        return diffs

    if a != b:
        diffs.append(f"{path or 'value'} mismatch: {a!r} vs {b!r}")
    return diffs


def main() -> None:
    connector = HyperliquidConnector()
    exchange = connector.exchange

    coin = "HYPE"
    test_price = 30.5
    test_size = 0.1
    order_type = {"limit": {"tif": "Ioc"}}

    order_request = {
        "coin": coin,
        "is_buy": True,
        "sz": test_size,
        "limit_px": test_price,
        "order_type": order_type,
        "reduce_only": False,
    }

    sdk_asset_id = exchange.info.name_to_asset(coin)
    sdk_order_wire = order_request_to_order_wire(order_request, sdk_asset_id)
    sdk_action = order_wires_to_order_action([sdk_order_wire])

    manual_action = _manual_action(
        asset_id=sdk_asset_id,
        is_buy=True,
        price=test_price,
        size=test_size,
    )

    _print_section("SDK ACTION (order_request_to_order_wire)")
    print(json.dumps(sdk_action, indent=2))
    print("msgpack:", _hex_dump(sdk_action))

    _print_section("MANUAL ACTION (ExecutionManager._build_order)")
    print(json.dumps(manual_action, indent=2))
    print("msgpack:", _hex_dump(manual_action))

    diffs = _diff_structures(sdk_action, manual_action)
    _print_section("STRUCTURAL DIFFERENCES")
    if not diffs:
        print("✅ No differences detected.")
    else:
        for diff in diffs:
            print("❌", diff)

    is_mainnet = "api.hyperliquid.xyz" in (exchange.base_url or "")

    _print_section("SIGNATURE RECOVERY TEST")
    nonce = int(time.time() * 1000)
    signature = sign_l1_action(connector.agent_account, manual_action, None, nonce, None, is_mainnet)
    recovered = recover_agent_or_user_from_l1_action(
        manual_action,
        signature,
        None,
        nonce,
        None,
        is_mainnet,
    )
    print("Expected Wallet :", connector.agent_account.address)
    print("Recovered Wallet:", recovered)

    if recovered.lower() == connector.agent_account.address.lower():
        print("✅ Signature recovers to the correct wallet.")
    else:
        print("❌ Signature recovers to a DIFFERENT wallet. Payload mismatch confirmed.")


if __name__ == "__main__":
    main()
```

### `tools/debug_wallet.py`

```python
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

        print("✅ Wallet verification passed. Signing logic is inconsistent.")
    finally:
        execution.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"❌ {exc}")
        sys.exit(1)
```

### `tools/find_token_by_price.py`

```python
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from hyperliquid.info import Info


def find_token_index(meta: Dict[str, Any], coin_name: str) -> Optional[int]:
    for token in meta.get("tokens", []):
        if token.get("name") == coin_name:
            return token.get("index")
    for entry in meta.get("universe", []):
        if entry.get("name") == coin_name:
            return entry.get("index")
    return None


def main() -> None:
    info = Info(skip_ws=True)
    prices: Dict[str, str] = info.all_mids()
    spot_meta = info.spot_meta()

    matches = []
    for coin, price_str in prices.items():
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            continue
        if 30.0 <= price <= 45.0:
            index = find_token_index(spot_meta, coin)
            matches.append((coin, index, price))

    if not matches:
        print("No spot assets found with price between $30 and $45.")
        return

    for coin, index, price in matches:
        print(f"🎯 MATCH FOUND: Name: {coin} | Index: @{index if index is not None else '?'} | Price: ${price:.2f}")


if __name__ == "__main__":
    main()
```

### `tools/inspect_universe.py`

```python
from __future__ import annotations

import sys
from typing import Any, Dict, List

from hyperliquid.info import Info


def print_tokens(tokens: List[Dict[str, Any]]) -> None:
    print("=== Spot Tokens ===")
    for token in tokens:
        index = token.get("index")
        name = token.get("name")
        token_id = token.get("tokenId")
        print(f"Index: @{index} | Name: {name} | TokenId: {token_id}")


def search_keyword(tokens: List[Dict[str, Any]], universe: List[Dict[str, Any]], keyword: str) -> None:
    keyword_upper = keyword.upper()
    print(f"\n=== Matches for '{keyword_upper}' ===")
    token_matches = [
        token for token in tokens if keyword_upper in str(token.get("name", "")).upper()
    ]
    universe_matches = [
        entry for entry in universe if keyword_upper in str(entry.get("name", "")).upper()
    ]

    if not token_matches and not universe_matches:
        print("No matches found.")
        return

    if token_matches:
        print("\nTokens:")
        for token in token_matches:
            print(
                f"Index: @{token.get('index')} | Name: {token.get('name')} | TokenId: {token.get('tokenId')}"
            )

    if universe_matches:
        print("\nUniverse Pairs:")
        for entry in universe_matches:
            print(f"Index: @{entry.get('index')} | Pair Name: {entry.get('name')} | Tokens: {entry.get('tokens')}")


def inspect_index(tokens: List[Dict[str, Any]], target_index: int) -> None:
    print(f"\n=== Inspecting Index @{target_index} ===")
    for token in tokens:
        if token.get("index") == target_index:
            print(
                f"Index: @{token.get('index')} | Name: {token.get('name')} | TokenId: {token.get('tokenId')}"
            )
            return
    print(f"No token found with index @{target_index}")


def main() -> None:
    info = Info(skip_ws=True)
    spot_meta = info.spot_meta()

    tokens = spot_meta.get("tokens", [])
    universe = spot_meta.get("universe", [])

    print_tokens(tokens)
    search_keyword(tokens, universe, "HYPE")
    inspect_index(tokens, 150)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
```

### `tools/test_one_way_v2.py`

```python
from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict, List, Optional

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def next_full_minute() -> dt.datetime:
    now = dt.datetime.utcnow()
    return now.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)


def busy_wait(target_ts: float) -> None:
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            break
        if remaining > 0.01:
            time.sleep(remaining - 0.005)
        else:
            pass


def extract_oid(response: Dict[str, Any]) -> Optional[int]:
    statuses: List[Dict[str, Any]] = response.get("statuses", [])
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled") or status.get("resting")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    target_dt = next_full_minute()
    target_ts = target_dt.timestamp()
    print(f"⏳ Waiting for target time: {target_dt.strftime('%H:%M:%S.000')} UTC")
    busy_wait(target_ts)

    mids = connector.info.all_mids()
    spot_asset_id = connector.get_spot_asset_id("HYPE")
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    fire_time = time.time()
    entry_response = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
    if not entry_response:
        print("Entry failed.")
        return

    if isinstance(entry_response, dict):
        oid = extract_oid(entry_response)
    else:
        oid = None

    if not oid:
        print("Could not extract order id from response.")
        print(entry_response)
        return

    user = connector.master_address
    order_details = connector.info.query_order_by_oid(user, oid)
    exchange_ts_ms = order_details.get("time")

    if exchange_ts_ms is None:
        print("Missing exchange timestamp.")
    else:
        diff_ms = exchange_ts_ms - int(target_ts * 1000)
        print(f"Local fire time: {fire_time:.3f}")
        print(f"Exchange timestamp: {exchange_ts_ms}")
        print(f"One-way latency: {diff_ms} ms")

    execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")


if __name__ == "__main__":
    main()
```

### `tools/test_one_way.py`

```python
from __future__ import annotations

import datetime as dt
import time
from typing import Any, Dict

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def next_full_minute() -> dt.datetime:
    now = dt.datetime.utcnow()
    target = (now.replace(second=0, microsecond=0) + dt.timedelta(minutes=1))
    return target


def busy_wait(target_ts: float) -> None:
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            break
        if remaining > 0.01:
            time.sleep(remaining - 0.005)
        else:
            pass  # busy wait last milliseconds


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    spot_asset_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_price = float(mids[spot_asset_id])
    perp_price = float(mids["HYPE"])
    size = round(12.0 / spot_price, 2)

    target = next_full_minute()
    target_ts = target.timestamp()
    print(f"⏳ Waiting for target time: {target.strftime('%H:%M:%S.000')} UTC")
    busy_wait(target_ts)

    fire_time = time.time()
    entry_ok = execution.execute_entry_ioc(size, spot_price, perp_price, spot_asset_id)
    if not entry_ok:
        print("Entry failed.")
        return

    report = execution.last_report if hasattr(execution, "last_report") else {}
    order_ids = []
    for status in report.get("statuses", []):
        if status.get("status") == "ok" and "oid" in status:
            order_ids.append(status["oid"])

    if not order_ids:
        print("Could not retrieve order ids from response.")
        return

    user = connector.master_address
    oid = order_ids[0]
    print(f"Order OID: {oid}")
    order_details = connector.info.query_order_by_oid(user, oid)
    exchange_ts_ms = order_details.get("time")
    if exchange_ts_ms is None:
        print(f"Order details: {order_details}")
        print("Missing exchange timestamp.")
        return

    diff_ms = exchange_ts_ms - int(target_ts * 1000)
    print(f"Local fire time: {fire_time:.3f}")
    print(f"Exchange timestamp: {exchange_ts_ms}")
    print(f"One-way latency (target -> exchange): {diff_ms} ms")

    execution.execute_exit_alo_or_ioc(size, spot_price, perp_price, spot_asset_id, symbol="HYPE")


if __name__ == "__main__":
    main()
```

### `tools/test_parallel_latency.py`

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def extract_oid(response: Any) -> Optional[int]:
    if not isinstance(response, dict):
        return None
    payload = response.get("response") or response
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    statuses = data.get("statuses", [])
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


def extract_server_time(details: Dict[str, Any]) -> Optional[int]:
    if not details:
        return None
    if "time" in details:
        return details["time"]
    order_root = details.get("order")
    if isinstance(order_root, dict):
        if "time" in order_root:
            return order_root["time"]
        inner = order_root.get("order")
        if isinstance(inner, dict) and "timestamp" in inner:
            return inner["timestamp"]
    return None


def main() -> None:
    connector = HyperliquidConnector()
    execution = ExecutionManager(connector)

    spot_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_px = float(mids[spot_id])
    perp_px = float(mids["HYPE"])
    size = round(11.0 / spot_px, 2)

    print(f"\n🔥 Parallel Latency Test: {size} HYPE (≈${size * spot_px:.2f})")
    t_client_start_ns = time.time_ns()
    success = execution.execute_entry_parallel(size, spot_px, perp_px, spot_id)
    t_client_end_ns = time.time_ns()

    if not success:
        print("⚠️ Entry reported mismatch; continuing with timestamp analysis...")

    spot_resp = execution.last_responses.get("spot", {})
    perp_resp = execution.last_responses.get("perp", {})
    spot_oid = extract_oid(spot_resp)
    perp_oid = extract_oid(perp_resp)

    if not spot_oid or not perp_oid:
        print("❌ Could not extract order IDs from responses.")
        print("Spot resp:", spot_resp)
        print("Perp resp:", perp_resp)
        return

    user = connector.master_address
    spot_details = connector.info.query_order_by_oid(user, spot_oid)
    perp_details = connector.info.query_order_by_oid(user, perp_oid)
    spot_server_time = extract_server_time(spot_details)
    perp_server_time = extract_server_time(perp_details)

    if spot_server_time is None or perp_server_time is None:
        print("❌ Could not extract server timestamps.")
        print("Spot details:", spot_details)
        print("Perp details:", perp_details)
        return

    t_client_start_ms = t_client_start_ns / 1_000_000
    spot_one_way = spot_server_time - t_client_start_ms
    perp_one_way = perp_server_time - t_client_start_ms
    gap_ms = abs(spot_server_time - perp_server_time)

    print("\n=== PARALLEL LATENCY REPORT ===")
    print(f"Client send ms  : {t_client_start_ms:.3f}")
    print(f"Spot server ms  : {spot_server_time}")
    print(f"Perp server ms  : {perp_server_time}")
    print("--------------------------------")
    print(f"📡 Spot one-way  : {spot_one_way:.2f} ms")
    print(f"📡 Perp one-way  : {perp_one_way:.2f} ms")
    print(f"🚀 Execution gap : {gap_ms} ms")
    print(f"Client RTT       : {(t_client_end_ns - t_client_start_ns)/1_000_000:.2f} ms")
    print("================================")

    print("🧹 Closing positions...")
    execution.execute_exit_parallel(size, spot_px, perp_px, spot_id, "HYPE")


if __name__ == "__main__":
    main()
```

### `tools/test_raw_speed.py`

```python
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
```

### `tools/test_spot_latency.py`

```python
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.connector import HyperliquidConnector
from src.execution import ExecutionManager


def _round_to_sig_figs(value: float, sig_figs: int = 5) -> float:
    if value == 0:
        return 0.0
    return float(f"{value:.{sig_figs}g}")


def extract_oid(response: Any) -> Optional[int]:
    if not isinstance(response, dict):
        return None
    payload = response.get("response") or response
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    statuses = data.get("statuses")
    for status in statuses:
        if "oid" in status:
            return status["oid"]
        filled = status.get("filled")
        if isinstance(filled, dict) and "oid" in filled:
            return filled["oid"]
    return None


def extract_server_time(details: Dict[str, Any]) -> Optional[int]:
    if not details:
        return None
    if "time" in details:
        return details["time"]
    order_root = details.get("order")
    if isinstance(order_root, dict):
        if "time" in order_root:
            return order_root["time"]
        inner = order_root.get("order")
        if isinstance(inner, dict) and "timestamp" in inner:
            return inner["timestamp"]
    return None


def main() -> None:
    connector = HyperliquidConnector()
    exchange = connector.exchange

    spot_id = connector.get_spot_asset_id("HYPE")
    mids = connector.info.all_mids()
    spot_px = float(mids[spot_id])
    min_usd_value = 10.8  # min 10 USDC + small buffer
    usdc_balance = 0.0
    spot_state = connector.info.spot_user_state(connector.master_address)
    for bal in spot_state.get("balances", []):
        if bal.get("coin") == "USDC":
            usdc_balance = float(bal.get("total", 0.0))
            break
    max_affordable = usdc_balance * 0.8 / spot_px if spot_px > 0 else 0
    desired = 11.0 / spot_px
    min_size = min_usd_value / spot_px
    raw_size = min(max_affordable, desired)
    size = round(max(raw_size, min_size), 2)
    if size * spot_px < min_usd_value:
        print("❌ Not enough USDC balance (need at least ~10 USDC).")
        return

    print(f"🔥 Spot Latency Test: {size} HYPE (≈${size * spot_px:.2f})")
    t_client_start_ns = time.time_ns()
    response = exchange.order(
        name=spot_id,
        is_buy=True,
        sz=size,
        limit_px=_round_to_sig_figs(spot_px * 1.05),
        order_type={"limit": {"tif": "Ioc"}},
        reduce_only=False,
    )
    t_client_end_ns = time.time_ns()

    oid = extract_oid(response)
    if not oid:
        print("❌ Could not extract OID from response.")
        print(response)
        return

    server_time = None
    try:
        details = connector.info.query_order_by_oid(connector.master_address, oid)
        server_time = extract_server_time(details)
    except Exception:
        details = None

    if server_time is None:
        print("⚠️ query_order_by_oid returned no timestamp; falling back to fills.")
        time.sleep(1)
        fills = connector.info.user_fills(connector.master_address)
        for fill in fills:
            if float(fill.get("sz", 0)) == size and fill.get("coin") == spot_id:
                server_time = fill.get("time")
                break
        if server_time is None:
            print("❌ Could not extract server timestamp even from fills.")
            print("details:", details)
            return

    t_client_start_ms = t_client_start_ns / 1_000_000
    spot_one_way = server_time - t_client_start_ms

    print("\n=== SPOT LATENCY REPORT ===")
    print(f"Client send ms : {t_client_start_ms:.3f}")
    print(f"Exchange time  : {server_time}")
    print(f"One-way latency: {spot_one_way:.2f} ms")
    print(f"Client RTT     : {(t_client_end_ns - t_client_start_ns)/1_000_000:.2f} ms")


if __name__ == "__main__":
    main()
```

