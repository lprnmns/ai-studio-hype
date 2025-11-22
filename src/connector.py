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

