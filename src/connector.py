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
        token_id = token_meta.get("tokenId")
        if token_id:
            return token_id
        index = token_meta.get("index")
        if index is None:
            raise ValueError("Token metadata missing both 'tokenId' and 'index'.")
        return f"@{index}"

    @staticmethod
    def _format_asset_index(index: Optional[int]) -> str:
        if index is None:
            raise ValueError("Spot meta entry missing 'index'.")
        return f"@{index}"

    def get_spot_asset_id(self, symbol: str) -> str:
        """Resolve the internal asset identifier for a given spot symbol."""
        symbol_upper = symbol.upper()
        meta = self.info.spot_meta()
        tokens: List[Dict[str, Any]] = meta.get("tokens", [])

        for universe_entry in meta.get("universe", []):
            base_idx, quote_idx = universe_entry.get("tokens", [None, None])
            candidate_tokens: List[Dict[str, Any]] = []
            if base_idx is not None and base_idx < len(tokens):
                candidate_tokens.append(tokens[base_idx])
            if quote_idx is not None and quote_idx < len(tokens):
                candidate_tokens.append(tokens[quote_idx])

            for token_meta in candidate_tokens:
                if token_meta.get("name", "").upper() == symbol_upper:
                    return self._token_identifier(token_meta)

            pair_name = universe_entry.get("name", "")
            base_token = tokens[base_idx] if base_idx is not None and base_idx < len(tokens) else None
            quote_token = tokens[quote_idx] if quote_idx is not None and quote_idx < len(tokens) else None
            derived_pair_name = (
                f"{base_token.get('name')}/{quote_token.get('name')}"
                if base_token and quote_token
                else ""
            )

            if pair_name.upper() == symbol_upper or derived_pair_name.upper() == symbol_upper:
                return self._format_asset_index(universe_entry.get("index"))

        # Final fallback: check raw token list
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

