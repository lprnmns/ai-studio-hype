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

