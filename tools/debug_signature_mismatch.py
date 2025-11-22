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


