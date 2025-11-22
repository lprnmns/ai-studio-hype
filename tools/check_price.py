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

