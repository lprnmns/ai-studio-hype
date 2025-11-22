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

