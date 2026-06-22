from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.ebay_api_probe.http_client import EbayHttpProbeClient
from tools.ebay_api_probe.probe import EbayApiProbe


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    scopes = args.scope
    client = EbayHttpProbeClient(sandbox=False, token_env=args.token_env, marketplace_id=args.marketplace_id)
    report = EbayApiProbe(client, marketplace_id=args.marketplace_id, sandbox=False).run(
        scopes=scopes,
        seller_owned_listing_id=args.seller_owned_listing_id,
        buyer_participated_listing_id=args.buyer_participated_listing_id,
        unrelated_listing_id=args.unrelated_listing_id,
        authorized_production_user_token=True,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a read-only eBay API field-availability probe")
    parser.add_argument("--mode", choices=["production"], required=True)
    parser.add_argument("--token-env", required=True, help="Environment variable containing the OAuth user access token")
    parser.add_argument("--marketplace-id", default="EBAY_US")
    parser.add_argument("--scope", action="append", required=True, help="Authorized OAuth scope. Repeat for multiple scopes.")
    parser.add_argument("--seller-owned-listing-id", required=True)
    parser.add_argument("--buyer-participated-listing-id", required=True)
    parser.add_argument("--unrelated-listing-id", required=True)
    parser.add_argument("--output")
    return parser


if __name__ == "__main__":
    main()

