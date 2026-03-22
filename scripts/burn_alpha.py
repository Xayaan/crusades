#!/usr/bin/env python3
"""Burn submission fee alpha via the on-chain burn_alpha extrinsic.

This script is intended for the coldkey owner (not the validator process)
to manually burn collected submission fees. The burn_alpha extrinsic must
be signed by the coldkey that holds the alpha stake.

Usage:
    # Burn based on verified submissions in DB (0.05 TAO per submission, converted to alpha)
    uv run scripts/burn_alpha.py --from-db /path/to/crusades.db --network finney --netuid 39 --burn-uid 0 --wallet-name default

    # Override per-submission fee
    uv run scripts/burn_alpha.py --from-db /path/to/crusades.db --fee-tao 0.1 --network finney --netuid 39 --burn-uid 0

    # Burn a specific rao amount directly
    uv run scripts/burn_alpha.py --amount 50000000 --network finney --netuid 39 --burn-uid 0

    # Burn on localnet
    uv run scripts/burn_alpha.py --amount 1000000 --network local --netuid 2 --burn-uid 0 --wallet-name V1 --wallet-hotkey V1
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import bittensor as bt


def resolve_hotkey_for_uid(subtensor: bt.subtensor, netuid: int, uid: int) -> str | None:
    """Get the hotkey SS58 address for a given UID on a subnet."""
    try:
        metagraph = subtensor.metagraph(netuid=netuid)
        if 0 <= uid < len(metagraph.hotkeys):
            return metagraph.hotkeys[uid]
    except Exception as e:
        print(f"  ERROR resolving hotkey for UID {uid}: {e}")
    return None


def count_verified_submissions(db_path: str) -> int:
    """Count verified submissions in the database (read-only)."""
    path = Path(db_path)
    if not path.exists():
        print(f"  ERROR: Database file not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT COUNT(*) FROM verified_payments").fetchone()
        return row[0]
    finally:
        conn.close()


def get_tao_to_alpha_rate(subtensor: bt.subtensor, netuid: int) -> float:
    """Query on-chain pool reserves and return the alpha-per-TAO spot rate."""
    tao_in = subtensor.substrate.query("SubtensorModule", "SubnetTAO", [netuid]).value
    alpha_in = subtensor.substrate.query("SubtensorModule", "SubnetAlphaIn", [netuid]).value

    if not tao_in or not alpha_in:
        print(f"  ERROR: Could not read pool reserves (tao_in={tao_in}, alpha_in={alpha_in})")
        sys.exit(1)

    rate = alpha_in / tao_in
    print(f"  Pool reserves:  tao_in={tao_in:,}  alpha_in={alpha_in:,}")
    print(f"  Spot rate:      1 TAO = {rate:,.4f} alpha")
    return rate


def main():
    parser = argparse.ArgumentParser(
        description="Burn submission fee alpha on-chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/burn_alpha.py --from-db ./crusades.db --network finney --netuid 39 --burn-uid 0
  uv run scripts/burn_alpha.py --from-db ./crusades.db --fee-tao 0.1 --network finney --netuid 39 --burn-uid 0
  uv run scripts/burn_alpha.py --amount 50000000 --network finney --netuid 39 --burn-uid 0
        """,
    )

    amount_source = parser.add_mutually_exclusive_group(required=True)
    amount_source.add_argument("--amount", type=int, help="Amount of alpha to burn (in rao)")
    amount_source.add_argument(
        "--from-db",
        type=str,
        metavar="DB_PATH",
        help="Count verified submissions in DB and compute burn from fee × count",
    )

    parser.add_argument(
        "--fee-tao",
        type=float,
        default=0.05,
        help="Per-submission fee in TAO (default: 0.05). Converted to alpha on-chain.",
    )
    parser.add_argument("--network", type=str, required=True, help="Network: finney, local, test")
    parser.add_argument("--netuid", type=int, required=True, help="Subnet UID")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--burn-uid", type=int, help="UID to resolve hotkey from")
    target.add_argument("--hotkey", type=str, help="Hotkey SS58 to burn on")
    parser.add_argument("--wallet-name", default="default", help="Wallet name (default: 'default')")
    parser.add_argument(
        "--wallet-hotkey", default="default", help="Wallet hotkey name (default: 'default')"
    )
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets", help="Wallet path")
    args = parser.parse_args()

    print(f"\nConnecting to {args.network}...")
    sub = bt.subtensor(network=args.network)

    if args.from_db:
        num_submissions = count_verified_submissions(args.from_db)
        if num_submissions == 0:
            print("\n  No verified submissions found. Nothing to burn.")
            return

        fee_tao = args.fee_tao
        fee_rao = int(fee_tao * 1_000_000_000)

        print(f"\n  Verified submissions: {num_submissions}")
        print(f"  Fee per submission:   {fee_tao} TAO ({fee_rao:,} rao)")

        print(f"\n  Querying TAO→alpha exchange rate for netuid {args.netuid}...")
        rate = get_tao_to_alpha_rate(sub, args.netuid)

        alpha_per_fee = fee_tao * rate
        total_alpha = alpha_per_fee * num_submissions
        burn_amount = int(total_alpha * 1_000_000_000)

        print(f"\n  Alpha per {fee_tao} TAO fee: {alpha_per_fee:,.4f} alpha")
        print(
            f"  Total burn: {alpha_per_fee:,.4f} × {num_submissions} = {total_alpha:,.4f} alpha ({burn_amount:,} rao)"
        )
    else:
        if args.amount <= 0:
            print("ERROR: --amount must be positive")
            sys.exit(1)
        burn_amount = args.amount

    hotkey = args.hotkey
    if not hotkey:
        print(f"\n  Resolving hotkey for burn_uid {args.burn_uid} on subnet {args.netuid}...")
        hotkey = resolve_hotkey_for_uid(sub, args.netuid, args.burn_uid)
        if not hotkey:
            print(f"  ERROR: Could not resolve hotkey for UID {args.burn_uid}")
            sys.exit(1)

    alpha = burn_amount / 1_000_000_000
    print("\n" + "=" * 60)
    print("BURN ALPHA")
    print("=" * 60)
    print(f"  Network:     {args.network}")
    print(f"  Netuid:      {args.netuid}")
    print(f"  Hotkey:      {hotkey}")
    print(f"  Amount:      {burn_amount:,} rao ({alpha:.4f} alpha)")
    if args.from_db:
        print(
            f"  Source:      {num_submissions} submissions × {args.fee_tao} TAO from {args.from_db}"
        )
    print(f"  Wallet:      {args.wallet_name}")
    print("=" * 60)

    print(f"\n  Loading wallet '{args.wallet_name}'...")
    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path,
    )
    coldkey_ss58 = wallet.coldkeypub.ss58_address
    print(f"  Coldkey: {coldkey_ss58}")

    print("\n  Querying alpha stake before burn...")
    before_stake = None
    try:
        stake_info = sub.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey,
            netuids=[args.netuid],
        )
        if args.netuid in stake_info:
            before_stake = stake_info[args.netuid].stake
            print(f"  Alpha stake BEFORE: {before_stake}")
        else:
            print(f"  No stake found on netuid {args.netuid}")
    except Exception as e:
        print(f"  Could not query stake: {e}")

    call = sub.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burn_alpha",
        call_params={
            "hotkey": hotkey,
            "amount": burn_amount,
            "netuid": args.netuid,
        },
    )

    confirm = (
        input(f"\n  Burn {burn_amount:,} rao ({alpha:.4f} alpha) on netuid {args.netuid}? [y/N] ")
        .strip()
        .lower()
    )
    if confirm != "y":
        print("  Aborted.")
        return

    print("\n  Submitting burn_alpha extrinsic...")
    try:
        success, msg = sub.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if success:
            print(f"\n  SUCCESS: Burned {burn_amount:,} rao ({alpha:.4f} alpha)")
            print(f"  Message: {msg}")
        else:
            print(f"\n  FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print("\n  Querying alpha stake after burn...")
    try:
        stake_info = sub.get_stake_for_coldkey_and_hotkey(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey,
            netuids=[args.netuid],
        )
        if args.netuid in stake_info:
            after_stake = stake_info[args.netuid].stake
            print(f"  Alpha stake AFTER:  {after_stake}")
            if before_stake is not None:
                print(
                    f"  Difference:         {before_stake - after_stake} (expected: ~{burn_amount:,} rao)"
                )
        else:
            print(f"  No stake info after burn on netuid {args.netuid}")
    except Exception as e:
        print(f"  Could not query stake after burn: {e}")


if __name__ == "__main__":
    main()
