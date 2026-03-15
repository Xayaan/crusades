"""Miner CLI for Templar Crusades.

URL-Based Architecture:
1. Host your train.py code at any URL (Gist, Pastebin, raw GitHub, etc.)
2. Run: miner submit <code_url>
3. The URL is timelock encrypted on blockchain
4. After reveal, validators fetch and evaluate your code

The URL acts as a secret - only those who know it can access.
Timelock encryption keeps it hidden until reveal_blocks pass.
"""

import argparse
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request

import bittensor as bt

from crusades.chain.payment import find_payment_extrinsic, resolve_payment_address
from crusades.config import HParams


def validate_code_url(url: str) -> tuple[bool, str, str | None]:
    """Validate that the URL points to a SINGLE valid train.py file.

    Accepts ANY URL that returns valid Python code with inner_steps function.
    Examples: GitHub Gist, raw GitHub, Pastebin, any HTTP/HTTPS URL.

    IMPORTANT: Must be a single file URL, not a folder/directory or repo.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message_or_validated_url, code_hash_or_none)
        code_hash is sha256 hex digest of the code content, included in commitment
        so validators can verify URL content hasn't changed after commit.
    """
    if not url:
        return False, "URL cannot be empty", None

    # Must be HTTP or HTTPS
    if not url.startswith("http://") and not url.startswith("https://"):
        return False, "URL must start with http:// or https://", None

    # Block obvious directory/folder URLs
    blocked_patterns = [
        "/tree/",  # GitHub repo tree
        "/blob/",  # GitHub blob without raw - redirect to raw
        "?tab=",  # GitHub tab navigation
        "/commits",  # GitHub commits page
        "/pulls",  # GitHub PRs
        "/issues",  # GitHub issues
        "/actions",  # GitHub actions
    ]
    for pattern in blocked_patterns:
        if pattern in url.lower():
            return (
                False,
                "URL appears to be a folder/page, not a single file. Use raw file URL.",
                None,
            )

    # Normalize gist URLs to the short gist.github.com/{user}/{id} form for
    # commitment (saves ~15-30 chars vs gist.githubusercontent.com/.../raw/...).
    # The validator expands to the raw form before fetching.
    # Handles all variants: .../raw, .../raw/revision, .../raw/revision/filename
    commit_url = url.rstrip("/")
    gist_match = re.match(
        r"https?://gist\.githubusercontent\.com/([^/]+/[0-9a-f]+)", commit_url, re.I
    )
    if gist_match:
        commit_url = f"https://gist.github.com/{gist_match.group(1)}"

    # Build raw fetch URL for validation
    fetch_url = commit_url
    if "gist.github.com" in fetch_url.lower() and "/raw" not in fetch_url.lower():
        fetch_url = re.sub(
            r"gist\.github\.com", "gist.githubusercontent.com", fetch_url, flags=re.I
        )
        if not fetch_url.endswith("/raw"):
            fetch_url = fetch_url.rstrip("/") + "/raw"

    # Verify the URL is accessible and contains valid code
    try:
        req = urllib.request.Request(fetch_url, headers={"User-Agent": "templar-crusades"})
        with urllib.request.urlopen(req, timeout=10) as response:
            code = response.read().decode("utf-8")

            # Check it's not HTML (indicates folder/page, not file)
            if "<html" in code.lower()[:500] or "<!doctype html" in code.lower()[:500]:
                return False, "URL returns HTML page, not a code file. Use raw file URL.", None

            # Check it's not JSON (could be API response listing files)
            if code.strip().startswith("{") and '"files"' in code[:500]:
                return (
                    False,
                    "URL returns JSON (possibly file listing), not a single code file.",
                    None,
                )

            # Basic validation - must contain inner_steps
            if "def inner_steps" not in code:
                return False, "Code must contain 'def inner_steps' function", None

            # Size sanity check (single file should be < 100KB typically)
            if len(code) > 500_000:  # 500KB max
                return (
                    False,
                    f"File too large ({len(code)} bytes). Max 500KB for single train.py",
                    None,
                )

            # Compute code hash for commitment integrity verification
            # Truncated to 128-bit (32 hex chars) — 2^64 birthday-attack collision resistance
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:32]

            print(f"   [OK] URL accessible ({len(code)} bytes)")
            print("   [OK] Single file detected")
            print("   [OK] Contains inner_steps function")
            print(f"   [OK] Code hash: {code_hash}")

    except urllib.error.HTTPError as e:
        return False, f"Cannot access URL: HTTP {e.code}", None
    except urllib.error.URLError as e:
        return False, f"Cannot access URL: {e.reason}", None
    except Exception as e:
        return False, f"Error validating URL: {e}", None

    return True, commit_url, code_hash


def _resolve_burn_hotkey(subtensor: bt.subtensor, netuid: int, burn_uid: int) -> str | None:
    """Look up the hotkey registered at burn_uid on the subnet."""
    try:
        metagraph = subtensor.metagraph(netuid)
        if burn_uid < len(metagraph.hotkeys):
            return metagraph.hotkeys[burn_uid]
        return None
    except Exception:
        return None


def pay_submission_fee(
    wallet: bt.wallet,
    subtensor: bt.subtensor,
    netuid: int,
    fee_rao: int,
    burn_uid: int,
    payment_coldkey: str,
) -> tuple[bool, dict | str]:
    """Pay submission fee by staking TAO then transferring the alpha to the owner.

    Two-step process:
    1. add_stake: converts TAO to alpha on the burn_uid's hotkey
    2. transfer_stake: moves alpha ownership to the subnet operator's coldkey

    The transfer_stake makes the payment irreversible — the miner cannot
    reclaim the alpha after it's transferred to a different coldkey.

    Args:
        wallet: Bittensor wallet (coldkey signs both transactions)
        subtensor: Subtensor connection
        netuid: Subnet to stake into
        fee_rao: Amount in RAO (1 TAO = 1e9 RAO)
        burn_uid: UID of the burn address on the subnet
        payment_coldkey: Destination coldkey SS58 address (owner)

    Returns:
        Tuple of (success, result_dict_or_error_message)
    """
    fee_tao = fee_rao / 1e9
    tx_fee_buffer_rao = 100_000_000  # 0.1 TAO reserve for two tx fees

    burn_hotkey = _resolve_burn_hotkey(subtensor, netuid, burn_uid)
    if burn_hotkey is None:
        return False, f"Could not resolve hotkey for burn_uid {burn_uid} on subnet {netuid}"

    try:
        coldkey_addr = wallet.coldkey.ss58_address
        balance = subtensor.get_balance(coldkey_addr)
        balance_rao = balance.rao if hasattr(balance, "rao") else int(balance)
        required_rao = fee_rao + tx_fee_buffer_rao

        if balance_rao < required_rao:
            return False, (
                f"Insufficient balance. Need {fee_tao:.4f} TAO "
                f"+ ~{tx_fee_buffer_rao / 1e9:.2f} TAO tx fees, "
                f"have {balance_rao / 1e9:.4f} TAO"
            )
    except Exception as e:
        return False, f"Failed to check balance: {e}"

    print(f"\n{'=' * 60}")
    print("SUBMISSION FEE (alpha transfer)")
    print(f"{'=' * 60}")
    print(f"   Amount: {fee_tao:.4f} TAO")
    print(f"   Subnet: {netuid}")
    print(f"   Burn UID: {burn_uid}")
    print(f"   Burn hotkey: {burn_hotkey[:16]}...")
    print(f"   Owner coldkey: {payment_coldkey[:16]}...")
    print(f"   Your balance: {balance_rao / 1e9:.4f} TAO")
    print(f"\nThis stakes {fee_tao:.4f} TAO as alpha, then transfers it to the")
    print("subnet operator's coldkey (irreversible).")

    confirm = input("\nProceed with payment? [y/N]: ").strip().lower()
    if confirm != "y":
        return False, "Payment cancelled by user"

    # Step 1: Stake TAO → alpha
    print("\n[1/2] Staking TAO to create alpha...")

    # Snapshot pre-existing alpha so we only transfer the newly created amount.
    # get_stake returns the TOTAL balance for (coldkey, hotkey, netuid); without
    # this delta, a retry after a failed transfer_stake would sweep all alpha.
    try:
        pre_stake = subtensor.get_stake(
            coldkey_ss58=coldkey_addr, hotkey_ss58=burn_hotkey, netuid=netuid
        )
        pre_alpha_rao = pre_stake.rao if hasattr(pre_stake, "rao") else int(pre_stake)
    except Exception:
        pre_alpha_rao = 0

    try:
        amount = bt.Balance.from_rao(fee_rao)
        stake_ok = subtensor.add_stake(
            wallet=wallet,
            hotkey_ss58=burn_hotkey,
            netuid=netuid,
            amount=amount,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if not stake_ok:
            return False, "add_stake transaction failed"
    except Exception as e:
        return False, f"add_stake error: {e}"

    # Determine how much new alpha was created (post - pre)
    post_stake = subtensor.get_stake(
        coldkey_ss58=coldkey_addr, hotkey_ss58=burn_hotkey, netuid=netuid
    )
    post_alpha_rao = post_stake.rao if hasattr(post_stake, "rao") else int(post_stake)
    new_alpha_rao = post_alpha_rao - pre_alpha_rao

    if new_alpha_rao <= 0:
        return False, "add_stake succeeded but no new alpha balance detected"
    print(
        f"      Alpha created: {new_alpha_rao / 1e9:.4f} (total on hotkey: {post_alpha_rao / 1e9:.4f})"
    )

    transfer_amount = bt.Balance.from_rao(new_alpha_rao)

    # Step 2: Transfer only the new alpha to the owner's coldkey
    print("[2/2] Transferring alpha to subnet operator...")
    try:
        transfer_ok = subtensor.transfer_stake(
            wallet=wallet,
            destination_coldkey_ss58=payment_coldkey,
            hotkey_ss58=burn_hotkey,
            origin_netuid=netuid,
            destination_netuid=netuid,
            amount=transfer_amount,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        if not transfer_ok:
            return (
                False,
                "transfer_stake transaction failed (alpha still staked under your coldkey)",
            )
    except Exception as e:
        return False, f"transfer_stake error: {e} (alpha still staked under your coldkey)"

    # Locate the transfer_stake extrinsic so it can be embedded in the
    # commitment for O(1) validator verification.
    # Retry with increasing lookback to handle RPC lag / finalization delay.
    payment_ref = None
    for attempt, lookback in enumerate((5, 10, 20), start=1):
        payment_ref = find_payment_extrinsic(
            subtensor=subtensor,
            miner_coldkey=coldkey_addr,
            payment_address=payment_coldkey,
            netuid=netuid,
            lookback_blocks=lookback,
        )
        if payment_ref is not None:
            break
        if attempt < 3:
            print(
                f"\n   Retrying payment extrinsic lookup (attempt {attempt}/3, lookback={lookback})..."
            )
            time.sleep(3)

    if payment_ref is None:
        return (
            False,
            "Could not locate payment extrinsic on-chain after retries. "
            "The payment was sent but cannot be referenced — validator will reject. "
            "Try again or contact support.",
        )

    payment_block, payment_index = payment_ref
    print(f"\n   Payment extrinsic: block {payment_block}, index {payment_index}")

    current_block = subtensor.get_current_block()
    block_hash = subtensor.get_block_hash(current_block)

    result = {
        "amount_rao": fee_rao,
        "alpha_rao": new_alpha_rao,
        "block": current_block,
        "block_hash": block_hash,
        "burn_hotkey": burn_hotkey,
        "payment_coldkey": payment_coldkey,
        "netuid": netuid,
        "payment_block": payment_block,
        "payment_index": payment_index,
    }

    print("\n[OK] Submission fee paid!")
    print(f"   TAO spent: {fee_tao:.4f}")
    print(f"   Alpha transferred: {new_alpha_rao / 1e9:.4f}")
    print(f"   Block: {current_block}")
    print(f"   Block hash: {block_hash}")
    print("\n   SAVE THESE DETAILS - they are your proof of payment for disputes")

    return True, result


def commit_to_chain(
    wallet: bt.wallet,
    code_url: str,
    code_hash: str | None = None,
    network: str = "finney",
) -> tuple[bool, dict | str]:
    """Commit code URL to blockchain (timelock encrypted).

    The URL is encrypted via drand and only revealed after reveal_blocks.
    This keeps your code location private until evaluation time.

    Args:
        wallet: Bittensor wallet
        code_url: URL containing train.py code
        code_hash: SHA256 hex digest of code content (for integrity verification)
        network: Subtensor network (finney, test, or local)

    Returns:
        Tuple of (success, result_dict or error_message)
    """
    if not code_hash:
        return False, "code_hash is required (run validate_code_url first)"

    # Load settings from hparams
    hparams = HParams.load()
    netuid = hparams.netuid
    blocks_until_reveal = hparams.reveal_blocks
    block_time = hparams.block_time

    # Connect to blockchain first to check registration
    print(f"\nConnecting to {network}...")
    try:
        subtensor = bt.subtensor(network=network)
        current_block = subtensor.get_current_block()
        print(f"   Current block: {current_block}")
    except Exception as e:
        return False, f"Failed to connect to {network}: {e}"

    # Check if hotkey is registered on subnet
    hotkey = wallet.hotkey.ss58_address
    if not subtensor.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey):
        return False, f"Hotkey {hotkey} is not registered on subnet {netuid}"

    # Get miner UID
    uid = subtensor.get_uid_for_hotkey_on_subnet(hotkey_ss58=hotkey, netuid=netuid)
    print(f"   Miner UID: {uid}")

    # Pre-flight: check commitment size before taking payment.
    # Format: "<32hex>:<block>:<index>:<url>" (payment) or "<32hex>:<url>" (no payment)
    # Use actual current_block to estimate block digits; add 1 digit of headroom for index.
    max_commitment = 128
    is_burn_uid = uid == hparams.burn_uid  # subnet operator — exempt from fees
    if hparams.payment.enabled and not is_burn_uid:
        block_digits = len(str(current_block)) + 1  # +1 for power-of-10 rollover
        index_digits = 3  # extrinsic index rarely exceeds 999
        overhead = len(code_hash) + 1 + block_digits + 1 + index_digits + 1
    else:
        overhead = len(code_hash) + 1
    if len(code_url) + overhead > max_commitment:
        max_url = max_commitment - overhead
        return False, (
            f"URL too long ({len(code_url)} chars, max ~{max_url}). "
            f"Commitment would exceed {max_commitment}-byte on-chain limit. "
            f"Use a shorter URL."
        )

    # --- Payment step ---
    if is_burn_uid:
        print(f"\n   Submitting as burn_uid {hparams.burn_uid} — payment exempt.")
        pay_result_dict = None
    elif hparams.payment.enabled:
        fee_rao = hparams.payment.fee_rao

        # Use explicit payment_address if configured, otherwise derive from burn_uid
        if hparams.payment.payment_address:
            payment_coldkey = hparams.payment.payment_address
        else:
            payment_coldkey = resolve_payment_address(subtensor, netuid, hparams.burn_uid)
            if payment_coldkey is None:
                return (
                    False,
                    f"Payment failed: could not resolve payment address from burn_uid {hparams.burn_uid}",
                )

        print(
            f"\n--- SUBMISSION FEE ({fee_rao / 1e9:.4f} TAO as alpha → {payment_coldkey[:16]}...) ---"
        )

        pay_success, pay_result = pay_submission_fee(
            wallet=wallet,
            subtensor=subtensor,
            netuid=netuid,
            fee_rao=fee_rao,
            burn_uid=hparams.burn_uid,
            payment_coldkey=payment_coldkey,
        )

        if not pay_success:
            return False, f"Payment failed: {pay_result}"

        print("\n   Payment successful. Proceeding to commit...")
        pay_result_dict = pay_result  # type: dict
    else:
        print("\n   Payment disabled in hparams. Skipping fee.")
        pay_result_dict = None

    print("\nCommitting to blockchain...")
    print(f"   Network: {network}")
    print(f"   Subnet: {netuid} (from hparams.json)")
    print(f"   Hotkey: {wallet.hotkey.ss58_address}")
    print(f"   Reveal blocks: {blocks_until_reveal} (from hparams.json)")
    print(f"   Block time: {block_time}s (from hparams.json)")

    # Commitment data: packed format to fit 128-byte on-chain limit.
    #   <32hex>:<block>:<index>:<url>   (~46 bytes overhead, ~82 for URL)
    # Without payment ref (payment disabled):
    #   <32hex>:<url>                   (33 bytes overhead, 95 for URL)
    pay_block = pay_result_dict.get("payment_block") if pay_result_dict else None
    pay_index = pay_result_dict.get("payment_index") if pay_result_dict else None
    if pay_block is not None and pay_index is not None:
        commitment_data = f"{code_hash}:{pay_block}:{pay_index}:{code_url}"
    else:
        commitment_data = f"{code_hash}:{code_url}"

    print(f"   Commitment size: {len(commitment_data)} bytes")
    if len(commitment_data) > 128:
        return False, (
            f"Commitment too large ({len(commitment_data)} bytes, max 128). "
            f"Use a shorter URL (max ~95 chars)."
        )

    # Commit using timelock encryption (drand)
    print("\nCommitting to chain...")
    print("   Using set_reveal_commitment (timelock encrypted)")

    try:
        if not hasattr(subtensor, "set_reveal_commitment"):
            return False, "Subtensor does not support set_reveal_commitment()"

        success, reveal_round = subtensor.set_reveal_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            blocks_until_reveal=blocks_until_reveal,
            block_time=block_time,
        )

        if success:
            commit_block = subtensor.get_current_block()
            reveal_block = commit_block + blocks_until_reveal

            result = {
                "code_url": code_url,
                "commit_block": commit_block,
                "reveal_block": reveal_block,
                "reveal_round": reveal_round,
                "hotkey": wallet.hotkey.ss58_address,
                "netuid": netuid,
            }

            print("\n[OK] Commitment successful!")
            print(f"   Commit block: {commit_block}")
            print(f"   Reveal block: {reveal_block}")
            print(f"   Reveal round: {reveal_round}")
            print(f"\nValidators will evaluate after block {reveal_block}")

            return True, result
        else:
            return False, "Commitment transaction failed"

    except Exception as e:
        return False, f"Blockchain error: {e}"


def cmd_submit(args):
    """Submit a code URL to the crusades."""
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    print("=" * 60)
    print("TEMPLAR CRUSADES - SUBMIT CODE")
    print("=" * 60)
    print(f"\nWallet: {args.wallet_name}/{args.wallet_hotkey}")
    print(f"Hotkey: {wallet.hotkey.ss58_address}")

    # Validate code URL
    print("\n--- STEP 1: VALIDATE CODE URL ---")
    print("Validating URL...")
    print(f"   URL: {args.code_url}")

    valid, result, code_hash = validate_code_url(args.code_url)
    if not valid:
        print(f"\n[FAILED] Invalid URL: {result}")
        return 1

    final_url = result
    print(f"   Final URL: {final_url}")

    # Commit to blockchain
    print("\n--- STEP 2: COMMIT TO BLOCKCHAIN ---")

    success, result = commit_to_chain(
        wallet=wallet,
        code_url=final_url,
        code_hash=code_hash,
        network=args.network,
    )

    if success:
        print("\n" + "=" * 60)
        print("SUBMISSION COMPLETE!")
        print("=" * 60)
        print("\nYour code URL is now timelock encrypted on the blockchain.")
        print(f"After block {result['reveal_block']}, validators will:")
        print("  1. Verify your submission fee payment on-chain")
        print("  2. Decrypt and retrieve your code URL")
        print("  3. Fetch your train.py code")
        print("  4. Evaluate and score your submission")
        print("\nWARNING: Do NOT delete or modify your code until evaluation is complete!")
        return 0
    else:
        print(f"\n[FAILED] Commit failed: {result}")
        payment_failed = isinstance(result, str) and result.startswith("Payment failed")
        if not payment_failed:
            try:
                _hparams = HParams.load()
                if _hparams.payment.enabled:
                    print("\nNOTE: Your submission fee was already paid.")
                    print(
                        "If you need a refund, contact the validator operator "
                        "with your payment details above."
                    )
            except Exception as e:
                print(f"\n(Could not check payment status: {e})")
        return 1


def cmd_status(args):
    """Check blockchain status and connection."""
    try:
        print(f"\nConnecting to {args.network}...")
        subtensor = bt.subtensor(network=args.network)
        current_block = subtensor.get_current_block()

        # Load hparams
        hparams = HParams.load()

        print("\n[OK] Connected to blockchain")
        print(f"   Network: {args.network}")
        print(f"   Current block: {current_block}")
        print(f"   Subnet: {hparams.netuid}")
        print(f"   Reveal blocks: {hparams.reveal_blocks}")
        print(f"   Block time: {hparams.block_time}s")

        # Check if subnet exists
        if subtensor.subnet_exists(hparams.netuid):
            print(f"\n[OK] Subnet {hparams.netuid} exists")
            try:
                meta = bt.metagraph(netuid=hparams.netuid, network=args.network)
                print(f"   Neurons: {meta.n.item()}")
            except Exception as e:
                print(f"   (Could not fetch neuron count: {e})")
        else:
            print(f"\n[WARNING] Subnet {hparams.netuid} does not exist on {args.network}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1


def cmd_validate(args):
    """Validate a code URL without submitting."""
    print("=" * 60)
    print("VALIDATE CODE URL")
    print("=" * 60)
    print(f"\nValidating: {args.code_url}")

    valid, result, code_hash = validate_code_url(args.code_url)

    if valid:
        print("\n[OK] URL is valid!")
        print(f"   Final URL: {result}")
        if code_hash:
            print(f"   Code hash: {code_hash}")
            packed_size = len(f"{code_hash}:{result}")
            print(f"   Packed commitment: {packed_size}/128 bytes")
            if packed_size > 128:
                print(f"   [WARNING] Too large! Shorten URL by {packed_size - 128} chars")
        print("\nTo submit, run:")
        print(
            f"  uv run -m neurons.miner submit '{result}' --wallet.name <name> --wallet.hotkey <hotkey>"
        )
        return 0
    else:
        print(f"\n[FAILED] Invalid: {result}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Templar Crusades Miner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How to Submit:

  1. Host your train.py code at any URL:
     - GitHub Gist (secret recommended)
     - Raw GitHub file
     - Pastebin or any paste service
     - Any HTTP/HTTPS URL that returns the code

  2. Submit to the crusades:
     uv run -m neurons.miner submit <code_url> \\
         --wallet.name miner --wallet.hotkey default --network finney

  3. Your code URL is timelock encrypted - validators can only see it
     after reveal_blocks pass.

Examples:
  # Validate a URL without submitting
  uv run -m neurons.miner validate https://example.com/train.py

  # Submit to mainnet
  uv run -m neurons.miner submit https://example.com/train.py \\
      --wallet.name miner --wallet.hotkey default --network finney

  # Check blockchain status
  uv run -m neurons.miner status --network finney

Settings from hparams.json:
  netuid        - Subnet ID
  reveal_blocks - Blocks until commitment revealed
  block_time    - Seconds per block (for drand calculation)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # SUBMIT command
    submit_parser = subparsers.add_parser("submit", help="Submit a code URL to the crusades")
    submit_parser.add_argument("code_url", help="URL containing your train.py code")
    submit_parser.add_argument("--wallet.name", dest="wallet_name", default="default")
    submit_parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="default")
    submit_parser.add_argument(
        "--network", default="finney", help="Network: finney (mainnet), test, or local"
    )
    submit_parser.set_defaults(func=cmd_submit)

    # VALIDATE command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a code URL without submitting"
    )
    validate_parser.add_argument("code_url", help="URL to validate")
    validate_parser.set_defaults(func=cmd_validate)

    # STATUS command
    status_parser = subparsers.add_parser("status", help="Check blockchain status")
    status_parser.add_argument(
        "--network", default="finney", help="Network: finney (mainnet), test, or local"
    )
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
