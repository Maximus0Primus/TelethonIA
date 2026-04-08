"""
v119: Jupiter Trigger V2 API — on-chain stop-loss/trailing stop orders.

Places trigger orders via Jupiter keepers that execute instantly when price
hits the SL level. No polling delay — keepers monitor on-chain prices.

Architecture:
  After buy → place SL order (tokens deposited to Jupiter vault)
  Monitor loop → PATCH SL price upward as trail moves
  Keeper fills → poll detects fill → update DB
  Timeout → cancel order → market sell via Ultra

Fallback: if any trigger API call fails, live_trader falls back to
polling + Ultra swap (current behavior). trigger_order_id=NULL in DB.
"""

import os
import time
import base64
import logging

import requests
import base58
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIGGER_BASE = "https://api.jup.ag/trigger/v2"
WSOL_MINT = "So11111111111111111111111111111111111111112"
TIMEOUT = 15  # seconds for API calls

# ---------------------------------------------------------------------------
# Module-level cached state (follows _rt_config pattern in safe_scraper.py)
# ---------------------------------------------------------------------------

_jwt_token: str | None = None
_jwt_expires_at: float = 0.0       # epoch seconds
_vault_address: str | None = None   # permanent once registered
_keypair: Keypair | None = None


def _get_keypair() -> Keypair | None:
    """Lazy-load Solana keypair from env (same key as live_trader)."""
    global _keypair
    if _keypair is not None:
        return _keypair
    pk_str = os.environ.get("SOLANA_PRIVATE_KEY", "")
    if not pk_str:
        return None
    try:
        _keypair = Keypair.from_bytes(base58.b58decode(pk_str))
        return _keypair
    except Exception as e:
        logger.error("jupiter_trigger: failed to load keypair: %s", e)
        return None


def _get_api_key() -> str:
    return os.environ.get("JUPITER_API_KEY", "")


def _base_headers() -> dict:
    """Headers without auth (for challenge endpoint)."""
    h = {"Content-Type": "application/json"}
    api_key = _get_api_key()
    if api_key:
        h["x-api-key"] = api_key
    return h


# ---------------------------------------------------------------------------
# Auth: challenge-response JWT (24h TTL)
# ---------------------------------------------------------------------------

def _ensure_auth() -> str | None:
    """Get valid JWT, refreshing if needed. Returns token or None."""
    global _jwt_token, _jwt_expires_at

    # Return cached if still valid (refresh 5min before expiry)
    if _jwt_token and time.time() < _jwt_expires_at - 300:
        return _jwt_token

    kp = _get_keypair()
    if not kp:
        return None

    wallet = str(kp.pubkey())
    headers = _base_headers()

    try:
        # Step 1: Request challenge
        resp = requests.post(
            f"{TRIGGER_BASE}/auth/challenge",
            headers=headers,
            json={"walletPubkey": wallet, "type": "message"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        challenge = resp.json().get("challenge")
        if not challenge:
            logger.warning("jupiter_trigger: auth challenge empty")
            return None

        # Step 2: Sign challenge message
        message_bytes = challenge.encode("utf-8")
        signature = kp.sign_message(message_bytes)
        sig_b58 = base58.b58encode(bytes(signature)).decode()

        # Step 3: Verify and get JWT
        resp2 = requests.post(
            f"{TRIGGER_BASE}/auth/verify",
            headers=headers,
            json={
                "type": "message",
                "walletPubkey": wallet,
                "signature": sig_b58,
            },
            timeout=TIMEOUT,
        )
        resp2.raise_for_status()
        token = resp2.json().get("token")
        if not token:
            logger.warning("jupiter_trigger: auth verify returned no token")
            return None

        _jwt_token = token
        _jwt_expires_at = time.time() + 86400  # 24h
        logger.info("jupiter_trigger: authenticated (wallet: %s)", wallet[:8])
        return _jwt_token

    except Exception as e:
        logger.warning("jupiter_trigger: auth failed: %s", e)
        return None


def _auth_headers() -> dict | None:
    """Full headers with JWT auth. Returns None if auth fails."""
    token = _ensure_auth()
    if not token:
        return None
    h = _base_headers()
    h["Authorization"] = f"Bearer {token}"
    return h


# ---------------------------------------------------------------------------
# Vault management
# ---------------------------------------------------------------------------

def _get_vault() -> str | None:
    """Get or register vault address (cached permanently)."""
    global _vault_address
    if _vault_address:
        return _vault_address

    headers = _auth_headers()
    if not headers:
        return None

    try:
        # Try to get existing vault
        resp = requests.get(f"{TRIGGER_BASE}/vault", headers=headers, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            vault = data.get("publicKey") or data.get("vaultPubkey") or data.get("address")
            if vault:
                _vault_address = vault
                logger.info("jupiter_trigger: vault found: %s", vault[:12])
                return vault

        # Register new vault
        resp2 = requests.get(f"{TRIGGER_BASE}/vault/register", headers=headers, timeout=TIMEOUT)
        resp2.raise_for_status()
        data2 = resp2.json()
        vault = data2.get("publicKey") or data2.get("vaultPubkey") or data2.get("address")
        if vault:
            _vault_address = vault
            logger.info("jupiter_trigger: vault registered: %s", vault[:12])
            return vault

        logger.warning("jupiter_trigger: vault response has no address: %s", data2)
        return None

    except Exception as e:
        logger.warning("jupiter_trigger: vault setup failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Transaction signing helper
# ---------------------------------------------------------------------------

def _sign_transaction(tx_base64: str) -> str | None:
    """Partial-sign a base64-encoded versioned transaction.

    v120: The tx from Jupiter may already contain signatures from other
    signers (vault, program). We must preserve those and only fill in our
    own slot — identified by matching our pubkey in the account keys.
    """
    kp = _get_keypair()
    if not kp:
        return None
    try:
        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        msg = tx.message

        # Sign the serialized message with our keypair
        # VersionedTransaction (v0) signs with 0x80 prefix before message bytes
        our_sig = kp.sign_message(b"\x80" + bytes(msg))
        our_pubkey = kp.pubkey()

        # Find our pubkey's position in the static account keys
        # (signers are always in the first num_required_signatures slots)
        sigs = list(tx.signatures)
        found = False
        for i, key in enumerate(msg.account_keys):
            if key == our_pubkey:
                sigs[i] = our_sig
                found = True
                break

        if not found:
            logger.warning("jupiter_trigger: our pubkey %s not in tx account_keys", our_pubkey)
            return None

        signed_tx = VersionedTransaction.populate(msg, sigs)
        return base64.b64encode(bytes(signed_tx)).decode()
    except Exception as e:
        logger.warning("jupiter_trigger: tx signing failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Order CRUD
# ---------------------------------------------------------------------------

def place_stop_loss(
    ca: str,
    amount_tokens: int,
    sl_price_usd: float,
    expiry_seconds: int = 14400,
    sl_slippage_bps: int = 2000,
) -> dict | None:
    """
    Place a stop-loss trigger order: sell tokens when price drops below sl_price_usd.

    Flow: get vault → craft deposit → sign → create order
    Returns {"order_id": str} or None on failure.
    """
    headers = _auth_headers()
    if not headers:
        return None

    vault = _get_vault()
    if not vault:
        return None

    kp = _get_keypair()
    if not kp:
        return None
    wallet = str(kp.pubkey())

    try:
        # Step 1: Craft deposit transaction
        resp = requests.post(
            f"{TRIGGER_BASE}/deposit/craft",
            headers=headers,
            json={
                "inputMint": ca,
                "outputMint": WSOL_MINT,
                "userAddress": wallet,
                "amount": str(amount_tokens),
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("jupiter_trigger: deposit/craft %d: %s", resp.status_code, resp.text[:500])
            return None
        deposit = resp.json()

        tx_b64 = deposit.get("transaction")
        request_id = deposit.get("requestId")
        if not tx_b64 or not request_id:
            logger.warning("jupiter_trigger: deposit craft missing fields: %s",
                           list(deposit.keys()))
            return None

        # Step 2: Sign deposit transaction
        signed_b64 = _sign_transaction(tx_b64)
        if not signed_b64:
            return None

        # Step 3: Create stop-loss order
        expires_at_ms = int((time.time() + expiry_seconds) * 1000)

        resp2 = requests.post(
            f"{TRIGGER_BASE}/orders/price",
            headers=headers,
            json={
                "orderType": "single",
                "depositRequestId": request_id,
                "depositSignedTx": signed_b64,
                "userPubkey": wallet,
                "inputMint": ca,
                "inputAmount": str(amount_tokens),
                "outputMint": WSOL_MINT,
                "triggerMint": ca,
                "triggerCondition": "below",
                "triggerPriceUsd": sl_price_usd,
                "slippageBps": sl_slippage_bps,
                "expiresAt": expires_at_ms,
            },
            timeout=TIMEOUT,
        )
        if resp2.status_code != 200:
            logger.warning("jupiter_trigger: orders/price %d: %s", resp2.status_code, resp2.text[:500])
            return None
        order = resp2.json()

        order_id = order.get("id") or order.get("orderId")
        if not order_id:
            logger.warning("jupiter_trigger: order response missing id: %s",
                           list(order.keys()))
            return None

        logger.info(
            "TRIGGER ORDER PLACED: %s | SL=$%.8f | slip=%dbps | expiry=%ds | id=%s",
            ca[:12], sl_price_usd, sl_slippage_bps, expiry_seconds, order_id[:16],
        )
        return {"order_id": order_id}

    except Exception as e:
        logger.warning("jupiter_trigger: place_stop_loss failed for %s: %s", ca[:12], e)
        return None


def update_sl_price(order_id: str, new_sl_price_usd: float) -> bool:
    """PATCH trigger order with new SL price (for trailing stop)."""
    headers = _auth_headers()
    if not headers:
        return False

    try:
        resp = requests.patch(
            f"{TRIGGER_BASE}/orders/price/{order_id}",
            headers=headers,
            json={"triggerPriceUsd": new_sl_price_usd},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("jupiter_trigger: update_sl_price %d for %s: %s",
                           resp.status_code, order_id[:16], resp.text[:300])
            return False
        return True
    except Exception as e:
        logger.warning("jupiter_trigger: update_sl_price failed for %s: %s",
                       order_id[:16], e)
        return False


def cancel_order(order_id: str) -> bool:
    """Cancel a trigger order (2-step: initiate + sign withdrawal)."""
    headers = _auth_headers()
    if not headers:
        return False

    try:
        # Step 1: Initiate cancel
        resp = requests.post(
            f"{TRIGGER_BASE}/orders/price/cancel/{order_id}",
            headers=headers,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        cancel_data = resp.json()

        tx_b64 = cancel_data.get("transaction")
        cancel_request_id = cancel_data.get("requestId")

        if not tx_b64 or not cancel_request_id:
            # Some orders may cancel without needing a withdrawal tx
            logger.info("jupiter_trigger: order %s cancelled (no withdrawal needed)",
                        order_id[:16])
            return True

        # Step 2: Sign withdrawal transaction
        signed_b64 = _sign_transaction(tx_b64)
        if not signed_b64:
            return False

        # Step 3: Confirm cancel
        resp2 = requests.post(
            f"{TRIGGER_BASE}/orders/price/confirm-cancel/{order_id}",
            headers=headers,
            json={
                "cancelRequestId": cancel_request_id,
                "signedTransaction": signed_b64,
            },
            timeout=TIMEOUT,
        )
        resp2.raise_for_status()
        logger.info("jupiter_trigger: order %s cancelled", order_id[:16])
        return True

    except Exception as e:
        logger.warning("jupiter_trigger: cancel_order failed for %s: %s",
                       order_id[:16], e)
        return False


def check_order_status(order_id: str) -> dict | None:
    """
    Check status of a trigger order.
    Returns dict with at least {"state": "active"|"filled"|"cancelled"|"expired"}
    or None on failure.
    """
    headers = _auth_headers()
    if not headers:
        return None

    try:
        # Try direct order lookup first
        resp = requests.get(
            f"{TRIGGER_BASE}/orders/history",
            headers=headers,
            params={"state": "active", "limit": 50},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        orders = data.get("orders", data) if isinstance(data, dict) else data
        if isinstance(orders, list):
            for o in orders:
                oid = o.get("id") or o.get("orderId")
                if oid == order_id:
                    return {"state": "active", "order": o}

        # Not in active — check past (filled/cancelled/expired)
        resp2 = requests.get(
            f"{TRIGGER_BASE}/orders/history",
            headers=headers,
            params={"state": "past", "limit": 20},
            timeout=TIMEOUT,
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        past_orders = data2.get("orders", data2) if isinstance(data2, dict) else data2
        if isinstance(past_orders, list):
            for o in past_orders:
                oid = o.get("id") or o.get("orderId")
                if oid == order_id:
                    # Extract fill info
                    state = o.get("state", "unknown")
                    events = o.get("events", [])
                    fill_event = next((e for e in events if e.get("type") == "fill"), None)
                    return {
                        "state": state,
                        "order": o,
                        "fill_event": fill_event,
                        "output_amount": fill_event.get("outputAmount") if fill_event else None,
                        "output_mint": fill_event.get("outputMint") if fill_event else None,
                    }

        return None

    except Exception as e:
        logger.debug("jupiter_trigger: check_order_status failed for %s: %s",
                     order_id[:16], e)
        return None
