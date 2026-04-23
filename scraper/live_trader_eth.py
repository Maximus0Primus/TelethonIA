"""v14e — ETH live trading stub.

Phase 2 (live ETH) is NOT approved yet. Decision gate: 14 days of paper data
with WR >= 65% AND EV net >= +10%/trade (see tasks/todo.md Sprint #ETH-1).
ETA Mai 07 2026.

This stub exists so:
  1. The dispatcher in safe_scraper can `from live_trader_eth import execute_buy`
     without a module-not-found crash if someone flips the config flag early.
  2. The NotImplementedError is loud — if a code path reaches here, something
     upstream failed to chain-gate and we want to know immediately.

When Phase 2 is greenlit:
  - Swap NotImplementedError for a real web3.py + Uniswap V3 SwapRouter02 call.
  - Route ALL transactions through a MEV-protect RPC (Flashbots Protect or
    rpc.mevblocker.io) — ETH memecoin sniping is brutal without it.
  - Use a SEPARATE wallet from Solana (ETH private key env var), not the same
    key. One compromise != two chains drained.
"""
from __future__ import annotations


def execute_buy(ca: str, amount_usd: float, slippage_bps: int = 300) -> dict:
    raise NotImplementedError(
        "ETH live trading not yet approved (Phase 2 gate pending — tasks/todo.md)"
    )


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 500) -> dict:
    raise NotImplementedError(
        "ETH live trading not yet approved (Phase 2 gate pending — tasks/todo.md)"
    )


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> dict:
    raise NotImplementedError(
        "ETH live trading not yet approved (Phase 2 gate pending — tasks/todo.md)"
    )
