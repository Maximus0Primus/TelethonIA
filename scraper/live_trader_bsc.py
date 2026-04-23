"""v14e — BSC live trading stub.

BSC rollout is conditioned on ETH Phase 2 proving the multi-chain stack works.
No paper data yet. Keep as loud stub until there is a KOL+data plan.
"""
from __future__ import annotations


def execute_buy(ca: str, amount_usd: float, slippage_bps: int = 300) -> dict:
    raise NotImplementedError("BSC live trading not implemented")


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 500) -> dict:
    raise NotImplementedError("BSC live trading not implemented")


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> dict:
    raise NotImplementedError("BSC live trading not implemented")
