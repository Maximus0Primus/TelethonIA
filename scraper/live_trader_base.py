"""v14e — Base live trading stub.

Base rollout mirrors BSC — no paper, no plan, loud stub.
"""
from __future__ import annotations


def execute_buy(ca: str, amount_usd: float, slippage_bps: int = 300) -> dict:
    raise NotImplementedError("Base live trading not implemented")


def execute_sell(ca: str, amount_tokens: int | None = None, slippage_bps: int = 500) -> dict:
    raise NotImplementedError("Base live trading not implemented")


def open_live_trade(client_sb, token_entry: dict, strategy: str,
                    position_usd: float, config: dict) -> dict:
    raise NotImplementedError("Base live trading not implemented")
