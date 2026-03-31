"""
Telegram bot command handler — polls getUpdates for /commands.

Commands:
  /bank    — Bankroll, deployed, available, positions summary
  /pos     — List all open positions with current PnL
  /stats   — 24h and 7d performance summary
  /help    — List available commands

Runs as an async background task in the main event loop.
Polls every 5s, only responds in the configured MONITOR_CHAT_ID.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

_BOT_TOKEN = os.environ.get("MONITOR_BOT_TOKEN")
_CHAT_ID = os.environ.get("MONITOR_CHAT_ID")
_API_BASE = "https://api.telegram.org/bot{token}"

_last_update_id = 0


def _send(text: str) -> bool:
    """Send HTML message to the monitor chat."""
    if not _BOT_TOKEN or not _CHAT_ID:
        return False
    try:
        resp = requests.post(
            f"{_API_BASE.format(token=_BOT_TOKEN)}/sendMessage",
            json={
                "chat_id": _CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning("bot_commands: send failed: %s", e)
        return False


def _get_updates() -> list:
    """Poll for new messages using long polling."""
    global _last_update_id
    if not _BOT_TOKEN:
        return []
    try:
        resp = requests.get(
            f"{_API_BASE.format(token=_BOT_TOKEN)}/getUpdates",
            params={
                "offset": _last_update_id + 1,
                "timeout": 3,
                "allowed_updates": '["message"]',
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("result", [])
        if results:
            _last_update_id = results[-1]["update_id"]
        return results
    except Exception:
        return []


def _handle_bank(sb) -> str:
    """Handle /bank command."""
    from paper_trader import get_open_portfolio
    from safe_scraper import _rt_load_bankroll

    try:
        bankroll = _rt_load_bankroll()
        bal = float(bankroll.get("current_balance", 0))
        start = float(bankroll.get("starting_capital", 500))
        peak = float(bankroll.get("peak_balance", bal))
        total_pnl = float(bankroll.get("total_pnl", 0))
        dd = float(bankroll.get("max_drawdown_pct", 0))
    except Exception:
        bal, start, peak, total_pnl, dd = 0, 500, 0, 0, 0

    portfolio = get_open_portfolio(sb)
    deployed = portfolio["deployed_usd"]
    n_open = portfolio["open_count"]
    available = bal - deployed

    pnl_pct = (bal / start - 1) * 100 if start > 0 else 0
    emoji = "📈" if total_pnl >= 0 else "📉"

    return (
        f"💰 <b>BANKROLL</b>\n\n"
        f"{emoji} Balance: <b>${bal:.2f}</b> ({pnl_pct:+.1f}%)\n"
        f"📊 PnL total: ${total_pnl:+.2f}\n"
        f"🏔 Peak: ${peak:.2f} | DD max: {dd:.1f}%\n"
        f"\n<b>Portfolio:</b>\n"
        f"📦 Déployé: ${deployed:.0f} ({n_open} positions)\n"
        f"💵 Disponible: ${available:.0f}\n"
        f"💼 Capital initial: ${start:.0f}"
    )


def _handle_pos(sb) -> str:
    """Handle /pos command — list open positions."""
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,strategy,position_usd,entry_price,kol_group,cycle_ts")
            .eq("status", "open")
            .eq("is_shadow", False)
            .order("cycle_ts", desc=True)
            .limit(20)
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return "📭 <b>Aucune position ouverte</b>"

    total = sum(float(t.get("position_usd") or 0) for t in trades)
    lines = []
    for t in trades:
        sym = t.get("symbol", "?")
        pos = float(t.get("position_usd") or 0)
        kol = t.get("kol_group", "?")
        ts = t.get("cycle_ts", "")
        # Time since open
        try:
            opened = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_min = (datetime.now(timezone.utc) - opened).total_seconds() / 60
            age_str = f"{age_min:.0f}min" if age_min < 120 else f"{age_min/60:.1f}h"
        except Exception:
            age_str = "?"
        lines.append(f"  • <b>{sym}</b> ${pos:.0f} | {kol} | {age_str}")

    return (
        f"📦 <b>{len(trades)} POSITIONS OUVERTES</b> (${total:.0f} déployé)\n\n"
        + "\n".join(lines)
    )


def _handle_stats(sb) -> str:
    """Handle /stats command — 24h and 7d performance."""
    now = datetime.now(timezone.utc)

    def _period_stats(hours: int) -> dict:
        cutoff = (now - timedelta(hours=hours)).isoformat()
        try:
            result = (
                sb.table("paper_trades")
                .select("pnl_usd,pnl_pct,status")
                .eq("is_shadow", False)
                .neq("status", "open")
                .gte("exit_at", cutoff)
                .execute()
            )
            trades = result.data or []
        except Exception:
            return {"count": 0, "pnl": 0, "wins": 0, "losses": 0}

        pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
        wins = sum(1 for t in trades if (float(t.get("pnl_usd") or 0)) > 0)
        losses = sum(1 for t in trades if (float(t.get("pnl_usd") or 0)) < 0)
        return {"count": len(trades), "pnl": pnl, "wins": wins, "losses": losses}

    d1 = _period_stats(24)
    d7 = _period_stats(168)

    def _fmt(d: dict, label: str) -> str:
        if d["count"] == 0:
            return f"<b>{label}:</b> Aucun trade"
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        emoji = "📈" if d["pnl"] >= 0 else "📉"
        return (
            f"<b>{label}:</b>\n"
            f"  {emoji} PnL: <b>${d['pnl']:+.2f}</b>\n"
            f"  📊 {d['count']} trades | {wr:.0f}% WR ({d['wins']}W / {d['losses']}L)"
        )

    return (
        f"📊 <b>PERFORMANCE</b>\n\n"
        f"{_fmt(d1, '24h')}\n\n"
        f"{_fmt(d7, '7 jours')}"
    )


COMMANDS = {
    "/bank": _handle_bank,
    "/pos": _handle_pos,
    "/stats": _handle_stats,
}

HELP_TEXT = (
    "🤖 <b>Commandes disponibles</b>\n\n"
    "/bank — Bankroll, déployé, disponible\n"
    "/pos — Positions ouvertes\n"
    "/stats — Performance 24h / 7j\n"
    "/help — Cette aide"
)


def process_updates(sb) -> int:
    """Poll and process pending commands. Returns number processed."""
    updates = _get_updates()
    processed = 0

    for update in updates:
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()

        # Only respond in the configured monitor chat
        if chat_id != _CHAT_ID:
            continue

        if not text.startswith("/"):
            continue

        cmd = text.split()[0].split("@")[0].lower()  # handle /bank@botname

        if cmd == "/help":
            _send(HELP_TEXT)
            processed += 1
        elif cmd in COMMANDS:
            try:
                response = COMMANDS[cmd](sb)
                _send(response)
            except Exception as e:
                _send(f"❌ Erreur {cmd}: {e}")
                logger.warning("bot_commands: %s failed: %s", cmd, e)
            processed += 1

    return processed
