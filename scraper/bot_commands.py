"""
Telegram bot command handler — polls getUpdates for /commands.

Commands:
  /bank    — Bankroll, deployed, available, positions summary
  /pos     — Open positions (or last closed if none open)
  /trades  — Last 10 closed trades
  /stats   — Performance: 24h, 7d, all-time (DTRAIL10 only)
  /help    — List available commands

Runs as an async background task in the main event loop.
Polls every 5s, only responds in the configured MONITOR_CHAT_ID.
"""

import os
import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

_BOT_TOKEN = os.environ.get("MONITOR_BOT_TOKEN")
_CHAT_ID = os.environ.get("MONITOR_CHAT_ID")
_API_BASE = "https://api.telegram.org/bot{token}"

_last_update_id = 0

# Active strategy — only this counts for stats
_ACTIVE_STRATEGY = "DTRAIL10_ACT15_SL70"


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


def _age_str(ts_str: str) -> str:
    """Convert ISO timestamp to human-readable age."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        mins = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if mins < 60:
            return f"{mins:.0f}min"
        if mins < 1440:
            return f"{mins/60:.1f}h"
        return f"{mins/1440:.0f}j"
    except Exception:
        return "?"


def _exit_emoji(status: str, pnl: float) -> str:
    if status == "trail_stop":
        return "🟢" if pnl > 0.5 else ("✅" if pnl > 0 else "🟡")
    if status == "tp_hit":
        return "🎯"
    if status == "sl_hit":
        return "🔴"
    if status == "timeout":
        return "⏰" if pnl >= 0 else "🟠"
    return "📊"


# ── /bank ──

def _handle_bank(sb) -> str:
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
        f"📦 Déployé: ${deployed:.0f} ({n_open} pos)\n"
        f"💵 Disponible: ${available:.0f}\n"
        f"💼 Capital: ${start:.0f} | Stratégie: {_ACTIVE_STRATEGY}"
    )


# ── /pos ──

def _handle_pos(sb) -> str:
    # Open main positions
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

    if trades:
        total = sum(float(t.get("position_usd") or 0) for t in trades)
        lines = []
        for t in trades:
            age = _age_str(t.get("cycle_ts", ""))
            lines.append(
                f"  • <b>{t.get('symbol', '?')}</b> ${float(t.get('position_usd') or 0):.0f}"
                f" | {t.get('kol_group', '?')} | {age}"
            )
        return (
            f"📦 <b>{len(trades)} POSITIONS OUVERTES</b> (${total:.0f} déployé)\n\n"
            + "\n".join(lines)
        )

    # No open positions — show last closed trade
    try:
        last = (
            sb.table("paper_trades")
            .select("symbol,strategy,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes")
            .eq("is_shadow", False)
            .eq("strategy", _ACTIVE_STRATEGY)
            .neq("status", "open")
            .order("exit_at", desc=True)
            .limit(1)
            .execute()
        )
        last_trade = (last.data or [None])[0]
    except Exception:
        last_trade = None

    if last_trade:
        pnl_pct = float(last_trade.get("pnl_pct") or 0)
        pnl_usd = float(last_trade.get("pnl_usd") or 0)
        emoji = _exit_emoji(last_trade.get("status", ""), pnl_pct)
        age = _age_str(last_trade.get("exit_at", ""))
        return (
            f"📭 <b>Aucune position ouverte</b>\n\n"
            f"Dernier trade (il y a {age}):\n"
            f"  {emoji} <b>{last_trade.get('symbol', '?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {last_trade.get('kol_group', '?')}"
            f" | {last_trade.get('status', '?')}"
        )

    return "📭 <b>Aucune position ouverte</b>\nAucun trade DTRAIL10 enregistré."


# ── /trades ──

def _handle_trades(sb) -> str:
    """Last 10 closed trades (DTRAIL10 only)."""
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes")
            .eq("is_shadow", False)
            .eq("strategy", _ACTIVE_STRATEGY)
            .neq("status", "open")
            .order("exit_at", desc=True)
            .limit(10)
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return "📭 Aucun trade DTRAIL10 fermé."

    total_pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
    lines = []
    for t in trades:
        pnl_pct = float(t.get("pnl_pct") or 0)
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        mins = int(t.get("exit_minutes") or 0)
        age = _age_str(t.get("exit_at", ""))
        lines.append(
            f"  {emoji} <b>{t.get('symbol', '?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {t.get('kol_group', '?')}"
            f" | {mins}min | {age} ago"
        )

    return (
        f"📋 <b>10 DERNIERS TRADES</b> ({_ACTIVE_STRATEGY})\n"
        f"Total: <b>${total_pnl:+.2f}</b>\n\n"
        + "\n".join(lines)
    )


# ── /stats ──

def _handle_stats(sb) -> str:
    """Performance stats: 24h, 7d, all-time — DTRAIL10 only."""
    now = datetime.now(timezone.utc)

    def _query_period(hours: int = 0) -> dict:
        """Query closed DTRAIL10 trades. hours=0 means all-time."""
        try:
            q = (
                sb.table("paper_trades")
                .select("pnl_usd,pnl_pct,status,exit_minutes")
                .eq("is_shadow", False)
                .eq("strategy", _ACTIVE_STRATEGY)
                .neq("status", "open")
            )
            if hours > 0:
                cutoff = (now - timedelta(hours=hours)).isoformat()
                q = q.gte("exit_at", cutoff)
            result = q.execute()
            trades = result.data or []
        except Exception:
            return {"count": 0, "pnl": 0, "wins": 0, "losses": 0, "avg_min": 0}

        if not trades:
            return {"count": 0, "pnl": 0, "wins": 0, "losses": 0, "avg_min": 0}

        pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
        wins = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl_usd") or 0) < 0)
        avg_min = sum(int(t.get("exit_minutes") or 0) for t in trades) / len(trades)
        return {"count": len(trades), "pnl": pnl, "wins": wins, "losses": losses, "avg_min": avg_min}

    d1 = _query_period(24)
    d7 = _query_period(168)
    dall = _query_period(0)

    def _fmt(d: dict, label: str) -> str:
        if d["count"] == 0:
            return f"<b>{label}:</b> Aucun trade"
        wr = d["wins"] / d["count"] * 100
        emoji = "📈" if d["pnl"] >= 0 else "📉"
        return (
            f"<b>{label}:</b>\n"
            f"  {emoji} PnL: <b>${d['pnl']:+.2f}</b>\n"
            f"  📊 {d['count']} trades | {wr:.0f}% WR ({d['wins']}W/{d['losses']}L)\n"
            f"  ⏱ Durée moy: {d['avg_min']:.0f}min"
        )

    return (
        f"📊 <b>PERFORMANCE {_ACTIVE_STRATEGY}</b>\n\n"
        f"{_fmt(d1, '24h')}\n\n"
        f"{_fmt(d7, '7 jours')}\n\n"
        f"{_fmt(dall, 'Depuis le début')}"
    )


# ── Command registry ──

COMMANDS = {
    "/bank": _handle_bank,
    "/pos": _handle_pos,
    "/trades": _handle_trades,
    "/stats": _handle_stats,
}

HELP_TEXT = (
    "🤖 <b>Commandes disponibles</b>\n\n"
    "/bank — Bankroll, déployé, disponible\n"
    "/pos — Positions ouvertes (ou dernier trade)\n"
    "/trades — 10 derniers trades fermés\n"
    "/stats — Performance 24h / 7j / all-time\n"
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
