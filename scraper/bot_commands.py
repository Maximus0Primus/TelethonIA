"""
Telegram bot command handler — polls getUpdates for /commands.

All commands accept optional arguments:
  /bank            — Bankroll, deployed, available
  /pos             — Open positions (or last closed if none)
  /trades [N]      — Last N closed trades (default 5)
  /kol [period]    — KOL leaderboard: /kol, /kol 24h, /kol 7d
  /stats [period]  — Performance: /stats, /stats 24h, /stats 7d
  /help            — List commands

Runs as async background task. Polls every 5s.
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

_ACTIVE_STRATEGY = "DTRAIL10_ACT15_SL70"

# Period aliases
_PERIODS = {
    "1h": 1, "2h": 2, "6h": 6, "12h": 12,
    "24h": 24, "1d": 24, "48h": 48, "2d": 48,
    "7d": 168, "7j": 168, "14d": 336, "30d": 720,
    "all": 0, "tout": 0,
}


def _parse_period(arg: str) -> tuple[int, str]:
    """Parse period argument. Returns (hours, label). 0 = all-time."""
    arg = arg.lower().strip()
    if arg in _PERIODS:
        h = _PERIODS[arg]
        label = "All-time" if h == 0 else arg
        return h, label
    return 0, "All-time"


def _parse_int(arg: str, default: int) -> int:
    """Parse integer argument with bounds."""
    try:
        n = int(arg)
        return max(1, min(50, n))
    except (ValueError, TypeError):
        return default


def _send(text: str) -> bool:
    if not _BOT_TOKEN or not _CHAT_ID:
        return False
    try:
        resp = requests.post(
            f"{_API_BASE.format(token=_BOT_TOKEN)}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning("bot_commands: send failed: %s", e)
        return False


def _get_updates() -> list:
    global _last_update_id
    if not _BOT_TOKEN:
        return []
    try:
        resp = requests.get(
            f"{_API_BASE.format(token=_BOT_TOKEN)}/getUpdates",
            params={"offset": _last_update_id + 1, "timeout": 3,
                    "allowed_updates": '["message"]'},
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


def _query_trades(sb, hours: int = 0, limit: int = 0):
    """Query closed DTRAIL10 main trades. hours=0 = all-time, limit=0 = no limit."""
    q = (
        sb.table("paper_trades")
        .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes")
        .eq("is_shadow", False)
        .eq("strategy", _ACTIVE_STRATEGY)
        .neq("status", "open")
    )
    if hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = q.gte("exit_at", cutoff)
    q = q.order("exit_at", desc=True)
    if limit > 0:
        q = q.limit(limit)
    return q.execute().data or []


# ── /bank ──

def _handle_bank(sb, args: str) -> str:
    from paper_trader import get_open_portfolio
    from safe_scraper import _rt_load_bankroll

    try:
        bk = _rt_load_bankroll()
        bal = float(bk.get("current_balance", 0))
        start = float(bk.get("starting_capital", 500))
        peak = float(bk.get("peak_balance", bal))
        total_pnl = float(bk.get("total_pnl", 0))
        dd = float(bk.get("max_drawdown_pct", 0))
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
        f"\n📦 Déployé: ${deployed:.0f} ({n_open} pos)"
        f" | Dispo: ${available:.0f}\n"
        f"💼 Capital: ${start:.0f} | {_ACTIVE_STRATEGY}"
    )


# ── /pos ──

def _handle_pos(sb, args: str) -> str:
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,strategy,position_usd,entry_price,kol_group,cycle_ts")
            .eq("status", "open").eq("is_shadow", False)
            .order("cycle_ts", desc=True).limit(20).execute()
        )
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if trades:
        total = sum(float(t.get("position_usd") or 0) for t in trades)
        lines = [
            f"  • <b>{t.get('symbol','?')}</b> ${float(t.get('position_usd') or 0):.0f}"
            f" | {t.get('kol_group','?')} | {_age_str(t.get('cycle_ts',''))}"
            for t in trades
        ]
        return f"📦 <b>{len(trades)} POSITIONS OUVERTES</b> (${total:.0f})\n\n" + "\n".join(lines)

    # No open → show last closed
    try:
        last = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at")
            .eq("is_shadow", False).eq("strategy", _ACTIVE_STRATEGY)
            .neq("status", "open").order("exit_at", desc=True).limit(1).execute()
        )
        t = (last.data or [None])[0]
    except Exception:
        t = None

    if t:
        pnl_pct = float(t.get("pnl_pct") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        return (
            f"📭 <b>Aucune position ouverte</b>\n\n"
            f"Dernier trade ({_age_str(t.get('exit_at',''))}):\n"
            f"  {emoji} <b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${float(t.get('pnl_usd') or 0):+.2f})"
            f" | {t.get('kol_group','?')} | {t.get('status','?')}"
        )
    return "📭 <b>Aucune position ouverte</b>"


# ── /trades [N] ──

def _handle_trades(sb, args: str) -> str:
    """Last N closed trades. Usage: /trades, /trades 20"""
    n = _parse_int(args, 5)

    try:
        trades = _query_trades(sb, hours=0, limit=n)
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return "📭 Aucun trade fermé."

    total_pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
    lines = []
    for t in trades:
        pnl_pct = float(t.get("pnl_pct") or 0)
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        mins = int(t.get("exit_minutes") or 0)
        lines.append(
            f"  {emoji} <b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {t.get('kol_group','?')}"
            f" | {mins}min"
        )

    return (
        f"📋 <b>{len(trades)} DERNIERS TRADES</b>\n"
        f"PnL: <b>${total_pnl:+.2f}</b>\n\n"
        + "\n".join(lines)
    )


# ── /kol [period] ──

def _handle_kol(sb, args: str) -> str:
    """KOL leaderboard. Usage: /kol, /kol 24h, /kol 7d"""
    hours, label = _parse_period(args) if args else (0, "All-time")

    try:
        trades = _query_trades(sb, hours=hours)
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return f"📭 Aucun trade ({label})."

    # Aggregate by KOL
    kols: dict[str, dict] = {}
    for t in trades:
        k = t.get("kol_group", "?")
        if k not in kols:
            kols[k] = {"pnl": 0, "count": 0, "wins": 0}
        pnl = float(t.get("pnl_usd") or 0)
        kols[k]["pnl"] += pnl
        kols[k]["count"] += 1
        if pnl > 0:
            kols[k]["wins"] += 1

    # Sort by PnL descending
    sorted_kols = sorted(kols.items(), key=lambda x: -x[1]["pnl"])

    lines = []
    for i, (name, d) in enumerate(sorted_kols):
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        if d["pnl"] > 0:
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "🟢"))
        elif d["pnl"] == 0:
            medal = "⚪"
        else:
            medal = "🔴"
        lines.append(
            f"  {medal} <b>{name}</b>"
            f" ${d['pnl']:+.0f}"
            f" | {d['count']} trades | {wr:.0f}% WR"
        )

    total_pnl = sum(d["pnl"] for d in kols.values())
    total_trades = sum(d["count"] for d in kols.values())

    return (
        f"👥 <b>KOL LEADERBOARD</b> ({label})\n"
        f"{len(kols)} KOLs | {total_trades} trades | ${total_pnl:+.2f}\n\n"
        + "\n".join(lines)
    )


# ── /stats [period] ──

def _handle_stats(sb, args: str) -> str:
    """Performance stats. Usage: /stats, /stats 24h, /stats 7d"""
    now = datetime.now(timezone.utc)

    def _compute(hours: int) -> dict:
        try:
            trades = _query_trades(sb, hours=hours)
        except Exception:
            return {"count": 0, "pnl": 0, "wins": 0, "losses": 0, "avg_min": 0}
        if not trades:
            return {"count": 0, "pnl": 0, "wins": 0, "losses": 0, "avg_min": 0}
        pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
        wins = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl_usd") or 0) < 0)
        avg_min = sum(int(t.get("exit_minutes") or 0) for t in trades) / len(trades)
        return {"count": len(trades), "pnl": pnl, "wins": wins, "losses": losses, "avg_min": avg_min}

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

    # If user specified a period, show only that
    if args:
        hours, label = _parse_period(args)
        d = _compute(hours)
        return f"📊 <b>PERFORMANCE {_ACTIVE_STRATEGY}</b>\n\n{_fmt(d, label)}"

    # Default: show 24h, 7d, all-time — skip redundant
    d1 = _compute(24)
    d7 = _compute(168)
    dall = _compute(0)

    sections = []
    if d1["count"] > 0 and d1["count"] < dall["count"]:
        sections.append(_fmt(d1, "24h"))
    if d7["count"] > 0 and d7["count"] < dall["count"] and d7["count"] != d1["count"]:
        sections.append(_fmt(d7, "7 jours"))
    sections.append(_fmt(dall, "All-time"))

    return f"📊 <b>PERFORMANCE {_ACTIVE_STRATEGY}</b>\n\n" + "\n\n".join(sections)


# ── Command registry ──

COMMANDS = {
    "/bank": _handle_bank,
    "/pos": _handle_pos,
    "/trades": _handle_trades,
    "/kol": _handle_kol,
    "/stats": _handle_stats,
}

HELP_TEXT = (
    "🤖 <b>Commandes</b>\n\n"
    "/bank — Bankroll + portfolio\n"
    "/pos — Positions ouvertes\n"
    "/trades [N] — Derniers N trades (défaut 5)\n"
    "/kol [période] — Leaderboard KOL\n"
    "/stats [période] — Performance\n"
    "/help — Cette aide\n"
    "\n<b>Périodes:</b> 1h 6h 24h 7d 14d 30d all"
)


def process_updates(sb) -> int:
    """Poll and process pending commands. Returns number processed."""
    updates = _get_updates()
    processed = 0

    for update in updates:
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()

        if chat_id != _CHAT_ID:
            continue
        if not text.startswith("/"):
            continue

        parts = text.split(maxsplit=1)
        cmd = parts[0].split("@")[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            _send(HELP_TEXT)
            processed += 1
        elif cmd in COMMANDS:
            try:
                response = COMMANDS[cmd](sb, args)
                _send(response)
            except Exception as e:
                _send(f"❌ Erreur {cmd}: {e}")
                logger.warning("bot_commands: %s failed: %s", cmd, e)
            processed += 1

    return processed
