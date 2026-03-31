"""
Telegram bot command handler — polls getUpdates for /commands.

Commands (all accept optional args):
  /bank            — Bankroll, deployed, available
  /pos             — Open positions (or last closed if none)
  /trades [N]      — Last N closed trades (default 5)
  /kol [period]    — KOL leaderboard
  /stats [period]  — Performance summary
  /shadow [period] — Top shadow strategies vs main
  /today           — Today's summary
  /config          — Current active config
  /pnl <KOL>       — PnL for a specific KOL
  /best            — Best trade all-time
  /worst           — Worst trade all-time
  /help            — List commands

Periods: 1h 6h 24h 7d 14d 30d all
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

_PERIODS = {
    "1h": 1, "2h": 2, "6h": 6, "12h": 12,
    "24h": 24, "1d": 24, "48h": 48, "2d": 48,
    "7d": 168, "7j": 168, "14d": 336, "30d": 720,
    "all": 0, "tout": 0,
}


# ── Helpers ──

def _parse_period(arg: str) -> tuple[int, str]:
    arg = arg.lower().strip()
    if arg in _PERIODS:
        h = _PERIODS[arg]
        return h, ("All-time" if h == 0 else arg)
    return 0, "All-time"


def _parse_int(arg: str, default: int) -> int:
    try:
        return max(1, min(50, int(arg)))
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


def _query_trades(sb, hours: int = 0, limit: int = 0, strategy: str = ""):
    """Query closed main trades. strategy="" = active strategy only."""
    strat = strategy or _ACTIVE_STRATEGY
    q = (
        sb.table("paper_trades")
        .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,token_address,entry_price,exit_price,high_price_seen,cycle_ts")
        .eq("is_shadow", False)
        .eq("strategy", strat)
        .neq("status", "open")
    )
    if hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = q.gte("exit_at", cutoff)
    q = q.order("exit_at", desc=True)
    if limit > 0:
        q = q.limit(limit)
    return q.execute().data or []


def _compute_stats(trades: list) -> dict:
    """Compute stats from a list of trade rows."""
    if not trades:
        return {"count": 0, "pnl": 0, "wins": 0, "losses": 0, "avg_min": 0}
    pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl_usd") or 0) < 0)
    avg_min = sum(int(t.get("exit_minutes") or 0) for t in trades) / len(trades)
    return {"count": len(trades), "pnl": pnl, "wins": wins, "losses": losses, "avg_min": avg_min}


def _fmt_stats(d: dict, label: str) -> str:
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
    n = _parse_int(args, 5)
    try:
        trades = _query_trades(sb, limit=n)
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
    hours, label = _parse_period(args) if args else (0, "All-time")
    try:
        trades = _query_trades(sb, hours=hours)
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return f"📭 Aucun trade ({label})."

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
            f" | {d['count']}t | {wr:.0f}%"
        )

    total_pnl = sum(d["pnl"] for d in kols.values())
    return (
        f"👥 <b>KOL LEADERBOARD</b> ({label})\n"
        f"{len(kols)} KOLs | {sum(d['count'] for d in kols.values())} trades | ${total_pnl:+.2f}\n\n"
        + "\n".join(lines)
    )


# ── /stats [period] ──

def _handle_stats(sb, args: str) -> str:
    if args:
        hours, label = _parse_period(args)
        trades = _query_trades(sb, hours=hours)
        d = _compute_stats(trades)
        return f"📊 <b>PERFORMANCE {_ACTIVE_STRATEGY}</b>\n\n{_fmt_stats(d, label)}"

    d1 = _compute_stats(_query_trades(sb, hours=24))
    d7 = _compute_stats(_query_trades(sb, hours=168))
    dall = _compute_stats(_query_trades(sb))

    sections = []
    if d1["count"] > 0 and d1["count"] < dall["count"]:
        sections.append(_fmt_stats(d1, "24h"))
    if d7["count"] > 0 and d7["count"] < dall["count"] and d7["count"] != d1["count"]:
        sections.append(_fmt_stats(d7, "7 jours"))
    sections.append(_fmt_stats(dall, "All-time"))

    return f"📊 <b>PERFORMANCE {_ACTIVE_STRATEGY}</b>\n\n" + "\n\n".join(sections)


# ── /shadow [period] ──

def _handle_shadow(sb, args: str) -> str:
    """Compare shadow strategies vs main. Shows top performers."""
    hours, label = _parse_period(args) if args else (0, "All-time")

    # Query all shadow trades for the period
    q = (
        sb.table("paper_trades")
        .select("strategy,pnl_pct,status")
        .eq("is_shadow", True)
        .neq("status", "open")
    )
    if hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = q.gte("exit_at", cutoff)
    try:
        shadows = q.execute().data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not shadows:
        return f"📭 Aucun shadow trade ({label})."

    # Also get main strategy stats for comparison
    main_trades = _query_trades(sb, hours=hours)
    main_stats = _compute_stats(main_trades)

    # Aggregate shadows by strategy
    strats: dict[str, dict] = {}
    for t in shadows:
        s = t.get("strategy", "?")
        if s not in strats:
            strats[s] = {"count": 0, "wins": 0, "pnl_pct_sum": 0}
        strats[s]["count"] += 1
        pnl_pct = float(t.get("pnl_pct") or 0)
        strats[s]["pnl_pct_sum"] += pnl_pct
        if pnl_pct > 0:
            strats[s]["wins"] += 1

    # Sort by avg pnl_pct (since shadows have $0 position, we compare % not $)
    for d in strats.values():
        d["avg_pnl_pct"] = d["pnl_pct_sum"] / d["count"] if d["count"] > 0 else 0
        d["wr"] = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
    sorted_strats = sorted(strats.items(), key=lambda x: -x[1]["avg_pnl_pct"])

    # Main strategy line
    main_wr = main_stats["wins"] / main_stats["count"] * 100 if main_stats["count"] > 0 else 0
    main_avg = sum(float(t.get("pnl_pct") or 0) for t in main_trades) / len(main_trades) * 100 if main_trades else 0

    lines = [f"  ⭐ <b>{_ACTIVE_STRATEGY}</b> avg {main_avg:+.1f}% | {main_stats['count']}t | {main_wr:.0f}% WR"]

    for s_name, d in sorted_strats[:10]:
        better = d["avg_pnl_pct"] * 100 > main_avg
        indicator = "🟢" if better else "🔴"
        lines.append(
            f"  {indicator} <b>{s_name}</b>"
            f" avg {d['avg_pnl_pct']*100:+.1f}%"
            f" | {d['count']}t | {d['wr']:.0f}%"
        )

    return (
        f"🔬 <b>SHADOW COMPARISON</b> ({label})\n"
        f"⭐ = stratégie active, 🟢 = bat le main, 🔴 = pire\n\n"
        + "\n".join(lines)
    )


# ── /today ──

def _handle_today(sb, args: str) -> str:
    """Today's summary: trades, PnL, active KOLs, calls received."""
    from safe_scraper import _rt_load_bankroll

    # Trades closed today
    trades = _query_trades(sb, hours=24)
    stats = _compute_stats(trades)

    # Active KOLs today
    kols_today = set(t.get("kol_group", "") for t in trades if t.get("kol_group"))

    # Bankroll
    try:
        bk = _rt_load_bankroll()
        bal = float(bk.get("current_balance", 0))
        start = float(bk.get("starting_capital", 500))
    except Exception:
        bal, start = 0, 500

    # Open positions
    from paper_trader import get_open_portfolio
    portfolio = get_open_portfolio(sb)

    # Best and worst trade today
    best_t = max(trades, key=lambda t: float(t.get("pnl_usd") or 0)) if trades else None
    worst_t = min(trades, key=lambda t: float(t.get("pnl_usd") or 0)) if trades else None

    pnl_emoji = "📈" if stats["pnl"] >= 0 else "📉"
    bal_pct = (bal / start - 1) * 100 if start > 0 else 0

    text = (
        f"📅 <b>RÉSUMÉ DU JOUR</b>\n\n"
        f"💰 Bankroll: <b>${bal:.2f}</b> ({bal_pct:+.1f}%)\n"
        f"📦 En cours: {portfolio['open_count']} pos (${portfolio['deployed_usd']:.0f})\n"
    )

    if stats["count"] > 0:
        wr = stats["wins"] / stats["count"] * 100
        text += (
            f"\n<b>Trades (24h):</b>\n"
            f"  {pnl_emoji} PnL: <b>${stats['pnl']:+.2f}</b>\n"
            f"  📊 {stats['count']} trades | {wr:.0f}% WR ({stats['wins']}W/{stats['losses']}L)\n"
            f"  👥 {len(kols_today)} KOLs actifs\n"
        )
        if best_t:
            bp = float(best_t.get("pnl_pct") or 0)
            text += f"\n  🏆 Best: <b>{best_t.get('symbol','?')}</b> {bp*100:+.1f}% (${float(best_t.get('pnl_usd') or 0):+.2f}) | {best_t.get('kol_group','?')}"
        if worst_t and float(worst_t.get("pnl_usd") or 0) < 0:
            wp = float(worst_t.get("pnl_pct") or 0)
            text += f"\n  💀 Worst: <b>{worst_t.get('symbol','?')}</b> {wp*100:+.1f}% (${float(worst_t.get('pnl_usd') or 0):+.2f}) | {worst_t.get('kol_group','?')}"
    else:
        text += "\n📭 Aucun trade fermé aujourd'hui"

    return text


# ── /config ──

def _handle_config(sb, args: str) -> str:
    """Show current active configuration."""
    try:
        result = sb.table("scoring_config").select("paper_trade_config,rt_trade_config").eq("id", 1).execute()
        row = result.data[0] if result.data else {}
    except Exception as e:
        return f"❌ Erreur: {e}"

    ptc = row.get("paper_trade_config", {}) or {}
    rtc = row.get("rt_trade_config", {}) or {}

    # Key values
    active = ptc.get("active_strategies", ["?"])
    budget = rtc.get("base_budget_usd", "?")
    max_pos = rtc.get("max_position_usd", "?")
    kelly = rtc.get("sizing", {}).get("kelly_fraction", "?")
    cooldown = rtc.get("cooldown_seconds", 0)
    wl_enabled = rtc.get("whitelist_enabled", False)
    batch = ptc.get("batch_trading_enabled", False) if "batch_trading_enabled" in ptc else "?"
    hybrid = rtc.get("hybrid_strategy", {})
    hybrid_on = hybrid.get("enabled", False)
    hybrid_alloc = hybrid.get("allocations", {})
    dedup = ptc.get("dedup_cooldown_hours", 24)
    ml_mode = ptc.get("ml_gate_mode", "disabled")
    slippage_buy = ptc.get("buy_slippage_bps", 100)
    slippage_sell = ptc.get("sell_slippage_bps", 200)

    # KOL filter
    kol_filter = rtc.get("kol_filter", {})
    kol_filter_on = kol_filter.get("enabled", False)
    kol_min_calls = kol_filter.get("min_calls", 3)
    kol_wr_thresh = kol_filter.get("wr_threshold", 0.4)

    alloc_str = ", ".join(f"{k}={v:.0%}" for k, v in hybrid_alloc.items()) if hybrid_alloc else "?"

    return (
        f"⚙️ <b>CONFIG</b>\n\n"
        f"<b>Stratégie:</b>\n"
        f"  Active: {', '.join(active)}\n"
        f"  Hybrid: {'ON' if hybrid_on else 'OFF'} ({alloc_str})\n"
        f"\n<b>Sizing:</b>\n"
        f"  Budget: ${budget} | Max: ${max_pos}\n"
        f"  Kelly: {kelly} | Dedup: {dedup}h\n"
        f"  Slippage: buy {slippage_buy}bps / sell {slippage_sell}bps\n"
        f"\n<b>Filtres:</b>\n"
        f"  Whitelist: {'ON' if wl_enabled else 'OFF'}\n"
        f"  KOL filter: {'ON' if kol_filter_on else 'OFF'}"
        f" (min {kol_min_calls} calls, {kol_wr_thresh:.0%} WR)\n"
        f"  ML: {ml_mode}\n"
        f"  Batch: {'ON' if batch else 'OFF'}\n"
        f"  Cooldown: {cooldown}s"
    )


# ── /pnl <KOL> ──

def _handle_pnl(sb, args: str) -> str:
    """PnL for a specific KOL. Usage: /pnl FrenzGems"""
    if not args:
        return "Usage: /pnl <nom_du_KOL>\nExemple: /pnl FrenzGems"

    kol_name = args.strip()

    # Query all trades for this KOL (case-insensitive search)
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,exit_at,position_usd,exit_minutes,strategy")
            .eq("is_shadow", False)
            .eq("strategy", _ACTIVE_STRATEGY)
            .ilike("kol_group", f"%{kol_name}%")
            .neq("status", "open")
            .order("exit_at", desc=True)
            .limit(20)
            .execute()
        )
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return f"📭 Aucun trade trouvé pour « {kol_name} »"

    stats = _compute_stats(trades)
    wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0

    lines = []
    for t in trades[:10]:  # Show max 10
        pnl_pct = float(t.get("pnl_pct") or 0)
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        mins = int(t.get("exit_minutes") or 0)
        lines.append(
            f"  {emoji} <b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {mins}min"
        )

    pnl_emoji = "📈" if stats["pnl"] >= 0 else "📉"
    return (
        f"👤 <b>KOL: {kol_name}</b>\n\n"
        f"{pnl_emoji} PnL: <b>${stats['pnl']:+.2f}</b>\n"
        f"📊 {stats['count']} trades | {wr:.0f}% WR ({stats['wins']}W/{stats['losses']}L)\n"
        f"⏱ Durée moy: {stats['avg_min']:.0f}min\n\n"
        + "\n".join(lines)
    )


# ── /best ──

def _handle_best(sb, args: str) -> str:
    """Best trade all-time."""
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,entry_price,exit_price,high_price_seen,token_address")
            .eq("is_shadow", False).eq("strategy", _ACTIVE_STRATEGY)
            .neq("status", "open")
            .order("pnl_usd", desc=True).limit(1).execute()
        )
        t = (result.data or [None])[0]
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not t:
        return "📭 Aucun trade."

    return _format_highlight_trade(t, "🏆 BEST TRADE")


def _handle_worst(sb, args: str) -> str:
    """Worst trade all-time."""
    try:
        result = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,entry_price,exit_price,high_price_seen,token_address")
            .eq("is_shadow", False).eq("strategy", _ACTIVE_STRATEGY)
            .neq("status", "open")
            .order("pnl_usd", desc=False).limit(1).execute()
        )
        t = (result.data or [None])[0]
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not t:
        return "📭 Aucun trade."

    return _format_highlight_trade(t, "💀 WORST TRADE")


def _format_highlight_trade(t: dict, title: str) -> str:
    pnl_pct = float(t.get("pnl_pct") or 0)
    pnl_usd = float(t.get("pnl_usd") or 0)
    pos = float(t.get("position_usd") or 0)
    entry = float(t.get("entry_price") or 0)
    exit_p = float(t.get("exit_price") or 0)
    high = float(t.get("high_price_seen") or 0)
    mins = int(t.get("exit_minutes") or 0)
    max_gain = ((high / entry) - 1) * 100 if entry and high else 0
    ca = t.get("token_address", "")
    link = f'\n🔗 <a href="https://dexscreener.com/solana/{ca}">DexScreener</a>' if ca else ""

    emoji = _exit_emoji(t.get("status", ""), pnl_pct)

    return (
        f"{emoji} <b>{title}</b>\n\n"
        f"<b>{t.get('symbol','?')}</b> | {t.get('status','?')}\n"
        f"👤 KOL: {t.get('kol_group','?')}\n"
        f"📈 PnL: <b>{pnl_pct*100:+.1f}%</b> (${pnl_usd:+.2f})\n"
        f"💵 Position: ${pos:.0f} | ⏱ {mins}min\n"
        f"📊 Entry: ${entry:.8f} → Exit: ${exit_p:.8f}\n"
        f"🔝 Max vu: {max_gain:+.0f}%\n"
        f"📅 {_age_str(t.get('exit_at', ''))}"
        f"{link}"
    )


# ── Command registry ──

COMMANDS = {
    "/bank": _handle_bank,
    "/pos": _handle_pos,
    "/trades": _handle_trades,
    "/kol": _handle_kol,
    "/stats": _handle_stats,
    "/shadow": _handle_shadow,
    "/today": _handle_today,
    "/config": _handle_config,
    "/pnl": _handle_pnl,
    "/best": _handle_best,
    "/worst": _handle_worst,
}

HELP_TEXT = (
    "🤖 <b>Commandes</b>\n\n"
    "<b>Portfolio:</b>\n"
    "  /bank — Bankroll + portfolio\n"
    "  /pos — Positions ouvertes\n"
    "  /today — Résumé du jour\n"
    "\n<b>Trades:</b>\n"
    "  /trades [N] — Derniers N trades (défaut 5)\n"
    "  /best — Meilleur trade\n"
    "  /worst — Pire trade\n"
    "\n<b>Analyse:</b>\n"
    "  /stats [période] — Performance\n"
    "  /kol [période] — Leaderboard KOL\n"
    "  /pnl &lt;KOL&gt; — Stats d'un KOL\n"
    "  /shadow [période] — Shadow vs main\n"
    "\n<b>Système:</b>\n"
    "  /config — Config active\n"
    "  /help — Cette aide\n"
    "\n<b>Périodes:</b> 1h 6h 24h 7d 14d 30d all"
)


def process_updates(sb) -> int:
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
