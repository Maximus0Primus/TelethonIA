"""
Telegram bot command handler — polls getUpdates for /commands.

Paper commands:
  /bank, /pos, /trades, /kol, /stats, /shadow, /today, /config, /pnl, /best, /worst

Live trading commands (v124):
  /live [on|off]   — Status / enable / disable live trading
  /wallet          — SOL balance + token holdings
  /livepos         — Open live positions with current PnL
  /livetrades [N]  — Last N closed live trades
  /livepnl [period]— Live PnL by period
  /setpos <SOL>    — Set max position size
  /setmax <N>      — Set max concurrent positions
  /setlimit <period> <SOL> — Set loss limits
  /setstrat <name> — Set active strategy
  /sell <SYMBOL>   — Sell one position
  /sellall confirm — Emergency sell ALL positions

Periods: 1h 6h 24h 7d 14d 30d all
"""

import os
import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

# v116: Reuse strategy name shortener from alerter
try:
    from alerter import short_strat as _short_strat
except ImportError:
    def _short_strat(name: str) -> str:
        return name.replace("_ACT", "A").replace("_SL", "S")

_BOT_TOKEN = os.environ.get("MONITOR_BOT_TOKEN")
_CHAT_ID = os.environ.get("MONITOR_CHAT_ID")
_API_BASE = "https://api.telegram.org/bot{token}"

_last_update_id = 0

# v116: Active strategies loaded from DB config, cached 5min
_active_strategies_cache: list[str] = []
_active_strategies_ts: float = 0


def _get_active_strategies(sb) -> list[str]:
    """Load active strategies from scoring_config. Cached 5min."""
    global _active_strategies_cache, _active_strategies_ts
    import time
    now = time.time()
    if _active_strategies_cache and now - _active_strategies_ts < 300:
        return _active_strategies_cache
    try:
        result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
        if result.data:
            hybrid = (result.data[0].get("rt_trade_config") or {}).get("hybrid_strategy", {})
            if hybrid.get("enabled"):
                strats = list(hybrid.get("allocations", {}).keys())
                if strats:
                    _active_strategies_cache = strats
                    _active_strategies_ts = now
                    return strats
    except Exception:
        pass
    if not _active_strategies_cache:
        _active_strategies_cache = ["DTRAIL10_ACT15_SL70"]
    return _active_strategies_cache

_PERIODS = {
    "1h": 1, "2h": 2, "6h": 6, "12h": 12,
    "24h": 24, "1d": 24, "48h": 48, "2d": 48,
    "7d": 168, "7j": 168, "14d": 336, "30d": 720,
    "all": 0, "tout": 0,
}

# v14e: chain filter tokens. Accept common aliases (sol/eth/bsc/base).
# `all` / `allchains` disables filter; if the user types nothing, default is
# per-command (usually "all chains"). Every handler that queries paper_trades
# must pass the resolved chain to _query_trades to avoid mixing chains.
_CHAIN_ALIASES: dict[str, str] = {
    "sol": "solana", "solana": "solana", "solona": "solana",
    "eth": "ethereum", "ethereum": "ethereum", "ether": "ethereum",
    "bsc": "bsc", "bnb": "bsc",
    "base": "base",
}


def _chain_tag(chain: str | None) -> str:
    """Compact chain emoji for inline display in lists. Keep 1-char +
    whitespace so lines stay readable on mobile."""
    c = (chain or "solana").lower()
    return {
        "solana":   "🟣",
        "ethereum": "🔷",
        "bsc":      "🟡",
        "base":     "🔵",
    }.get(c, "⚪")


def _explorer_url(chain: str, ca: str) -> str:
    """Canonical block explorer URL per chain for /best and /worst links."""
    c = (chain or "solana").lower()
    if c == "ethereum":
        return f"https://etherscan.io/token/{ca}"
    if c == "bsc":
        return f"https://bscscan.com/token/{ca}"
    if c == "base":
        return f"https://basescan.org/token/{ca}"
    return f"https://solscan.io/token/{ca}"


def _parse_chain_args(args: str) -> tuple[str, str | None]:
    """Pop a chain filter token from args. Returns (remaining_args, chain_or_None).
    Accepts sol/eth/bsc/base and common aliases. 'all'/'allchains' returns
    (remaining, None) explicitly — used by handlers that want to distinguish
    'user asked for all' from 'no filter'."""
    if not args:
        return "", None
    parts = args.split()
    for i, p in enumerate(parts):
        low = p.lower().strip()
        if low in ("all", "allchains", "allchain", "toutes"):
            return " ".join(parts[:i] + parts[i+1:]).strip(), None
        if low in _CHAIN_ALIASES:
            return " ".join(parts[:i] + parts[i+1:]).strip(), _CHAIN_ALIASES[low]
    return args, None


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


def _parse_strategy(arg: str, sb) -> str | None:
    """Fuzzy-match a strategy name from user input.
    Returns canonical strategy name or None if no match.
    Matches case-insensitively and supports partial prefixes like 'dtrail10'."""
    if not arg:
        return None
    from strategies import STRATEGIES
    arg_upper = arg.upper().strip()
    # Exact match
    if arg_upper in STRATEGIES:
        return arg_upper
    # Prefix match (e.g. 'dtrail10' matches 'DTRAIL10_ACT15_SL70')
    # Only match within active strategies first, then all
    active = _get_active_strategies(sb)
    for pool in [active, list(STRATEGIES.keys())]:
        matches = [s for s in pool if s.upper().startswith(arg_upper)]
        if len(matches) == 1:
            return matches[0]
    # Substring match as fallback
    for pool in [active, list(STRATEGIES.keys())]:
        matches = [s for s in pool if arg_upper in s.upper()]
        if len(matches) == 1:
            return matches[0]
    return None


def _split_strategy_args(args: str, sb) -> tuple[str, str | None]:
    """Split args into (remaining_args, strategy_name).
    Tries each word as a potential strategy name.
    Returns (other_args, matched_strategy or None)."""
    if not args:
        return "", None
    parts = args.split()
    # Try last word first (most natural: /trades 10 dtrail10)
    for i in range(len(parts) - 1, -1, -1):
        strat = _parse_strategy(parts[i], sb)
        if strat:
            remaining = " ".join(parts[:i] + parts[i+1:]).strip()
            return remaining, strat
    return args, None


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


def _query_trades(sb, hours: int = 0, limit: int = 0, strategy: str = "",
                  chain: str | None = None):
    """Query closed main trades across all active strategies.
    v116: strategy="" queries all active strategies (not just one).
    v14e: chain filter restricts to paper_trades.chain = X; None = all chains.
    Select list now includes chain so every display can tag it."""
    strategies = [strategy] if strategy else _get_active_strategies(sb)
    q = (
        sb.table("paper_trades")
        .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,token_address,entry_price,exit_price,high_price_seen,cycle_ts,strategy,chain")
        .eq("is_shadow", False)
        .in_("strategy", strategies)
        .neq("status", "open")
    )
    if chain:
        q = q.eq("chain", chain)
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
    """v14e: /bank [chain] — Bankroll groupé par chain.
    Sans argument: montre les 4 chains côte à côte (skip celles vides).
    Avec chain: détail par stratégie de la chain demandée."""
    from paper_trader import get_open_portfolio
    from safe_scraper import _rt_load_bankroll

    _, chain_filter = _parse_chain_args(args)

    try:
        bk = _rt_load_bankroll()
        per_chain = bk.get("strategy_bankrolls_per_chain") or {}
        if not per_chain:
            # v14e legacy fallback: partition the flat dict by strategy prefix.
            from safe_scraper import _rt_strategy_bankrolls_for_chain
            per_chain = {c: _rt_strategy_bankrolls_for_chain(bk, c)
                         for c in ("solana", "ethereum", "bsc", "base")}
    except Exception:
        bk = {}
        per_chain = {}

    portfolio = get_open_portfolio(sb)
    n_open = portfolio["open_count"]
    deployed = portfolio["deployed_usd"]

    sections: list[str] = []
    grand_bal = 0.0
    grand_pnl = 0.0
    for c in ("solana", "ethereum", "bsc", "base"):
        if chain_filter and c != chain_filter:
            continue
        strat_bals = per_chain.get(c) or {}
        if not strat_bals:
            continue
        c_bal = 0.0
        c_pnl = 0.0
        lines = []
        for sname, sdata in sorted(strat_bals.items()):
            bal = float(sdata.get("balance", 500))
            pnl = float(sdata.get("pnl", 0))
            trades = int(sdata.get("trades", 0))
            short = _short_strat(sname)
            emoji = "📈" if pnl >= 0 else "📉"
            lines.append(f"  {emoji} <b>{short}</b>: ${bal:.0f} ({pnl:+.0f}) | {trades}t")
            c_bal += bal
            c_pnl += pnl
        sections.append(
            f"{_chain_tag(c)} <b>{c.upper()}</b> — ${c_bal:.0f} ({c_pnl:+.0f})\n" + "\n".join(lines)
        )
        grand_bal += c_bal
        grand_pnl += c_pnl

    if not sections:
        if chain_filter:
            return f"📭 Aucune stratégie active sur {_chain_tag(chain_filter)} {chain_filter.upper()}"
        grand_bal = float(bk.get("current_balance", 0))
        grand_pnl = float(bk.get("total_pnl", 0))
        sections.append("  Aucune stratégie")

    available = grand_bal - deployed
    header = f"💰 <b>BANKROLL{f' — {_chain_tag(chain_filter)} {chain_filter.upper()}' if chain_filter else ''}</b>"
    return (
        f"{header}\n\n"
        + "\n\n".join(sections)
        + f"\n\n💵 Total: <b>${grand_bal:.0f}</b> ({grand_pnl:+.0f})\n"
        f"📦 Déployé: ${deployed:.0f} ({n_open} pos)"
        f" | Dispo: ${available:.0f}"
    )


# ── /pos ──

def _handle_pos(sb, args: str) -> str:
    """v14e: /pos [chain] — Positions ouvertes groupées par chain puis stratégie."""
    _, chain_filter = _parse_chain_args(args)
    try:
        q = (
            sb.table("paper_trades")
            .select("symbol,strategy,position_usd,entry_price,kol_group,cycle_ts,chain")
            .eq("status", "open").eq("is_shadow", False)
            .order("cycle_ts", desc=True).limit(50)
        )
        if chain_filter:
            q = q.eq("chain", chain_filter)
        result = q.execute()
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if trades:
        # v14e: group by (chain, strategy). Chain first so the reader scans
        # "how exposed am I on each chain" before drilling into strats.
        by_chain: dict[str, dict[str, list]] = {}
        for t in trades:
            c = t.get("chain") or "solana"
            s = t.get("strategy", "?")
            by_chain.setdefault(c, {}).setdefault(s, []).append(t)

        total = sum(float(t.get("position_usd") or 0) for t in trades)
        header = f"📦 <b>{len(trades)} POSITIONS OUVERTES</b> (${total:.0f})"
        if chain_filter:
            header = f"📦 <b>{len(trades)} POS {_chain_tag(chain_filter)} {chain_filter.upper()}</b> (${total:.0f})"
        lines = [header]

        chain_order = ["solana", "ethereum", "bsc", "base"]
        for c in chain_order:
            if c not in by_chain:
                continue
            c_trades = [t for strades in by_chain[c].values() for t in strades]
            c_total = sum(float(t.get("position_usd") or 0) for t in c_trades)
            if not chain_filter:
                lines.append(f"\n{_chain_tag(c)} <b>{c.upper()}</b> — {len(c_trades)} pos (${c_total:.0f})")
            for sname, strades in sorted(by_chain[c].items()):
                short = _short_strat(sname)
                stot = sum(float(t.get("position_usd") or 0) for t in strades)
                lines.append(f"\n  <b>{short}</b> (${stot:.0f}):")
                for t in strades:
                    lines.append(
                        f"    • <b>{t.get('symbol','?')}</b> ${float(t.get('position_usd') or 0):.0f}"
                        f" | {t.get('kol_group','?')} | {_age_str(t.get('cycle_ts',''))}"
                    )
        return "\n".join(lines)

    # No open positions — show last closed (respects chain filter)
    try:
        q = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,chain")
            .eq("is_shadow", False).in_("strategy", _get_active_strategies(sb))
            .neq("status", "open").order("exit_at", desc=True).limit(1)
        )
        if chain_filter:
            q = q.eq("chain", chain_filter)
        last = q.execute()
        t = (last.data or [None])[0]
    except Exception:
        t = None

    if t:
        pnl_pct = float(t.get("pnl_pct") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        c_tag = _chain_tag(t.get("chain"))
        return (
            f"📭 <b>Aucune position ouverte{(' ' + c_tag + ' ' + chain_filter.upper()) if chain_filter else ''}</b>\n\n"
            f"Dernier trade ({_age_str(t.get('exit_at',''))}):\n"
            f"  {emoji} {c_tag} <b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${float(t.get('pnl_usd') or 0):+.2f})"
            f" | {t.get('kol_group','?')} | {t.get('status','?')}"
        )
    return "📭 <b>Aucune position ouverte</b>"


# ── /trades [N] ──

def _handle_trades(sb, args: str) -> str:
    """v14e: /trades [N] [chain] [strat] — Derniers trades avec tag chain sur chaque ligne."""
    args2, chain_filter = _parse_chain_args(args)
    remaining, strat = _split_strategy_args(args2, sb)
    n = _parse_int(remaining, 5) if remaining else 5
    try:
        trades = _query_trades(sb, limit=n, strategy=strat or "", chain=chain_filter)
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        empty_label = ""
        if chain_filter:
            empty_label += f" {_chain_tag(chain_filter)} {chain_filter.upper()}"
        if strat:
            empty_label += f" ({_short_strat(strat)})"
        return f"📭 Aucun trade fermé{empty_label}."

    total_pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
    label_bits = []
    if chain_filter:
        label_bits.append(f"{_chain_tag(chain_filter)} {chain_filter.upper()}")
    if strat:
        label_bits.append(_short_strat(strat))
    label = " — " + " | ".join(label_bits) if label_bits else ""

    lines = []
    for t in trades:
        pnl_pct = float(t.get("pnl_pct") or 0)
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        mins = int(t.get("exit_minutes") or 0)
        strat_tag = "" if strat else f" [{_short_strat(t.get('strategy',''))}]"
        # v14e: always tag chain when the user didn't filter — essential for
        # reading a mixed list. When filtered, skip the tag to save width.
        c_tag = "" if chain_filter else f"{_chain_tag(t.get('chain'))} "
        lines.append(
            f"  {emoji} {c_tag}<b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {t.get('kol_group','?')}"
            f" | {mins}min{strat_tag}"
        )

    return (
        f"📋 <b>{len(trades)} DERNIERS TRADES{label}</b>\n"
        f"PnL: <b>${total_pnl:+.2f}</b>\n\n"
        + "\n".join(lines)
    )


# ── /kol [period] ──

def _handle_kol(sb, args: str) -> str:
    """v14e: /kol [période] [chain] [strat] — KOL leaderboard scoped per chain."""
    args2, chain_filter = _parse_chain_args(args)
    remaining, strat = _split_strategy_args(args2, sb)
    hours, label = _parse_period(remaining) if remaining else (0, "All-time")
    if chain_filter:
        label += f" | {_chain_tag(chain_filter)} {chain_filter.upper()}"
    if strat:
        label += f" | {_short_strat(strat)}"
    try:
        trades = _query_trades(sb, hours=hours, strategy=strat or "", chain=chain_filter)
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
    """v14e: /stats [période] [chain] [strat].
    Sans chain: breakdown 24h/7d/all-time PAR chain si multi-chain actif.
    Avec chain: stats restreintes à cette chain."""
    args2, chain_filter = _parse_chain_args(args)
    remaining, strat = _split_strategy_args(args2, sb)
    s = strat or ""

    label_bits = []
    if chain_filter:
        label_bits.append(f"{_chain_tag(chain_filter)} {chain_filter.upper()}")
    if strat:
        label_bits.append(_short_strat(strat))
    header_suffix = " — " + " | ".join(label_bits) if label_bits else ""

    # User specified a period → single window
    if remaining:
        hours, label = _parse_period(remaining)
        trades = _query_trades(sb, hours=hours, strategy=s, chain=chain_filter)
        # When no chain filter, break down by chain inside the single window.
        if not chain_filter:
            by_chain: dict[str, list] = {}
            for t in trades:
                by_chain.setdefault(t.get("chain") or "solana", []).append(t)
            parts = [_fmt_stats(_compute_stats(trades), label + " — TOTAL")]
            for c in ("solana", "ethereum", "bsc", "base"):
                if c in by_chain:
                    parts.append(_fmt_stats(_compute_stats(by_chain[c]),
                                            f"{_chain_tag(c)} {c.upper()}"))
            return f"📊 <b>PERFORMANCE{header_suffix}</b>\n\n" + "\n\n".join(parts)
        d = _compute_stats(trades)
        return f"📊 <b>PERFORMANCE{header_suffix}</b>\n\n{_fmt_stats(d, label)}"

    d1 = _compute_stats(_query_trades(sb, hours=24, strategy=s, chain=chain_filter))
    d7 = _compute_stats(_query_trades(sb, hours=168, strategy=s, chain=chain_filter))
    dall = _compute_stats(_query_trades(sb, strategy=s, chain=chain_filter))

    sections = []
    if d1["count"] > 0 and d1["count"] < dall["count"]:
        sections.append(_fmt_stats(d1, "24h"))
    if d7["count"] > 0 and d7["count"] < dall["count"] and d7["count"] != d1["count"]:
        sections.append(_fmt_stats(d7, "7 jours"))
    sections.append(_fmt_stats(dall, "All-time"))

    return f"📊 <b>PERFORMANCE{header_suffix}</b>\n\n" + "\n\n".join(sections)


# ── /shadow [period] ──

def _handle_shadow(sb, args: str) -> str:
    """v14e: /shadow [période] [chain] — Shadow vs main, scopé par chain.
    Comparer des shadows ETH à un main Solana = bruit total, donc on force un
    filtre chain (default: solana) quand l'user ne précise pas.
    """
    args2, chain_filter = _parse_chain_args(args)
    # Force solana par défaut pour éviter de mélanger; user peut demander 'all' via syntaxe.
    if chain_filter is None and " all" not in f" {args.lower()} ":
        chain_filter = "solana"
    hours, label = _parse_period(args2) if args2 else (0, "All-time")
    if chain_filter:
        label += f" | {_chain_tag(chain_filter)} {chain_filter.upper()}"

    # Query all shadow trades for the period
    q = (
        sb.table("paper_trades")
        .select("strategy,pnl_pct,status,chain")
        .eq("is_shadow", True)
        .neq("status", "open")
    )
    if chain_filter:
        q = q.eq("chain", chain_filter)
    if hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = q.gte("exit_at", cutoff)
    try:
        shadows = q.execute().data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not shadows:
        return f"📭 Aucun shadow trade ({label})."

    # Also get main strategy stats for comparison — same chain scope.
    main_trades = _query_trades(sb, hours=hours, chain=chain_filter)
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

    # v116: Main strategies line (combined)
    main_wr = main_stats["wins"] / main_stats["count"] * 100 if main_stats["count"] > 0 else 0
    main_avg = sum(float(t.get("pnl_pct") or 0) for t in main_trades) / len(main_trades) * 100 if main_trades else 0
    active = _get_active_strategies(sb)
    active_set = set(active)

    lines = [f"  ⭐ <b>MAIN ({len(active)} strats)</b> avg {main_avg:+.1f}% | {main_stats['count']}t | {main_wr:.0f}% WR"]

    for s_name, d in sorted_strats[:10]:
        if s_name in active_set:
            continue  # Don't show active strategies in shadow list
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
    """v14e: /today [chain] — Résumé du jour avec breakdown par chain."""
    from safe_scraper import _rt_load_bankroll

    _, chain_filter = _parse_chain_args(args)

    # Trades closed today (respect chain filter)
    trades = _query_trades(sb, hours=24, chain=chain_filter)
    stats = _compute_stats(trades)
    kols_today = set(t.get("kol_group", "") for t in trades if t.get("kol_group"))

    # Bankroll per-chain
    try:
        bk = _rt_load_bankroll()
        per_chain = bk.get("strategy_bankrolls_per_chain") or {}
        if not per_chain:
            from safe_scraper import _rt_strategy_bankrolls_for_chain
            per_chain = {c: _rt_strategy_bankrolls_for_chain(bk, c)
                         for c in ("solana", "ethereum", "bsc", "base")}
    except Exception:
        per_chain = {}

    from paper_trader import get_open_portfolio
    portfolio = get_open_portfolio(sb)

    best_t = max(trades, key=lambda t: float(t.get("pnl_usd") or 0)) if trades else None
    worst_t = min(trades, key=lambda t: float(t.get("pnl_usd") or 0)) if trades else None
    pnl_emoji = "📈" if stats["pnl"] >= 0 else "📉"

    # Bankroll rendering: per-chain rollup (total per chain).
    chain_parts = []
    grand_bal = 0.0
    for c in ("solana", "ethereum", "bsc", "base"):
        if chain_filter and c != chain_filter:
            continue
        strat_bals = per_chain.get(c) or {}
        if not strat_bals:
            continue
        c_bal = sum(float(sd.get("balance", 500)) for sd in strat_bals.values())
        c_pnl = sum(float(sd.get("pnl", 0)) for sd in strat_bals.values())
        e = "📈" if c_pnl >= 0 else "📉"
        chain_parts.append(f"  {e} {_chain_tag(c)} <b>{c.upper()}</b>: ${c_bal:.0f} ({c_pnl:+.0f})")
        grand_bal += c_bal

    header_suffix = f" — {_chain_tag(chain_filter)} {chain_filter.upper()}" if chain_filter else ""
    text = f"📅 <b>RÉSUMÉ DU JOUR{header_suffix}</b>\n\n"
    if chain_parts:
        text += "\n".join(chain_parts) + "\n"
    text += f"📦 En cours: {portfolio['open_count']} pos (${portfolio['deployed_usd']:.0f})\n"

    if stats["count"] > 0:
        wr = stats["wins"] / stats["count"] * 100
        text += (
            f"\n<b>Trades (24h):</b>\n"
            f"  {pnl_emoji} PnL: <b>${stats['pnl']:+.2f}</b>\n"
            f"  📊 {stats['count']} trades | {wr:.0f}% WR ({stats['wins']}W/{stats['losses']}L)\n"
            f"  👥 {len(kols_today)} KOLs actifs\n"
        )
        # Per-chain 24h breakdown (only when not filtered — otherwise redundant)
        if not chain_filter:
            by_chain: dict[str, list] = {}
            for t in trades:
                by_chain.setdefault(t.get("chain") or "solana", []).append(t)
            if len(by_chain) > 1:
                text += "\n<b>Par chain:</b>\n"
                for c in ("solana", "ethereum", "bsc", "base"):
                    if c not in by_chain:
                        continue
                    c_trades = by_chain[c]
                    c_pnl = sum(float(t.get("pnl_usd") or 0) for t in c_trades)
                    c_wins = sum(1 for t in c_trades if float(t.get("pnl_usd") or 0) > 0)
                    c_wr = c_wins / len(c_trades) * 100
                    text += f"  {_chain_tag(c)} {c.upper()}: ${c_pnl:+.2f} | {len(c_trades)}t | {c_wr:.0f}% WR\n"

        if best_t:
            bp = float(best_t.get("pnl_pct") or 0)
            text += f"\n  🏆 Best: {_chain_tag(best_t.get('chain'))} <b>{best_t.get('symbol','?')}</b> {bp*100:+.1f}% (${float(best_t.get('pnl_usd') or 0):+.2f}) | {best_t.get('kol_group','?')}"
        if worst_t and float(worst_t.get("pnl_usd") or 0) < 0:
            wp = float(worst_t.get("pnl_pct") or 0)
            text += f"\n  💀 Worst: {_chain_tag(worst_t.get('chain'))} <b>{worst_t.get('symbol','?')}</b> {wp*100:+.1f}% (${float(worst_t.get('pnl_usd') or 0):+.2f}) | {worst_t.get('kol_group','?')}"
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
    max_pos = float(rtc.get("max_position_usd", 120))
    kelly = float(rtc.get("sizing", {}).get("kelly_fraction", 0.127))
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

    # v116: Compute per-strategy position size
    from safe_scraper import _rt_load_bankroll
    try:
        bk = _rt_load_bankroll()
        strat_bals = bk.get("strategy_bankrolls") or {}
    except Exception:
        strat_bals = {}

    alloc_str = ", ".join(f"{k}={v:.0%}" for k, v in hybrid_alloc.items()) if hybrid_alloc else "?"

    # KOL access
    if wl_enabled:
        kol_filter = rtc.get("kol_filter", {})
        kol_text = (
            f"  KOLs: WHITELIST (min {kol_filter.get('min_calls', 3)} calls,"
            f" {float(kol_filter.get('wr_threshold', 0.4)):.0%} WR,"
            f" {kol_filter.get('lookback_days', 7)}j lookback)"
        )
    else:
        kol_text = "  KOLs: TOUS (pas de whitelist)"

    # Per-strategy sizing lines
    sizing_lines = []
    for sname in active:
        s_bal = float((strat_bals.get(sname) or {}).get("balance", 500))
        s_raw = s_bal * kelly
        s_pos = min(s_raw, max_pos)
        short = _short_strat(sname)
        cap_tag = " ⚠️cap" if s_raw > max_pos else ""
        sizing_lines.append(f"  {short}: <b>${s_pos:.0f}</b> (${s_bal:.0f} × {kelly:.1%}){cap_tag}")

    return (
        f"⚙️ <b>CONFIG</b>\n\n"
        f"<b>Stratégies ({len(active)}):</b>\n"
        f"  {', '.join(_short_strat(s) for s in active)}\n"
        f"  Hybrid: {'ON' if hybrid_on else 'OFF'} ({alloc_str})\n"
        f"\n<b>Sizing (Kelly {kelly:.1%}, cap ${max_pos:.0f}):</b>\n"
        + "\n".join(sizing_lines) + "\n"
        f"  Dedup: {dedup}h | Slippage: {slippage_buy}/{slippage_sell}bps\n"
        f"\n<b>Filtres:</b>\n"
        f"{kol_text}\n"
        f"  ML: {ml_mode}\n"
        f"  Batch: {'ON' if batch else 'OFF'}\n"
        f"  Cooldown: {cooldown}s"
    )


# ── /pnl <KOL> ──

def _handle_pnl(sb, args: str) -> str:
    """v14e: /pnl <KOL> [chain] [strat] — KOL stats, chain-scoped if passed."""
    if not args:
        return "Usage: /pnl <nom_du_KOL> [sol|eth|bsc|base] [stratégie]\nExemple: /pnl FrenzGems eth"

    args2, chain_filter = _parse_chain_args(args)
    remaining, strat = _split_strategy_args(args2, sb)
    kol_name = remaining.strip() if remaining else args2.strip()
    if not kol_name:
        return "Usage: /pnl <nom_du_KOL> [sol|eth|bsc|base] [stratégie]"

    strategies = [strat] if strat else _get_active_strategies(sb)
    label_bits = []
    if chain_filter:
        label_bits.append(f"{_chain_tag(chain_filter)} {chain_filter.upper()}")
    if strat:
        label_bits.append(_short_strat(strat))
    strat_label = f" ({' | '.join(label_bits)})" if label_bits else ""
    try:
        q = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,exit_at,position_usd,exit_minutes,strategy,chain")
            .eq("is_shadow", False)
            .in_("strategy", strategies)
            .ilike("kol_group", f"%{kol_name}%")
            .neq("status", "open")
            .order("exit_at", desc=True)
            .limit(20)
        )
        if chain_filter:
            q = q.eq("chain", chain_filter)
        result = q.execute()
        trades = result.data or []
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not trades:
        return f"📭 Aucun trade trouvé pour « {kol_name} »{strat_label}"

    stats = _compute_stats(trades)
    wr = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0

    lines = []
    for t in trades[:10]:
        pnl_pct = float(t.get("pnl_pct") or 0)
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl_pct)
        mins = int(t.get("exit_minutes") or 0)
        c_tag = "" if chain_filter else f"{_chain_tag(t.get('chain'))} "
        lines.append(
            f"  {emoji} {c_tag}<b>{t.get('symbol','?')}</b>"
            f" {pnl_pct*100:+.1f}% (${pnl_usd:+.2f})"
            f" | {mins}min"
        )

    pnl_emoji = "📈" if stats["pnl"] >= 0 else "📉"
    return (
        f"👤 <b>KOL: {kol_name}{strat_label}</b>\n\n"
        f"{pnl_emoji} PnL: <b>${stats['pnl']:+.2f}</b>\n"
        f"📊 {stats['count']} trades | {wr:.0f}% WR ({stats['wins']}W/{stats['losses']}L)\n"
        f"⏱ Durée moy: {stats['avg_min']:.0f}min\n\n"
        + "\n".join(lines)
    )


# ── /best ──

def _handle_best(sb, args: str) -> str:
    """v14e: /best [chain] [strat] — Meilleur trade all-time."""
    args2, chain_filter = _parse_chain_args(args)
    strat = _parse_strategy(args2.strip(), sb) if args2.strip() else None
    strategies = [strat] if strat else _get_active_strategies(sb)
    label_bits = []
    if chain_filter:
        label_bits.append(f"{_chain_tag(chain_filter)} {chain_filter.upper()}")
    if strat:
        label_bits.append(_short_strat(strat))
    strat_label = f" ({' | '.join(label_bits)})" if label_bits else ""
    try:
        q = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,entry_price,exit_price,high_price_seen,token_address,strategy,chain")
            .eq("is_shadow", False).in_("strategy", strategies)
            .neq("status", "open")
            .order("pnl_usd", desc=True).limit(1)
        )
        if chain_filter:
            q = q.eq("chain", chain_filter)
        result = q.execute()
        t = (result.data or [None])[0]
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not t:
        return f"📭 Aucun trade{strat_label}."

    return _format_highlight_trade(t, f"🏆 BEST TRADE{strat_label}")


def _handle_worst(sb, args: str) -> str:
    """v14e: /worst [chain] [strat] — Pire trade all-time."""
    args2, chain_filter = _parse_chain_args(args)
    strat = _parse_strategy(args2.strip(), sb) if args2.strip() else None
    strategies = [strat] if strat else _get_active_strategies(sb)
    label_bits = []
    if chain_filter:
        label_bits.append(f"{_chain_tag(chain_filter)} {chain_filter.upper()}")
    if strat:
        label_bits.append(_short_strat(strat))
    strat_label = f" ({' | '.join(label_bits)})" if label_bits else ""
    try:
        q = (
            sb.table("paper_trades")
            .select("symbol,pnl_pct,pnl_usd,status,kol_group,exit_at,position_usd,exit_minutes,entry_price,exit_price,high_price_seen,token_address,strategy,chain")
            .eq("is_shadow", False).in_("strategy", strategies)
            .neq("status", "open")
            .order("pnl_usd", desc=False).limit(1)
        )
        if chain_filter:
            q = q.eq("chain", chain_filter)
        result = q.execute()
        t = (result.data or [None])[0]
    except Exception as e:
        return f"❌ Erreur: {e}"

    if not t:
        return f"📭 Aucun trade{strat_label}."

    return _format_highlight_trade(t, f"💀 WORST TRADE{strat_label}")


def _format_highlight_trade(t: dict, title: str) -> str:
    """v14e: DexScreener URL + block explorer chosen per chain (Solscan for SOL,
    Etherscan/BscScan/BaseScan for EVM)."""
    pnl_pct = float(t.get("pnl_pct") or 0)
    pnl_usd = float(t.get("pnl_usd") or 0)
    pos = float(t.get("position_usd") or 0)
    entry = float(t.get("entry_price") or 0)
    exit_p = float(t.get("exit_price") or 0)
    high = float(t.get("high_price_seen") or 0)
    mins = int(t.get("exit_minutes") or 0)
    max_gain = ((high / entry) - 1) * 100 if entry and high else 0
    ca = t.get("token_address", "")
    chain = (t.get("chain") or "solana").lower()
    c_tag = _chain_tag(chain)

    link = ""
    if ca:
        ds = f'<a href="https://dexscreener.com/{chain}/{ca}">DexScreener</a>'
        exp_url = _explorer_url(chain, ca)
        exp_name = {"solana": "Solscan", "ethereum": "Etherscan",
                    "bsc": "BscScan", "base": "BaseScan"}.get(chain, "Explorer")
        link = f'\n🔗 {ds} | <a href="{exp_url}">{exp_name}</a>'

    emoji = _exit_emoji(t.get("status", ""), pnl_pct)

    return (
        f"{emoji} <b>{title}</b>\n\n"
        f"{c_tag} <b>{t.get('symbol','?')}</b> | {t.get('status','?')}\n"
        f"👤 KOL: {t.get('kol_group','?')}\n"
        f"📈 PnL: <b>{pnl_pct*100:+.1f}%</b> (${pnl_usd:+.2f})\n"
        f"💵 Position: ${pos:.0f} | ⏱ {mins}min\n"
        f"📊 Entry: ${entry:.8f} → Exit: ${exit_p:.8f}\n"
        f"🔝 Max vu: {max_gain:+.0f}%\n"
        f"📅 {_age_str(t.get('exit_at', ''))}"
        f"{link}"
    )


# ── Live trading commands ──

def _handle_live(sb, args: str) -> str:
    """Live trading status, enable/disable."""
    args_lower = args.strip().lower()

    # /live on — enable live trading
    if args_lower in ("on", "enable", "start"):
        try:
            result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
            cfg = result.data[0]["rt_trade_config"] if result.data else {}
            cfg["enabled"] = True
            sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
            return "✅ <b>Live trading ACTIVÉ</b>\nLes prochains signaux ouvriront des trades on-chain."
        except Exception as e:
            return f"❌ Erreur activation live: {e}"

    # /live off — disable live trading
    if args_lower in ("off", "disable", "stop"):
        try:
            result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
            cfg = result.data[0]["rt_trade_config"] if result.data else {}
            cfg["enabled"] = False
            sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
            return "⛔ <b>Live trading DÉSACTIVÉ</b>\nLes positions ouvertes sont toujours monitorées."
        except Exception as e:
            return f"❌ Erreur désactivation live: {e}"

    # /live — status
    try:
        from live_trader import get_wallet_balance
        from safe_scraper import _rt_load_config, _rt_load_bankroll

        config = _rt_load_config()
        live_cfg = config.get("live_trading", {})
        enabled = live_cfg.get("enabled", False)
        max_pos_sol = live_cfg.get("max_position_sol", 0.5)
        max_open = live_cfg.get("max_open_positions", 5)
        daily_limit = live_cfg.get("daily_loss_limit_sol", 2.0)
        weekly_limit = live_cfg.get("weekly_loss_limit_sol", 5.0)
        strategy = live_cfg.get("rt_strategies", "all")

        # Wallet balance
        wallet = get_wallet_balance()
        sol_bal = wallet["sol_balance"] if wallet else 0
        n_tokens = len(wallet.get("token_balances", {})) if wallet else 0

        # Open live trades
        open_trades = sb.table("paper_trades").select(
            "symbol,entry_price,position_usd,position_sol,created_at,strategy"
        ).eq("status", "open").eq("source", "rt_live").execute().data or []

        # Today's closed live trades
        today_cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
        closed_today = sb.table("paper_trades").select(
            "pnl_usd,pnl_pct,status"
        ).eq("source", "rt_live").neq("status", "open").gte("exit_at", today_cutoff).execute().data or []

        today_pnl = sum(float(t.get("pnl_usd") or 0) for t in closed_today)
        today_n = len(closed_today)
        today_wins = sum(1 for t in closed_today if float(t.get("pnl_usd") or 0) > 0)

        # Bankroll
        bk = _rt_load_bankroll()
        strat_bk = bk.get("strategy_bankrolls", {})

        status_emoji = "🟢" if enabled else "🔴"
        lines = [
            f"{status_emoji} <b>LIVE TRADING {'ACTIF' if enabled else 'INACTIF'}</b> {_chain_tag('solana')} Solana-only\n",
            f"<i>ETH/BSC/Base restent en paper tant que leurs live adapters ne sont pas livrés.</i>\n",
            f"<b>Wallet:</b>",
            f"  💰 {sol_bal:.4f} SOL",
            f"  🪙 {n_tokens} token(s) en portefeuille",
            f"\n<b>Positions ouvertes:</b> {len(open_trades)}/{max_open}",
        ]
        for t in open_trades:
            lines.append(f"  • {t['symbol']} — {float(t.get('position_sol') or 0):.3f} SOL (${float(t.get('position_usd') or 0):.1f})")

        lines.append(f"\n<b>Aujourd'hui:</b>")
        if today_n > 0:
            lines.append(f"  {'📈' if today_pnl >= 0 else '📉'} PnL: <b>${today_pnl:+.2f}</b> ({today_n} trades, {today_wins}W)")
        else:
            lines.append(f"  Aucun trade live aujourd'hui")

        lines.append(f"\n<b>Config:</b>")
        lines.append(f"  📏 Max position: {max_pos_sol} SOL")
        lines.append(f"  📊 Max open: {max_open}")
        lines.append(f"  🛡 Loss limits: {daily_limit} SOL/jour, {weekly_limit} SOL/sem")
        lines.append(f"  🎯 Stratégie: {strategy}")

        if strat_bk:
            lines.append(f"\n<b>Bankroll par stratégie:</b>")
            for s, b in sorted(strat_bk.items(), key=lambda x: -x[1].get("balance", 0)):
                bal = b.get("balance", 0)
                peak = b.get("peak", 0)
                dd = (1 - bal / peak) * 100 if peak > 0 else 0
                lines.append(f"  {_short_strat(s)}: ${bal:.0f} (DD {dd:.0f}%)")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Erreur live status: {e}"


def _handle_wallet(sb, args: str) -> str:
    """Wallet SOL balance + token holdings."""
    try:
        from live_trader import get_wallet_balance
        wallet = get_wallet_balance()
        if not wallet:
            return "❌ Impossible de lire le wallet"

        sol = wallet["sol_balance"]
        tokens = wallet.get("token_balances", {})

        lines = [f"💰 <b>Wallet</b>\n", f"  SOL: <b>{sol:.4f}</b>"]

        if tokens:
            lines.append(f"\n  <b>Tokens ({len(tokens)}):</b>")
            for mint, info in tokens.items():
                ui_amount = info.get("ui_amount", 0)
                if ui_amount > 0:
                    lines.append(f"  • {mint[:8]}... : {ui_amount:,.0f}")
        else:
            lines.append(f"\n  Aucun token")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Erreur wallet: {e}"


def _handle_livepos(sb, args: str) -> str:
    """Open live positions with current PnL."""
    try:
        from paper_trader import _fetch_prices_batch
        trades = sb.table("paper_trades").select(
            "id,symbol,token_address,entry_price,position_usd,position_sol,"
            "created_at,strategy,high_price_seen,chain"
        ).eq("status", "open").eq("source", "rt_live").execute().data or []

        if not trades:
            return "📭 Aucune position live ouverte"

        # v14e: route DS batch per chain.
        addrs = list({t["token_address"] for t in trades})
        _chain_map = {t["token_address"]: (t.get("chain") or "solana") for t in trades}
        prices = _fetch_prices_batch(addrs, chain_by_addr=_chain_map)

        lines = [f"📊 <b>{len(trades)} position(s) live</b>\n"]
        total_pnl = 0
        for t in trades:
            addr = t["token_address"]
            ep = float(t["entry_price"])
            cp = prices.get(addr)
            pos_usd = float(t.get("position_usd") or 0)
            age = _age_str(t["created_at"])
            high = float(t.get("high_price_seen") or 0)

            if cp and ep > 0:
                pnl_pct = (cp / ep - 1) * 100
                pnl_usd = pos_usd * (cp / ep - 1)
                peak_pct = (high / ep - 1) * 100 if high > 0 else 0
                total_pnl += pnl_usd
                emoji = "🟢" if pnl_pct > 0 else "🔴"
                lines.append(
                    f"  {emoji} <b>{t['symbol']}</b> ({_short_strat(t['strategy'])})\n"
                    f"    PnL: <b>{pnl_pct:+.1f}%</b> (${pnl_usd:+.2f}) | Peak: +{peak_pct:.0f}%\n"
                    f"    Pos: {float(t.get('position_sol') or 0):.3f} SOL | Age: {age}"
                )
            else:
                lines.append(f"  ⚠️ <b>{t['symbol']}</b> — pas de prix")

        lines.append(f"\n💰 PnL total unrealized: <b>${total_pnl:+.2f}</b>")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Erreur livepos: {e}"


def _handle_livetrades(sb, args: str) -> str:
    """Last N closed live trades. Optional: /livetrades 10 dtrail10"""
    remaining, strat = _split_strategy_args(args, sb)
    n = _parse_int(remaining, 10) if remaining else 10
    q = (
        sb.table("paper_trades")
        .select("symbol,strategy,pnl_pct,pnl_usd,status,exit_minutes,created_at,exit_at,position_sol")
        .eq("source", "rt_live")
        .neq("status", "open")
    )
    if strat:
        q = q.eq("strategy", strat)
    trades = q.order("exit_at", desc=True).limit(n).execute().data or []

    strat_label = f" — {_short_strat(strat)}" if strat else ""
    if not trades:
        return f"📭 Aucun trade live fermé{strat_label}"

    total = sum(float(t.get("pnl_usd") or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
    lines = [f"📋 <b>Derniers {len(trades)} trades live{strat_label}</b> (PnL: ${total:+.2f}, {wins}W/{len(trades)-wins}L)\n"]
    for t in trades:
        pnl = float(t.get("pnl_pct") or 0) * 100
        pnl_usd = float(t.get("pnl_usd") or 0)
        emoji = _exit_emoji(t.get("status", ""), pnl)
        age = _age_str(t.get("exit_at") or t["created_at"])
        strat_tag = "" if strat else f" {_short_strat(t.get('strategy',''))}"
        lines.append(
            f"  {emoji} {t['symbol']}{strat_tag} "
            f"<b>{pnl:+.1f}%</b> (${pnl_usd:+.2f}) {int(t.get('exit_minutes') or 0)}min — {age}"
        )
    return "\n".join(lines)


def _handle_livepnl(sb, args: str) -> str:
    """Live PnL by period. Optional: /livepnl 7d dtrail10"""
    remaining, strat = _split_strategy_args(args, sb)
    hours, label = _parse_period(remaining or "24h")
    if strat:
        label += f" | {_short_strat(strat)}"
    q = (
        sb.table("paper_trades")
        .select("pnl_pct,pnl_usd,status,strategy,created_at")
        .eq("source", "rt_live")
        .neq("status", "open")
    )
    if strat:
        q = q.eq("strategy", strat)
    if hours > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = q.gte("exit_at", cutoff)
    trades = q.order("exit_at", desc=True).execute().data or []

    if not trades:
        return f"📭 Aucun trade live ({label})"

    total = sum(float(t.get("pnl_usd") or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl_usd") or 0) > 0)
    wr = wins / len(trades) * 100 if trades else 0

    emoji = "📈" if total >= 0 else "📉"
    lines = [
        f"{emoji} <b>Live PnL — {label}</b>\n",
        f"  💰 Total: <b>${total:+.2f}</b>",
        f"  📊 {len(trades)} trades | {wr:.0f}% WR ({wins}W/{len(trades)-wins}L)",
    ]

    # Per strategy breakdown (only if not filtered to a single strategy)
    if not strat:
        by_strat = {}
        for t in trades:
            s = t.get("strategy", "?")
            by_strat.setdefault(s, []).append(float(t.get("pnl_usd") or 0))
        if by_strat:
            lines.append(f"\n  <b>Par stratégie:</b>")
            for s, pnls in sorted(by_strat.items(), key=lambda x: -sum(x[1])):
                lines.append(f"    {_short_strat(s)}: ${sum(pnls):+.2f} ({len(pnls)} trades)")

    return "\n".join(lines)


def _handle_setpos(sb, args: str) -> str:
    """Set max position size in SOL."""
    try:
        val = float(args.strip())
    except (ValueError, TypeError):
        return "❌ Usage: /setpos &lt;SOL&gt;\nEx: /setpos 0.2"
    if val <= 0 or val > 10:
        return "❌ Position doit être entre 0.01 et 10 SOL"

    try:
        result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
        cfg = result.data[0]["rt_trade_config"] if result.data else {}
        old = cfg.get("max_position_sol", "?")
        cfg["max_position_sol"] = val
        sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
        return f"✅ Position max: <b>{old} → {val} SOL</b>"
    except Exception as e:
        return f"❌ Erreur: {e}"


def _handle_setmax(sb, args: str) -> str:
    """Set max open positions."""
    try:
        val = int(args.strip())
    except (ValueError, TypeError):
        return "❌ Usage: /setmax &lt;N&gt;\nEx: /setmax 3"
    if val < 1 or val > 20:
        return "❌ Max positions entre 1 et 20"

    try:
        result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
        cfg = result.data[0]["rt_trade_config"] if result.data else {}
        old = cfg.get("max_open_positions", "?")
        cfg["max_open_positions"] = val
        sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
        return f"✅ Max positions: <b>{old} → {val}</b>"
    except Exception as e:
        return f"❌ Erreur: {e}"


def _handle_setlimit(sb, args: str) -> str:
    """Set loss limits (daily/weekly/monthly)."""
    parts = args.strip().split()
    if len(parts) != 2:
        return "❌ Usage: /setlimit &lt;daily|weekly|monthly&gt; &lt;SOL&gt;\nEx: /setlimit daily 1.5"
    period = parts[0].lower()
    try:
        val = float(parts[1])
    except (ValueError, TypeError):
        return "❌ Valeur SOL invalide"
    if val <= 0 or val > 100:
        return "❌ Limite entre 0.01 et 100 SOL"

    key_map = {"daily": "daily_loss_limit_sol", "weekly": "weekly_loss_limit_sol", "monthly": "monthly_loss_limit_sol"}
    if period not in key_map:
        return "❌ Période: daily, weekly, ou monthly"

    try:
        result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
        cfg = result.data[0]["rt_trade_config"] if result.data else {}
        old = cfg.get(key_map[period], "?")
        cfg[key_map[period]] = val
        sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
        return f"✅ Limite {period}: <b>{old} → {val} SOL</b>"
    except Exception as e:
        return f"❌ Erreur: {e}"


def _handle_setstrat(sb, args: str) -> str:
    """Set active live strategy."""
    strat = args.strip()
    if not strat:
        return "❌ Usage: /setstrat &lt;strategy&gt;\nEx: /setstrat DTRAIL5_ACT10_SL40\nOu: /setstrat all"

    try:
        result = sb.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
        cfg = result.data[0]["rt_trade_config"] if result.data else {}
        old = cfg.get("rt_strategies", "all")
        cfg["rt_strategies"] = strat
        sb.table("scoring_config").update({"rt_trade_config": cfg}).eq("id", 1).execute()
        return f"✅ Stratégie live: <b>{old} → {strat}</b>"
    except Exception as e:
        return f"❌ Erreur: {e}"


def _handle_sell(sb, args: str) -> str:
    """v14e: Solana-only — Jupiter Ultra. ETH/BSC/Base rejetés explicitement."""
    symbol = args.strip().upper()
    if not symbol:
        return "❌ Usage: /sell &lt;SYMBOL&gt;\nEx: /sell $MOJO"
    if not symbol.startswith("$"):
        symbol = "$" + symbol

    try:
        trades = sb.table("paper_trades").select(
            "id,symbol,token_address,position_sol,entry_price,chain"
        ).eq("status", "open").eq("source", "rt_live").eq("symbol", symbol).execute().data or []

        if not trades:
            return f"❌ Aucune position live ouverte pour {symbol}"

        trade = trades[0]
        ca = trade["token_address"]
        t_chain = trade.get("chain") or "solana"
        if t_chain != "solana":
            return (
                f"⛔ <b>{symbol}</b> est sur {_chain_tag(t_chain)} {t_chain.upper()}.\n"
                f"Le live trading n'est disponible que sur Solana (Jupiter Ultra).\n"
                f"L'adapter live pour {t_chain.upper()} n'est pas encore livré."
            )

        from live_trader import execute_sell
        result = execute_sell(ca)

        if result.get("success"):
            sig = result.get("signature", "")[:16]
            return f"✅ <b>SELL {symbol}</b> exécuté\nSignature: {sig}..."
        else:
            return f"❌ Sell {symbol} échoué: {result.get('error', 'unknown')}"
    except Exception as e:
        return f"❌ Erreur sell: {e}"


def _handle_sellall(sb, args: str) -> str:
    """Panic sell ALL live positions."""
    # Safety: require confirmation word
    if args.strip().lower() != "confirm":
        return "⚠️ <b>PANIC SELL</b>\nVend TOUTES les positions live immédiatement.\n\nPour confirmer: /sellall confirm"

    try:
        trades = sb.table("paper_trades").select(
            "id,symbol,token_address,position_sol,chain"
        ).eq("status", "open").eq("source", "rt_live").execute().data or []

        if not trades:
            return "📭 Aucune position live ouverte"

        from live_trader import execute_sell
        results = []
        skipped = 0
        for t in trades:
            # v14e: execute_sell has a Solana-only guard; skip non-Solana loudly
            # rather than get 'non-solana-mint' errors silently in the batch.
            if (t.get("chain") or "solana") != "solana":
                results.append(f"  ⚠️ {t['symbol']} — {_chain_tag(t.get('chain'))} skip (pas de live non-Solana)")
                skipped += 1
                continue
            try:
                r = execute_sell(t["token_address"])
                status = "✅" if r.get("success") else "❌"
                results.append(f"  {status} {t['symbol']}")
            except Exception as e:
                results.append(f"  ❌ {t['symbol']}: {e}")

        header = f"🚨 <b>SELLALL — {len(trades)} positions</b>"
        if skipped:
            header += f" <i>({skipped} non-Solana skipped)</i>"
        return header + "\n" + "\n".join(results)
    except Exception as e:
        return f"❌ Erreur sellall: {e}"


# ── Command registry ──

COMMANDS = {
    # Paper
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
    # Live
    "/live": _handle_live,
    "/wallet": _handle_wallet,
    "/livepos": _handle_livepos,
    "/livetrades": _handle_livetrades,
    "/livepnl": _handle_livepnl,
    "/setpos": _handle_setpos,
    "/setmax": _handle_setmax,
    "/setlimit": _handle_setlimit,
    "/setstrat": _handle_setstrat,
    "/sell": _handle_sell,
    "/sellall": _handle_sellall,
}

HELP_TEXT = (
    "🤖 <b>Commandes</b>\n\n"
    "<b>Portfolio:</b>\n"
    "  /bank [chain] — Bankroll per-chain\n"
    "  /pos [chain] — Positions ouvertes (paper)\n"
    "  /today [chain] — Résumé du jour\n"
    "\n<b>Trades:</b>\n"
    "  /trades [N] [chain] [strat] — Derniers trades\n"
    "  /best [chain] [strat] — Meilleur trade\n"
    "  /worst [chain] [strat] — Pire trade\n"
    "\n<b>Analyse:</b>\n"
    "  /stats [période] [chain] [strat] — Performance\n"
    "  /kol [période] [chain] [strat] — Leaderboard KOL\n"
    "  /pnl &lt;KOL&gt; [chain] [strat] — Stats d'un KOL\n"
    "  /shadow [période] [chain] — Shadow vs main (sol par défaut)\n"
    "\n<b>💎 Live Trading (Solana-only):</b>\n"
    "  /live — Status live (wallet, positions, PnL)\n"
    "  /live on|off — Activer/désactiver le live\n"
    "  /wallet — Balance SOL du wallet\n"
    "  /livepos — Positions live ouvertes + PnL\n"
    "  /livetrades [N] [strat] — Trades live\n"
    "  /livepnl [période] [strat] — PnL live\n"
    "\n<b>⚙️ Config Live:</b>\n"
    "  /setpos &lt;SOL&gt; — Position max (ex: /setpos 0.2)\n"
    "  /setmax &lt;N&gt; — Max positions simultanées\n"
    "  /setlimit &lt;daily|weekly|monthly&gt; &lt;SOL&gt;\n"
    "  /setstrat &lt;strategy&gt; — Stratégie live\n"
    "\n<b>🚨 Actions:</b>\n"
    "  /sell &lt;SYMBOL&gt; — Vendre une position (Solana)\n"
    "  /sellall confirm — PANIC SELL Solana\n"
    "\n<b>Système:</b>\n"
    "  /config — Config active\n"
    "  /help — Cette aide\n"
    "\n<b>Chains:</b> sol 🟣 / eth 🔷 / bsc 🟡 / base 🔵 / all\n"
    "<b>Périodes:</b> 1h 6h 24h 7d 14d 30d all\n"
    "<b>Filtre strat:</b> fast_tp50, bsc_tp100, etc."
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
