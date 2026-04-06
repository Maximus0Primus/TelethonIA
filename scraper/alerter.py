"""
v67: Telegram Bot API alerter for scraper monitoring.

Sends alerts to a private Telegram group via Bot API (zero new dependencies).
Throttled per category to avoid spam. Silent if env vars missing.
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


def short_strat(name: str) -> str:
    """v116: Shorten strategy names for Telegram display.
    v118: Add [LAZY] tag for strategies using LAZY check interval."""
    import re
    # v118: Check if strategy uses LAZY mode
    try:
        from paper_trader import LAZY_STRATEGIES
        is_lazy = name in LAZY_STRATEGIES
    except Exception:
        is_lazy = False
    tag = " [LAZY]" if is_lazy else ""
    # Split DIP: DIP30_B5_P1T5A10S70_P2T10A15S60_240m → DIP30_B5_P1T5A10S70_P2T10A15S60
    m = re.match(r"^(DIP\d+_B\d+_P1T\d+A\d+S\d+_P2T\d+A\d+S\d+)_\d+m$", name)
    if m:
        return m.group(1) + tag
    # Shared DIP: DIP30_B5_T5_A15_SL70_240m → DIP30_B5_T5A15S70
    name = re.sub(r"_B(\d+)_T(\d+)_A(\d+)_SL(\d+)_\d+m$", r"_B\1_T\2A\3S\4", name)
    # DTRAIL/standard: _ACT → A, _SL → S
    name = name.replace("_ACT", "A").replace("_SL", "S")
    return name + tag


TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

_BOT_TOKEN = os.environ.get("MONITOR_BOT_TOKEN")
_CHAT_ID = os.environ.get("MONITOR_CHAT_ID")

# Throttling: category -> last_send_timestamp
_last_alert_times: dict[str, float] = {}
# v5: Exponential backoff — category -> consecutive send count
_alert_counts: dict[str, int] = {}

# Cooldowns per category (seconds) — BASE cooldown, doubles on each repeat
_COOLDOWNS = {
    "cycle_failure": 300,          # 5 min base
    "rt_listener_down": 600,       # 10 min base → 20 → 40 → cap 2h
    "api_errors": 300,             # 5 min per API
    "egress_warning": 3600,        # 1 hour
    "egress_critical": 1800,       # 30 min
    "daily_summary": 86400,        # 24 hours
    "startup": 60,                 # 1 min (prevent double-send on fast restart)
    "live_trade": 0,               # No cooldown — alert every live trade execution
    "ml_disabled": 86400,          # v74: Once per day if ML quality gate failed
    "gh_actions_failure": 3600,    # v74: 1 hour cooldown
    "api_health_warning":  3600,   # v80: 1h between warnings (degraded 70-50%)
    "api_health_critical": 1800,   # v80: 30min between critiques (<50%)
    "api_health_ok":       7200,   # v80: 2h between "recovered" alerts
    # v105: Live trading alerts
    "live_trade_failed":   60,     # 1 min (alert every failure, backoff if chained)
    "wallet_low":          1800,   # 30 min
    "loss_limit_hit":      3600,   # 1 hour per period
    "slippage_deviation":  300,    # 5 min
    "score_anomaly":       3600,   # 1 hour
    "kol_trade":           0,      # v109: No cooldown — alert every KOL trade
    "trade_closed":        0,      # v110: No cooldown — alert every trade close
}

# Max consecutive alerts before going silent (0 = unlimited)
_MAX_ALERTS = {
    "rt_listener_down": 0,         # v74: unlimited (was 5 — silent after 4.5h was dangerous)
    "cycle_failure": 10,
}

# Max backoff cap (seconds)
_MAX_BACKOFF = 7200  # 2 hours


def _can_send(category: str) -> bool:
    base_cooldown = _COOLDOWNS.get(category, 300)
    count = _alert_counts.get(category, 0)
    # Exponential backoff: base * 2^(count-1), capped
    cooldown = min(base_cooldown * (2 ** max(0, count - 1)), _MAX_BACKOFF) if count > 0 else base_cooldown
    last = _last_alert_times.get(category, 0)
    # Check max alerts
    max_alerts = _MAX_ALERTS.get(category, 0)
    if max_alerts > 0 and count >= max_alerts:
        return False
    return (time.time() - last) >= cooldown


def _mark_sent(category: str):
    _last_alert_times[category] = time.time()
    _alert_counts[category] = _alert_counts.get(category, 0) + 1


def reset_alert(category: str):
    """Call when a condition resolves to re-enable alerts for next occurrence."""
    _alert_counts.pop(category, None)
    _last_alert_times.pop(category, None)


def _send(text: str, category: str) -> bool:
    """Send HTML-formatted message to Telegram. Returns True on success."""
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.debug("Monitor alert [%s] suppressed (no bot token/chat_id)", category)
        return False

    if not _can_send(category):
        logger.debug("Monitor alert [%s] throttled", category)
        return False

    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=_BOT_TOKEN),
            json={
                "chat_id": _CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            _mark_sent(category)
            return True
        else:
            logger.warning("Monitor alert [%s] failed: HTTP %d", category, resp.status_code)
            return False
    except Exception as e:
        logger.warning("Monitor alert [%s] error: %s", category, e)
        return False


# --- Public alert functions ---

def alert_cycle_failure(num: int, error: str, duration_s: float):
    text = (
        "<b>CYCLE FAILED</b>\n"
        f"Cycle #{num} crashed after {duration_s:.0f}s\n"
        f"<code>{_truncate(error, 300)}</code>"
    )
    _send(text, "cycle_failure")


def alert_rt_listener_down(last_event_age_min: float):
    text = (
        "<b>RT LISTENER DOWN?</b>\n"
        f"No RT events for {last_event_age_min:.0f} minutes.\n"
        "Possible causes: Telegram disconnect, no new KOL messages, handler crash."
    )
    _send(text, "rt_listener_down")


def alert_api_errors(api: str, error_rate: float, errors: int, calls: int):
    cat = f"api_errors_{api}"
    # Register dynamic category with same cooldown
    if cat not in _COOLDOWNS:
        _COOLDOWNS[cat] = _COOLDOWNS["api_errors"]
    text = (
        f"<b>API ERRORS: {api.upper()}</b>\n"
        f"Error rate: {error_rate*100:.0f}% ({errors}/{calls} calls in 1h)"
    )
    _send(text, cat)


def alert_egress_warning(total_mb: float, by_module: dict, threshold_mb: float):
    is_critical = total_mb >= 750
    category = "egress_critical" if is_critical else "egress_warning"
    level = "CRITICAL" if is_critical else "WARNING"

    top_modules = sorted(by_module.items(), key=lambda x: -x[1])[:5]
    module_lines = "\n".join(f"  {m}: {mb:.1f} MB" for m, mb in top_modules)

    text = (
        f"<b>EGRESS {level}: {total_mb:.0f} MB</b> (threshold: {threshold_mb:.0f} MB)\n"
        f"Breakdown:\n{module_lines}"
    )
    _send(text, category)


def send_daily_summary(snapshot: dict):
    """Send comprehensive daily summary."""
    uptime = snapshot.get("uptime_hours", 0)
    cycles = snapshot.get("cycles", {})
    rt = snapshot.get("rt", {})
    egress = snapshot.get("egress", {})
    paper = snapshot.get("paper", {})
    api_24h = snapshot.get("api_stats_24h", {})

    # Cycle stats
    completed = cycles.get("total_completed", 0)
    errors = len(cycles.get("recent_errors", []))

    # API stats summary
    api_lines = []
    for api, stats in sorted(api_24h.items()):
        api_lines.append(
            f"  {api}: {stats['calls']} calls, "
            f"{stats['error_rate']*100:.0f}% err, "
            f"{stats['avg_latency_ms']:.0f}ms avg"
        )
    api_text = "\n".join(api_lines) if api_lines else "  (no data)"

    # v116: Per-strategy bankroll info
    strat = snapshot.get("strategy", {})
    wl_count = strat.get("whitelist_count", 0)
    strat_bankrolls = strat.get("strategy_bankrolls") or {}

    # Per-strategy bankroll lines
    strat_lines = []
    total_bal = 0
    total_pnl_strat = 0
    for sname, sdata in sorted(strat_bankrolls.items()):
        bal = float(sdata.get("balance", 500))
        pnl = float(sdata.get("pnl", 0))
        trades_n = int(sdata.get("trades", 0))
        short = short_strat(sname)
        e = "📈" if pnl >= 0 else "📉"
        strat_lines.append(f"  {e} {short}: <b>${bal:.0f}</b> ({pnl:+.0f}) | {trades_n}t")
        total_bal += bal
        total_pnl_strat += pnl
    strat_text = "\n".join(strat_lines) if strat_lines else "  (no data)"

    # Paper trade PnL breakdown
    real_pnl = paper.get("pnl_today", 0)
    real_trades = paper.get("closes_today", 0)
    real_wins = paper.get("wins_today", 0)
    wr_today = f"{real_wins*100/real_trades:.0f}%" if real_trades > 0 else "N/A"

    # Whitelist status
    wl_text = f"{wl_count} KOLs" if wl_count > 0 else "OFF (all KOLs)"

    text = (
        "<b>📡 DAILY SUMMARY</b>\n"
        f"Uptime: {uptime:.1f}h\n"
        f"\n<b>📊 Bankrolls:</b>\n{strat_text}\n"
        f"  💵 Total: <b>${total_bal:.0f}</b> ({total_pnl_strat:+.0f})\n"
        f"Whitelist: {wl_text}\n"
        f"\n<b>💰 Paper PnL today:</b>\n"
        f"  Trades: {real_trades} closed ({wr_today} WR)\n"
        f"  PnL: <b>${real_pnl:+.2f}</b>\n"
        f"\n<b>🔄 Cycles:</b> {completed} completed, {errors} errors\n"
        f"\n<b>📡 RT:</b> {rt.get('events', 0)} events, "
        f"{rt.get('trades_opened', 0)} trades, "
        f"{rt.get('unique_kols', 0)} KOLs\n"
        f"\n<b>🔌 APIs (24h):</b>\n{api_text}\n"
        f"\n<b>📦 Egress:</b> {egress.get('total_mb', 0):.1f} MB today"
    )
    _send(text, "daily_summary")


def send_startup_message(total_groups: int, rt_groups: int):
    text = (
        "<b>SCRAPER STARTED</b>\n"
        f"Groups: {total_groups} total, {rt_groups} RT\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    _send(text, "startup")


def alert_live_trade(symbol: str, action: str, amount_sol: float, signature: str):
    """v72: Send Telegram alert for every live trade execution."""
    solscan_link = f"https://solscan.io/tx/{signature}"
    emoji = "BUY" if action == "BUY" else "SELL"
    text = (
        f"<b>LIVE {emoji}: ${symbol}</b>\n"
        f"Amount: {amount_sol:.4f} SOL\n"
        f"<a href=\"{solscan_link}\">View on Solscan</a>"
    )
    _send(text, "live_trade")


def alert_api_health(fill_rates: dict, total: int) -> None:
    """v80: Alert when API fill rates drop — called every 15min cycle.

    fill_rates: {api_name: fill_pct} where 0-100.
    Birdeye is reported as % of expected fill (top-N only), not absolute.
    """
    degraded = {api: pct for api, pct in fill_rates.items() if pct < 85}
    if not degraded:
        reset_alert("api_health_warning")
        reset_alert("api_health_critical")
        return

    critical = {api: pct for api, pct in degraded.items() if pct < 50}
    category = "api_health_critical" if critical else "api_health_warning"
    level    = "🚨 CRITIQUE"        if critical else "⚠️ DÉGRADÉ"

    lines = "\n".join(f"• {api.upper()}: {pct}%" for api, pct in sorted(degraded.items()))
    hints = []
    if "helius"  in degraded: hints.append("helius.dev/dashboard")
    hint_str = " | ".join(hints)

    text = (
        f"<b>API HEALTH {level}</b>\n"
        f"{lines}\n"
        f"Sur {total} tokens ce cycle"
        + (f"\n💡 Crédits: {hint_str}" if hint_str else "")
    )
    _send(text, category)


def alert_ml_disabled(reason: str, horizon: str = ""):
    """v74: Alert when ML model is disabled due to quality gate failure."""
    text = (
        "<b>ML MODEL DISABLED</b>\n"
        f"Horizon: {horizon or 'all'}\n"
        f"Reason: {_truncate(reason, 200)}\n"
        "ML multiplier = 1.0 (no effect on scoring)"
    )
    _send(text, "ml_disabled")


def alert_gh_actions_failure(workflow: str, step: str, error: str):
    """v74: Alert when a GH Actions step fails (called from workflow via curl)."""
    text = (
        f"<b>GH ACTIONS FAILURE</b>\n"
        f"Workflow: {workflow}\n"
        f"Step: {step}\n"
        f"<code>{_truncate(error, 200)}</code>"
    )
    _send(text, "gh_actions_failure")


def alert_live_trade_failed(symbol: str, action: str, error: str):
    """v105: Alert when a live buy/sell transaction fails."""
    text = (
        f"<b>LIVE {action} FAILED: ${symbol}</b>\n"
        f"<code>{_truncate(error, 300)}</code>"
    )
    _send(text, "live_trade_failed")


def alert_wallet_low(balance_sol: float):
    """v105: Alert when wallet SOL balance is critically low."""
    text = (
        f"<b>WALLET LOW: {balance_sol:.4f} SOL</b>\n"
        "Refill wallet to continue trading."
    )
    _send(text, "wallet_low")


def alert_loss_limit_hit(period: str, pnl_sol: float, limit_sol: float):
    """v105: Alert when daily/weekly/monthly loss limit is breached."""
    text = (
        f"<b>LOSS LIMIT HIT ({period.upper()})</b>\n"
        f"PnL: {pnl_sol:+.4f} SOL (limit: -{limit_sol:.1f} SOL)\n"
        "Live buying halted until period reset."
    )
    _send(text, "loss_limit_hit")


def alert_slippage_deviation(symbol: str, expected_bps: int, actual_bps: int):
    """v105: Alert when execution slippage exceeds tolerance."""
    text = (
        f"<b>SLIPPAGE ALERT: ${symbol}</b>\n"
        f"Expected: {expected_bps} bps | Actual: {actual_bps} bps\n"
        f"Deviation: {actual_bps - expected_bps:+d} bps"
    )
    _send(text, "slippage_deviation")


def alert_score_anomaly(old_avg: float, new_avg: float):
    """v105: Alert when average score drops significantly between cycles."""
    drop_pct = round((1 - new_avg / old_avg) * 100, 0) if old_avg > 0 else 0
    text = (
        f"<b>SCORE ANOMALY</b>\n"
        f"Avg score dropped {drop_pct:.0f}%: {old_avg:.1f} → {new_avg:.1f}\n"
        "Possible cause: API failures, data pipeline issue."
    )
    _send(text, "score_anomaly")


def alert_kol_silence(kol_name: str, hours_silent: float):
    """v105: Alert when a whitelisted KOL has been silent too long."""
    cat = f"kol_silence_{kol_name}"
    if cat not in _COOLDOWNS:
        _COOLDOWNS[cat] = 86400  # once per day per KOL
    text = (
        f"<b>KOL SILENT: {kol_name}</b>\n"
        f"No mentions for {hours_silent:.0f}h (threshold: 48h)"
    )
    _send(text, cat)


def alert_kol_trade(symbol: str, kol: str, price: float, pos_usd: float,
                    rt_score: float, liq_usd: float, is_bonding: bool = False,
                    ca: str = "", mcap: float = 0, tier: str = "",
                    bankroll: float = 0, deployed_usd: float = 0,
                    open_count: int = 0,
                    strategy_positions: dict | None = None):
    """v109: Alert when a whitelisted KOL calls a token and a paper trade is opened.
    v116: strategy_positions = {strat_name: {"pos": 120, "balance": 500}} for per-strategy detail."""
    bonding_tag = " 🟡BONDING" if is_bonding else ""
    tier_emoji = "⭐" if tier == "S" else "🔹"

    # Links
    links = []
    if ca:
        links.append(f'<a href="https://dexscreener.com/solana/{ca}">DexScreener</a>')
        if is_bonding:
            links.append(f'<a href="https://pump.fun/{ca}">Pump.fun</a>')
    links_text = " | ".join(links) if links else ""

    mcap_text = f"${mcap/1000:.0f}K" if mcap > 0 else "?"

    # v116: Per-strategy position breakdown replaces global bankroll line
    strat_text = ""
    if strategy_positions:
        parts = []
        for sname, sdata in sorted(strategy_positions.items()):
            pos = float(sdata.get("pos", 0))
            bal = float(sdata.get("balance", 500))
            short = short_strat(sname)
            parts.append(f"  {short}: <b>${pos:.0f}</b> (bank ${bal:.0f})")
        strat_text = "\n📊 Positions:\n" + "\n".join(parts)
    elif bankroll > 0:
        # Fallback for non-hybrid mode
        deployed_after = deployed_usd + pos_usd
        count_after = open_count + 1
        available = bankroll - deployed_after
        strat_text = (
            f"\n💰 <b>${bankroll:.0f}</b> bankroll"
            f" | ${deployed_after:.0f} déployé ({count_after} pos)"
            f" | ${available:.0f} dispo"
        )

    _send(
        f"📢 <b>KOL CALL</b>{bonding_tag}\n\n"
        f"<b>{symbol}</b> called by {tier_emoji}<b>{kol}</b>\n"
        f"💰 Entry: ${price:.8f}\n"
        f"📊 MCap: {mcap_text} | Liq: ${liq_usd/1000:.0f}K | Score: {rt_score:.0f}"
        f"{strat_text}\n"
        f"\n🔗 {links_text}",
        "kol_trade",
    )


def alert_trade_closed(symbol: str, strategy: str, exit_reason: str,
                       pnl_pct: float, pnl_usd: float, pos_usd: float,
                       entry_price: float, exit_price: float,
                       high_price: float, minutes: int,
                       kol: str = "", bankroll: float = 0,
                       ca: str = "", deployed_usd: float = 0,
                       open_count: int = 0,
                       strategy_bankrolls: dict | None = None):
    """v110: Alert when a main trade closes (trail_stop, sl_hit, timeout).
    v115: Shows per-strategy bankroll comparison."""
    # Emoji based on outcome
    if exit_reason == "trail_stop":
        if pnl_pct > 0.5:
            emoji = "🟢"   # big win (>+50%)
        elif pnl_pct > 0:
            emoji = "✅"   # small win
        else:
            emoji = "🟡"   # trail stop at loss (noise exit)
        reason_text = "TRAIL STOP"
    elif exit_reason == "tp_hit":
        emoji = "🎯"
        reason_text = "TP HIT"
    elif exit_reason == "sl_hit":
        emoji = "🔴"
        reason_text = "SL HIT"
    elif exit_reason == "timeout":
        emoji = "⏰" if pnl_pct >= 0 else "🟠"
        reason_text = "TIMEOUT"
    else:
        emoji = "📊"
        reason_text = exit_reason.upper()

    # Max gain from entry
    max_gain = ((high_price / entry_price) - 1) * 100 if entry_price and high_price else 0

    # v116: Per-strategy bankroll replaces global bankroll line
    strat_text = ""
    if strategy_bankrolls:
        parts = []
        for sname, sdata in sorted(strategy_bankrolls.items()):
            bal = float(sdata.get("balance", 500))
            pnl = float(sdata.get("pnl", 0))
            short = short_strat(sname)
            marker = " ◀" if sname == strategy else ""
            parts.append(f"  {short}: <b>${bal:.0f}</b> ({pnl:+.0f}){marker}")
        strat_text = "\n📊 Bankrolls:\n" + "\n".join(parts)
    elif bankroll > 0:
        # Fallback for non-hybrid mode
        deployed_after = max(0, deployed_usd - pos_usd)
        count_after = max(0, open_count - 1)
        available = bankroll - deployed_after
        strat_text = (
            f"\n💰 <b>${bankroll:.0f}</b> bankroll"
            f" | ${deployed_after:.0f} déployé ({count_after} pos)"
            f" | ${available:.0f} dispo"
        )

    # Link
    link_text = f'\n🔗 <a href="https://dexscreener.com/solana/{ca}">DexScreener</a>' if ca else ""

    _send(
        f"{emoji} <b>TRADE {reason_text}</b>\n\n"
        f"<b>{symbol}</b> | {short_strat(strategy)}\n"
        f"👤 KOL: {kol}\n"
        f"📈 PnL: <b>{pnl_pct*100:+.1f}%</b> (${pnl_usd:+.2f})\n"
        f"💵 Position: ${pos_usd:.0f} | ⏱ {minutes}min\n"
        f"📊 Entry: ${entry_price:.8f} → Exit: ${exit_price:.8f}\n"
        f"🔝 Max: {max_gain:+.0f}%"
        f"{strat_text}"
        f"{link_text}",
        "trade_closed",
    )


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
