"""
v76: Standalone daily summary — runs from GH Actions cron.
Two-message report:
  Msg 1: Alerts + 24h overview + per-strategy table (WR, PnL, ROI, avg pos, TP/SL)
  Msg 2: 7-day strategy simulation + per-day PnL table
"""

import os
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import requests
from supabase import create_client

logger = logging.getLogger(__name__)

# v92: dynamic deprecated — fallback to hardcoded when no DB client
from paper_trader import _DEFAULT_DEPRECATED
LEGACY_STRATEGIES = _DEFAULT_DEPRECATED


def _send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def _strat_row(strat: str, st: dict) -> str:
    n = st["n"]
    wins = st["wins"]
    tp = st["tp"]
    sl = st["sl"]
    to = st["to"]
    pnl = st["pnl"]
    invested = st["invested"]
    avg_pos = st["avg_pos"]
    wr = round(wins / n * 100) if n else 0
    roi = round(pnl / invested * 100, 1) if invested > 0 else 0.0
    sign = "📈" if pnl >= 0 else "📉"
    tag = " [leg]" if strat in LEGACY_STRATEGIES else ""
    return (
        f"  {sign} <b>{strat}{tag}</b>: {n}t "
        f"WR {wr}% TP {tp}/SL {sl}/TO {to} "
        f"${pnl:+.0f}/${invested:.0f} ({avg_pos:.0f}$/t ROI {roi:+.1f}%)"
    )


def _aggregate(trades: list[dict]) -> dict:
    """Per-strategy stats."""
    strats: dict = defaultdict(lambda: {
        "n": 0, "wins": 0, "tp": 0, "sl": 0, "to": 0,
        "pnl": 0.0, "invested": 0.0, "avg_pos": 0.0,
    })
    for t in trades:
        s = t.get("strategy", "?")
        pos = float(t.get("position_usd") or 0)
        pnl = float(t.get("pnl_usd") or 0)
        strats[s]["n"] += 1
        strats[s]["pnl"] += pnl
        strats[s]["invested"] += pos
        if pnl > 0:
            strats[s]["wins"] += 1
        st = t.get("status", "")
        if st == "tp_hit":
            strats[s]["tp"] += 1
        elif st == "sl_hit":
            strats[s]["sl"] += 1
        elif st == "timeout":
            strats[s]["to"] += 1
    for s, d in strats.items():
        d["avg_pos"] = d["invested"] / d["n"] if d["n"] else 0
    return strats


def _source_line(source: str, trades: list[dict]) -> str:
    if source == "rt":
        sub = [t for t in trades if t.get("source") == "rt"]
    else:
        # Batch trades have source=NULL in DB
        sub = [t for t in trades if t.get("source") != "rt"]
    if not sub:
        return ""
    n = len(sub)
    wins = sum(1 for t in sub if float(t.get("pnl_usd") or 0) > 0)
    pnl = sum(float(t.get("pnl_usd") or 0) for t in sub)
    inv = sum(float(t.get("position_usd") or 0) for t in sub)
    wr = round(wins / n * 100)
    roi = round(pnl / inv * 100, 1) if inv > 0 else 0
    sign = "📈" if pnl >= 0 else "📉"
    label = "RT" if source == "rt" else "BATCH"
    return f"  {label}: {n}t WR {wr}% {sign} ${pnl:+.0f}/${inv:.0f} (ROI {roi:+.1f}%)"


def send_summary():
    """Query Supabase for 24h + 7d stats and send two Telegram messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    bot_token = os.environ.get("MONITOR_BOT_TOKEN")
    chat_id = os.environ.get("MONITOR_CHAT_ID")

    if not url or not key:
        logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        return
    if not bot_token or not chat_id:
        logger.error("Missing MONITOR_BOT_TOKEN or MONITOR_CHAT_ID")
        return

    client = create_client(url, key)
    # v92: refresh deprecated strategies from DB
    from paper_trader import get_deprecated_strategies
    global LEGACY_STRATEGIES
    LEGACY_STRATEGIES = get_deprecated_strategies(client)
    now = datetime.now(timezone.utc)
    h24_ago = (now - timedelta(hours=24)).isoformat()
    d7_ago = (now - timedelta(days=7)).isoformat()
    d14_ago = (now - timedelta(days=14)).isoformat()  # wide fetch to catch long-running trades

    # ── API health ──
    api_alerts = []
    whale_fill_pct: str | int = "?"
    helius_fill_pct: str | int = "?"
    avg_score: str | float = "?"
    n_snapshots: str | int = "?"
    n_mentions: str | int = "?"

    try:
        r = client.table("token_snapshots").select("id", count="exact").gte(
            "snapshot_at", h24_ago).limit(1).execute()
        n_snapshots = r.count or 0
    except Exception:
        pass

    try:
        sample = client.table("token_snapshots").select(
            "whale_count,helius_holder_count,score_at_snapshot"
        ).gte("snapshot_at", h24_ago).limit(500).execute()
        rows = sample.data or []
        if rows:
            has_whale = sum(1 for r in rows if r.get("whale_count") is not None)
            has_helius = sum(1 for r in rows if r.get("helius_holder_count") is not None)
            scores = [float(r["score_at_snapshot"]) for r in rows
                      if r.get("score_at_snapshot") is not None]
            whale_fill_pct = round(has_whale / len(rows) * 100)
            helius_fill_pct = round(has_helius / len(rows) * 100)
            avg_score = round(sum(scores) / len(scores), 1) if scores else "?"
            if isinstance(whale_fill_pct, int) and whale_fill_pct < 50:
                api_alerts.append(
                    f"🚨 HELIUS DOWN — whale fill {whale_fill_pct}% (check helius.dev credits)"
                )
            if isinstance(helius_fill_pct, int) and helius_fill_pct < 50:
                api_alerts.append(
                    f"🚨 BIRDEYE/HELIUS DOWN — holder fill {helius_fill_pct}%"
                )
    except Exception as e:
        logger.warning("API health sample failed: %s", e)

    try:
        r = client.table("kol_mentions").select("id", count="exact").gte(
            "created_at", h24_ago).limit(1).execute()
        n_mentions = r.count or 0
    except Exception:
        pass

    # ── Trade counts ──
    n_opened: str | int = "?"
    n_open: str | int = "?"
    try:
        r = client.table("paper_trades").select("id", count="exact").gte(
            "created_at", h24_ago).limit(1).execute()
        n_opened = r.count or 0
    except Exception:
        pass
    try:
        r = client.table("paper_trades").select("id", count="exact").eq(
            "status", "open").limit(1).execute()
        n_open = r.count or 0
    except Exception:
        pass

    # ── Fetch closed trades (14d window to catch long-running MOONBAG/WIDE_RUNNER) ──
    trades_raw: list[dict] = []
    try:
        r = client.table("paper_trades").select(
            "pnl_usd,pnl_pct,status,strategy,source,position_usd,created_at,exit_at,kol_group,is_shadow"
        ).neq("status", "open").gte("created_at", d14_ago).execute()
        trades_raw = r.data or []
    except Exception as e:
        logger.warning("trades fetch failed: %s", e)

    # Separate real vs shadow trades
    real_raw = [t for t in trades_raw if not t.get("is_shadow")]
    shadow_raw = [t for t in trades_raw if t.get("is_shadow")]

    # Filter by exit_at for consistent time windows (a trade counts in the period it CLOSED)
    trades_7d = [t for t in real_raw if t.get("exit_at") and t["exit_at"] >= d7_ago]
    trades_24h = [t for t in real_raw if t.get("exit_at") and t["exit_at"] >= h24_ago]
    shadow_7d = [t for t in shadow_raw if t.get("exit_at") and t["exit_at"] >= d7_ago]
    shadow_24h = [t for t in shadow_raw if t.get("exit_at") and t["exit_at"] >= h24_ago]

    strats_24h = _aggregate(trades_24h)
    strats_7d = _aggregate(trades_7d)

    # ── 24h totals ──
    n_closed = len(trades_24h)
    pnl_24h = sum(float(t.get("pnl_usd") or 0) for t in trades_24h)
    wins_24h = sum(1 for t in trades_24h if float(t.get("pnl_usd") or 0) > 0)
    tp_24h = sum(1 for t in trades_24h if t.get("status") == "tp_hit")
    sl_24h = sum(1 for t in trades_24h if t.get("status") == "sl_hit")
    to_24h = sum(1 for t in trades_24h if t.get("status") == "timeout")
    invested_24h = sum(float(t.get("position_usd") or 0) for t in trades_24h)
    wr_24h = round(wins_24h / n_closed * 100) if n_closed else 0
    roi_24h = round(pnl_24h / invested_24h * 100, 1) if invested_24h > 0 else 0

    # ── ML status ──
    ml_status = "unknown"
    try:
        ml = client.table("scoring_config").select(
            "ml_shap_history,ml_horizon,ml_threshold"
        ).eq("id", 1).execute()
        if ml.data and ml.data[0].get("ml_shap_history"):
            h = ml.data[0].get("ml_horizon", "?")
            th = ml.data[0].get("ml_threshold", "?")
            ml_status = f"active (horizon={h}, threshold={th}x)"
        else:
            ml_status = "no model"
    except Exception:
        pass

    # ── Message 1: 24h ──
    alert_block = ("\n" + "\n".join(api_alerts) + "\n") if api_alerts else ""
    pnl_emoji = "📈" if pnl_24h >= 0 else "📉"

    strat_rows_24h = [
        _strat_row(s, strats_24h[s])
        for s in sorted(strats_24h, key=lambda x: strats_24h[x]["pnl"], reverse=True)
    ]
    rt_line = _source_line("rt", trades_24h)
    batch_line = _source_line("batch", trades_24h)  # "batch" → filters source != "rt"

    msg1 = (
        f"<b>📊 DAILY SUMMARY</b> — {now.strftime('%Y-%m-%d %H:%M UTC')}"
        f"\n{alert_block}"
        f"\n<b>Data Quality:</b>"
        f"\n  Snapshots: {n_snapshots} | Mentions: {n_mentions}"
        f"\n  Score moy: {avg_score}/100 | Whale: {whale_fill_pct}% | Holder: {helius_fill_pct}%"
        f"\n\n<b>Trades 24h:</b>"
        f"\n  Ouvert: {n_opened} | Fermé: {n_closed} | Open: {n_open}"
        f"\n  TP: {tp_24h} | SL: {sl_24h} | Timeout: {to_24h}"
        f"\n  WR: {wr_24h}% | {pnl_emoji} PnL: ${pnl_24h:+.2f}/${invested_24h:.0f} (ROI {roi_24h:+.1f}%)"
        + (f"\n{rt_line}" if rt_line else "")
        + (f"\n{batch_line}" if batch_line else "")
        + "\n\n<b>Stratégies 24h:</b>\n"
        + "\n".join(strat_rows_24h)
        + f"\n\n<b>ML:</b> {ml_status}"
    )

    # ── Message 2: 7-day ──
    strat_rows_7d = [
        _strat_row(s, strats_7d[s])
        for s in sorted(strats_7d, key=lambda x: strats_7d[x]["pnl"], reverse=True)
    ]

    # Daily PnL table
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_inv: dict[str, float] = defaultdict(float)
    for t in trades_7d:
        if t.get("exit_at"):
            day = t["exit_at"][:10]
            daily_pnl[day] += float(t.get("pnl_usd") or 0)
            daily_inv[day] += float(t.get("position_usd") or 0)

    daily_lines = []
    cumul = 0.0
    for day in sorted(daily_pnl):
        p = daily_pnl[day]
        inv = daily_inv[day]
        cumul += p
        roi_d = round(p / inv * 100, 1) if inv > 0 else 0
        sign = "📈" if p >= 0 else "📉"
        daily_lines.append(
            f"  {day}: {sign} ${p:+.0f}/${inv:.0f} "
            f"(ROI {roi_d:+.1f}%) cumul ${cumul:+.0f}"
        )

    total_7d_pnl = sum(daily_pnl.values())
    total_7d_inv = sum(daily_inv.values())
    total_7d_roi = round(total_7d_pnl / total_7d_inv * 100, 1) if total_7d_inv > 0 else 0
    emoji_7d = "📈" if total_7d_pnl >= 0 else "📉"

    # ── KOL Leaderboard (7d, split RT / batch) ──
    def _build_kol_stats(trade_list: list[dict]) -> dict:
        stats: dict[str, dict] = {}
        for t in trade_list:
            kol = t.get("kol_group")
            if not kol:
                continue
            if kol not in stats:
                stats[kol] = {"n": 0, "wins": 0, "wins50": 0, "pnl": 0.0}
            pnl = float(t.get("pnl_usd") or 0)
            pnl_pct = float(t.get("pnl_pct") or 0)
            stats[kol]["n"] += 1
            stats[kol]["pnl"] += pnl
            if pnl > 0:
                stats[kol]["wins"] += 1
            if pnl_pct >= 0.50:
                stats[kol]["wins50"] += 1
        return stats

    def _format_kol_lines(stats: dict) -> list[str]:
        lines = []
        for kol in sorted(stats, key=lambda k: stats[k]["pnl"], reverse=True):
            kd = stats[kol]
            wr = round(kd["wins"] / kd["n"] * 100) if kd["n"] else 0
            wr50 = round(kd["wins50"] / kd["n"] * 100) if kd["n"] else 0
            sign = "🟢" if kd["pnl"] >= 0 else "🔴"
            lines.append(
                f"  {sign} {kol}: {kd['n']}t WR {wr}% +50%:{wr50}% ${kd['pnl']:+.0f}"
            )
        return lines

    rt_7d = [t for t in trades_7d if t.get("source") == "rt"]
    batch_7d = [t for t in trades_7d if t.get("source") != "rt"]

    rt_kol = _build_kol_stats(rt_7d)
    batch_kol = _build_kol_stats(batch_7d)
    all_kol = _build_kol_stats(trades_7d)

    kol_block = ""
    rt_lines = _format_kol_lines(rt_kol)
    if rt_lines:
        kol_block += "\n\n<b>KOL Leaderboard 7j (RT):</b>\n" + "\n".join(rt_lines)
    batch_lines = _format_kol_lines(batch_kol)
    if batch_lines:
        kol_block += "\n\n<b>KOL Leaderboard 7j (Batch):</b>\n" + "\n".join(batch_lines)
    if not rt_lines and not batch_lines:
        all_lines = _format_kol_lines(all_kol)
        if all_lines:
            kol_block = "\n\n<b>KOL Leaderboard 7j:</b>\n" + "\n".join(all_lines)

    # ── Shadow Strategy Comparison (all strategies on same tokens) ──
    # Combine real + shadow trades for apples-to-apples comparison using pnl_pct
    VIRTUAL_POS = 15.0  # reference position for shadow PnL calc
    all_closed_7d = trades_7d + shadow_7d
    shadow_strats: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "tp": 0, "sl": 0, "to": 0})
    for t in all_closed_7d:
        s = t.get("strategy", "?")
        pnl_pct = float(t.get("pnl_pct") or 0)
        virtual_pnl = VIRTUAL_POS * pnl_pct
        shadow_strats[s]["n"] += 1
        shadow_strats[s]["pnl"] += virtual_pnl
        if pnl_pct > 0:
            shadow_strats[s]["wins"] += 1
        st = t.get("status", "")
        if st == "tp_hit":
            shadow_strats[s]["tp"] += 1
        elif st == "sl_hit":
            shadow_strats[s]["sl"] += 1
        elif st == "timeout":
            shadow_strats[s]["to"] += 1

    shadow_lines = []
    for s in sorted(shadow_strats, key=lambda x: shadow_strats[x]["pnl"], reverse=True):
        sd = shadow_strats[s]
        n = sd["n"]
        wr = round(sd["wins"] / n * 100) if n else 0
        roi = round(sd["pnl"] / (VIRTUAL_POS * n) * 100, 1) if n else 0
        sign = "📈" if sd["pnl"] >= 0 else "📉"
        shadow_lines.append(
            f"  {sign} {s}: {n}t WR {wr}% TP{sd['tp']}/SL{sd['sl']}/TO{sd['to']} "
            f"ROI {roi:+.1f}%"
        )
    shadow_block = ""
    if shadow_lines:
        n_shadow = sum(1 for t in all_closed_7d if t.get("is_shadow"))
        shadow_block = (
            f"\n\n<b>🔬 Shadow Comparison 7j</b> (mêmes tokens, ${VIRTUAL_POS:.0f}/t virtuel, {n_shadow} shadow):\n"
            + "\n".join(shadow_lines)
        )

    msg2 = (
        f"<b>📅 SIMULATION 7 JOURS</b> — au {now.strftime('%Y-%m-%d')}"
        f"\n\n<b>Stratégies 7j:</b>\n"
        + "\n".join(strat_rows_7d)
        + kol_block
        + shadow_block
        + f"\n\n<b>PnL journalier:</b>\n"
        + "\n".join(daily_lines)
        + f"\n\n<b>Total 7j:</b> {len(trades_7d)}t | "
        + f"{emoji_7d} ${total_7d_pnl:+.0f}/${total_7d_inv:.0f} (ROI {total_7d_roi:+.1f}%)"
    )

    ok1 = _send_telegram(bot_token, chat_id, msg1)
    ok2 = _send_telegram(bot_token, chat_id, msg2)
    if ok1 and ok2:
        logger.info("Daily summary sent (2 messages)")
    else:
        logger.error("Daily summary partial failure: msg1=%s msg2=%s", ok1, ok2)


if __name__ == "__main__":
    send_summary()
