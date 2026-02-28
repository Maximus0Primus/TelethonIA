"""
v75: Standalone daily summary — runs from GH Actions cron.
Adds: API health (whale/helius fill rates), per-strategy breakdown, RT vs Batch split,
      alerts when critical APIs are down.
"""

import os
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import requests
from supabase import create_client

logger = logging.getLogger(__name__)


def send_summary():
    """Query Supabase for 24h stats and send Telegram summary."""
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
    now = datetime.now(timezone.utc)
    yesterday = (now - timedelta(hours=24)).isoformat()

    # ── 1. Snapshots count + API health (sample 500 rows, representative) ──
    n_snapshots = "?"
    whale_fill_pct = "?"
    helius_fill_pct = "?"
    avg_score = "?"
    api_alerts = []
    try:
        count_result = client.table("token_snapshots").select(
            "id", count="exact"
        ).gte("snapshot_at", yesterday).limit(1).execute()
        n_snapshots = count_result.count or 0
    except Exception as e:
        logger.warning("Snapshot count failed: %s", e)

    try:
        sample = client.table("token_snapshots").select(
            "whale_count,helius_holder_count,score_at_snapshot"
        ).gte("snapshot_at", yesterday).limit(500).execute()
        rows = sample.data or []
        if rows:
            has_whale = sum(1 for r in rows if r.get("whale_count") is not None)
            has_helius = sum(1 for r in rows if r.get("helius_holder_count") is not None)
            scores = [float(r["score_at_snapshot"]) for r in rows
                      if r.get("score_at_snapshot") is not None]
            whale_fill_pct = round(has_whale / len(rows) * 100)
            helius_fill_pct = round(has_helius / len(rows) * 100)
            avg_score = round(sum(scores) / len(scores), 1) if scores else "?"
            if whale_fill_pct < 50:
                api_alerts.append(
                    f"🚨 HELIUS API DOWN — whale_count fill: {whale_fill_pct}%"
                    " (scores blind, check helius.dev credits)"
                )
            if helius_fill_pct < 50:
                api_alerts.append(
                    f"🚨 BIRDEYE/HELIUS DOWN — holder fill: {helius_fill_pct}%"
                )
    except Exception as e:
        logger.warning("API health sample failed: %s", e)

    # ── 2. Paper trades opened today ──
    n_opened = "?"
    n_open = "?"
    try:
        opened = client.table("paper_trades").select(
            "id", count="exact"
        ).gte("created_at", yesterday).limit(1).execute()
        n_opened = opened.count or 0
    except Exception:
        pass

    try:
        open_result = client.table("paper_trades").select(
            "id", count="exact"
        ).eq("status", "open").limit(1).execute()
        n_open = open_result.count or 0
    except Exception:
        pass

    # ── 3. Closed trades in last 24h (full fetch for aggregation) ──
    closed_trades = []
    try:
        closed = client.table("paper_trades").select(
            "pnl_usd,status,strategy,source"
        ).neq("status", "open").gte("exit_at", yesterday).execute()
        closed_trades = closed.data or []
    except Exception as e:
        logger.warning("Closed trades query failed: %s", e)

    n_closed = len(closed_trades)
    pnl_total = sum(float(t.get("pnl_usd") or 0) for t in closed_trades)
    tp_count = sum(1 for t in closed_trades if t.get("status") == "tp_hit")
    sl_count = sum(1 for t in closed_trades if t.get("status") == "sl_hit")
    to_count = sum(1 for t in closed_trades if t.get("status") == "timeout")
    wr = tp_count / n_closed * 100 if n_closed > 0 else 0

    # Per-source: RT vs Batch
    def _source_stats(source):
        trades = [t for t in closed_trades if t.get("source") == source]
        if not trades:
            return None
        n = len(trades)
        tp = sum(1 for t in trades if t.get("status") == "tp_hit")
        pnl = sum(float(t.get("pnl_usd") or 0) for t in trades)
        return {"n": n, "wr": round(tp / n * 100), "pnl": pnl}

    rt_stats = _source_stats("rt")
    batch_stats = _source_stats("batch")

    # Per-strategy breakdown (sorted by PnL desc)
    strat_stats = defaultdict(lambda: {"n": 0, "tp": 0, "sl": 0, "pnl": 0.0})
    for t in closed_trades:
        s = t.get("strategy", "?")
        strat_stats[s]["n"] += 1
        strat_stats[s]["pnl"] += float(t.get("pnl_usd") or 0)
        if t.get("status") == "tp_hit":
            strat_stats[s]["tp"] += 1
        elif t.get("status") == "sl_hit":
            strat_stats[s]["sl"] += 1

    strat_lines = []
    for strat, st in sorted(strat_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        wr_s = round(st["tp"] / st["n"] * 100) if st["n"] else 0
        sign = "📈" if st["pnl"] >= 0 else "📉"
        strat_lines.append(
            f"  {sign} {strat}: {st['n']} | WR {wr_s}% | ${st['pnl']:+.0f}"
        )

    # ── 4. Live trades ──
    n_live = 0
    live_pnl = 0.0
    try:
        live = client.table("paper_trades").select(
            "pnl_usd,status"
        ).eq("source", "rt_live").neq("status", "open").gte("exit_at", yesterday).execute()
        live_trades = live.data or []
        n_live = len(live_trades)
        live_pnl = sum(float(t.get("pnl_usd") or 0) for t in live_trades)
    except Exception:
        pass

    # ── 5. KOL mentions ──
    n_mentions = "?"
    try:
        mentions = client.table("kol_mentions").select(
            "id", count="exact"
        ).gte("created_at", yesterday).limit(1).execute()
        n_mentions = mentions.count or 0
    except Exception:
        pass

    # ── 6. ML status ──
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

    # ── Build message ──
    pnl_emoji = "📈" if pnl_total >= 0 else "📉"
    alert_block = ("\n\n" + "\n".join(api_alerts)) if api_alerts else ""

    rt_line = (
        f"\n  RT:    {rt_stats['n']} trades | WR {rt_stats['wr']}% | ${rt_stats['pnl']:+.0f}"
        if rt_stats else ""
    )
    batch_line = (
        f"\n  Batch: {batch_stats['n']} trades | WR {batch_stats['wr']}% | ${batch_stats['pnl']:+.0f}"
        if batch_stats else ""
    )
    strat_block = (
        "\n\n<b>Strategies (24h):</b>\n" + "\n".join(strat_lines)
        if strat_lines else ""
    )

    text = (
        f"<b>DAILY SUMMARY</b> — {now.strftime('%Y-%m-%d %H:%M UTC')}"
        f"{alert_block}\n"
        f"\n<b>Data Quality:</b>\n"
        f"  Snapshots: {n_snapshots} | KOL mentions: {n_mentions}\n"
        f"  Avg score: {avg_score}/100\n"
        f"  Whale fill: {whale_fill_pct}% | Holder fill: {helius_fill_pct}%\n"
        f"\n<b>Paper Trades (24h):</b>\n"
        f"  Opened: {n_opened} | Closed: {n_closed} | Open: {n_open}\n"
        f"  TP: {tp_count} | SL: {sl_count} | Timeout: {to_count}\n"
        f"  WR: {wr:.0f}% | {pnl_emoji} PnL: ${pnl_total:+.2f}"
        f"{rt_line}"
        f"{batch_line}"
        f"{strat_block}\n"
        f"\n<b>Live Trades (24h):</b> {n_live} closed | PnL: ${live_pnl:+.2f}\n"
        f"\n<b>ML:</b> {ml_status}"
    )

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
        if resp.status_code == 200:
            logger.info("Daily summary sent successfully")
        else:
            logger.error("Daily summary failed: HTTP %d", resp.status_code)
    except Exception as e:
        logger.error("Daily summary send failed: %s", e)


if __name__ == "__main__":
    send_summary()
