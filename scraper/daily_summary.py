"""
v74: Standalone daily summary — runs from GH Actions cron.
Queries Supabase for 24h stats and sends Telegram alert.
Independent of scraper loop (fires even if scraper is down).
"""

import os
import json
import logging
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

    # 1. Snapshots created in last 24h
    try:
        snap_result = client.table("token_snapshots").select("id", count="exact").gte(
            "snapshot_at", yesterday
        ).execute()
        n_snapshots = snap_result.count or 0
    except Exception:
        n_snapshots = "?"

    # 2. Paper trades opened/closed in last 24h
    try:
        opened = client.table("paper_trades").select("id", count="exact").gte(
            "created_at", yesterday
        ).execute()
        n_opened = opened.count or 0
    except Exception:
        n_opened = "?"

    try:
        closed = client.table("paper_trades").select("pnl_usd, status").neq(
            "status", "open"
        ).gte("exit_at", yesterday).execute()
        closed_trades = closed.data or []
        n_closed = len(closed_trades)
        pnl_total = sum(float(t.get("pnl_usd") or 0) for t in closed_trades)
        tp_count = sum(1 for t in closed_trades if t.get("status") == "tp_hit")
        sl_count = sum(1 for t in closed_trades if t.get("status") == "sl_hit")
        to_count = sum(1 for t in closed_trades if t.get("status") == "timeout")
        wr = tp_count / n_closed * 100 if n_closed > 0 else 0
    except Exception:
        n_closed = pnl_total = tp_count = sl_count = to_count = wr = 0

    # 3. Live trades
    try:
        live = client.table("paper_trades").select("pnl_usd, status").eq(
            "source", "rt_live"
        ).neq("status", "open").gte("exit_at", yesterday).execute()
        live_trades = live.data or []
        n_live = len(live_trades)
        live_pnl = sum(float(t.get("pnl_usd") or 0) for t in live_trades)
    except Exception:
        n_live = live_pnl = 0

    # 4. Open positions
    try:
        open_result = client.table("paper_trades").select("id", count="exact").eq(
            "status", "open"
        ).execute()
        n_open = open_result.count or 0
    except Exception:
        n_open = "?"

    # 5. KOL mentions in last 24h
    try:
        mentions = client.table("kol_mentions").select("id", count="exact").gte(
            "created_at", yesterday
        ).execute()
        n_mentions = mentions.count or 0
    except Exception:
        n_mentions = "?"

    # 6. ML model status
    try:
        ml = client.table("scoring_config").select("ml_shap_history").eq("id", 1).execute()
        ml_status = "active" if ml.data and ml.data[0].get("ml_shap_history") else "no model"
    except Exception:
        ml_status = "unknown"

    text = (
        "<b>DAILY SUMMARY</b>\n"
        f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"\n<b>Data:</b>\n"
        f"  Snapshots (24h): {n_snapshots}\n"
        f"  KOL mentions (24h): {n_mentions}\n"
        f"\n<b>Paper Trades (24h):</b>\n"
        f"  Opened: {n_opened} | Closed: {n_closed} | Open: {n_open}\n"
        f"  TP: {tp_count} | SL: {sl_count} | Timeout: {to_count}\n"
        f"  Win rate: {wr:.0f}%\n"
        f"  PnL: ${pnl_total:+.2f}\n"
        f"\n<b>Live Trades (24h):</b>\n"
        f"  Closed: {n_live} | PnL: ${live_pnl:+.2f}\n"
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
