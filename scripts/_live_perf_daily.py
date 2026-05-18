"""Daily live trading perf snapshot — runs under GH cron live-perf-daily.yml.

Pulls live trades (source='rt_live'), paper main equivalent (source='rt' AND is_shadow=false),
shadow projection (14d with rolling-24h dedup), computes:
  - per-day breakdown last 14d per strategy
  - drift live↔paper main (mean pp)
  - red-day streak vs 25d historical max
  - KOL concentration on today's trades
  - daily loss vs daily_loss_limit_sol

Telegram alert sent if:
  - drift > 15pp on today's trades
  - 3+ red days in a row (record)
  - daily loss > 50% of daily_loss_limit
  - KOL concentration > 50% on 1 day

References:
  - skill `live-perf-snapshot` — manual interactive equivalent
  - tasks/dedup_rules.md — rolling-24h dedup canonical rule
"""
from __future__ import annotations
import os, sys, io, json, statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Local .env in dev; CI sets env vars directly
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "scraper" / ".env")
except ImportError:
    pass

import requests
from supabase import create_client

SB_URL = os.environ.get("SUPABASE_URL")
SB_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("MONITOR_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("MONITOR_CHAT_ID")

if not SB_URL or not SB_KEY:
    print("FATAL: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing")
    sys.exit(1)

sb = create_client(SB_URL, SB_KEY)
NOW = datetime.now(timezone.utc)
POSITION_MAIN = 50.0  # paper main reference position size
POSITION_LIVE = 1.0   # current live size ($1/trade)


# ──────────────────────────────────────────────────────────────────────────
# Step 1: load production config
# ──────────────────────────────────────────────────────────────────────────
cfg = sb.table("scoring_config").select("paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
SOL_BL = set(cfg["paper_trade_config"].get("kol_chain_blacklist", {}).get("solana", []))
rt_cfg = cfg.get("rt_trade_config", {})
ALLOCS = rt_cfg.get("hybrid_strategy", {}).get("allocations", {})
LIVE_CFG = rt_cfg.get("live_trading", {})
LIVE_ENABLED = LIVE_CFG.get("enabled", False)
DAILY_LOSS_LIMIT_SOL = LIVE_CFG.get("daily_loss_limit_sol", 0.5)
SOL_STRATS = [k for k in ALLOCS if not k.startswith(("ETH_", "BSC_", "BASE_"))]

print(f"=== Live Perf Daily Snapshot — {NOW.isoformat()} ===")
print(f"LIVE_ENABLED: {LIVE_ENABLED} | SOL strats: {SOL_STRATS}")
print(f"SOL blacklist: {len(SOL_BL)} KOLs | daily_loss_limit: {DAILY_LOSS_LIMIT_SOL} SOL")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def pull_paginated(query_builder):
    rows = []
    offset = 0
    while True:
        res = query_builder.order("created_at").range(offset, offset + 999).execute()
        b = res.data or []
        rows.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
    return rows


def rolling_24h_dedup(rows, blacklist):
    """Sort rows by created_at, drop kol_group in blacklist, then dedup by
    (strategy, token_address) within rolling 24h window."""
    rows_sorted = sorted(rows, key=lambda r: r["created_at"])
    last_seen = {}
    allowed = []
    for r in rows_sorted:
        if r.get("kol_group") in blacklist:
            continue
        try:
            ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        except Exception:
            continue
        key = (r["strategy"], r.get("token_address"))
        prev = last_seen.get(key)
        if prev and (ts - prev) < timedelta(hours=24):
            continue
        last_seen[key] = ts
        allowed.append(r)
    return allowed


def closed_only(rows):
    return [r for r in rows if r.get("status") in ("tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop")]


# ──────────────────────────────────────────────────────────────────────────
# Step 2-3: pull live + paper main (last 14d)
# ──────────────────────────────────────────────────────────────────────────
SINCE_14D = (NOW - timedelta(days=14)).isoformat().replace("+00:00", "Z")
SINCE_25D = (NOW - timedelta(days=25)).isoformat().replace("+00:00", "Z")

live_rows = []
main_rows = []
for strat in SOL_STRATS:
    live_rows.extend(pull_paginated(
        sb.table("paper_trades").select(
            "id,strategy,kol_group,token_address,symbol,status,pnl_pct,pnl_usd,position_usd,created_at"
        ).eq("chain", "solana").eq("source", "rt_live").eq("strategy", strat).gte("created_at", SINCE_14D)
    ))
    main_rows.extend(pull_paginated(
        sb.table("paper_trades").select(
            "id,strategy,kol_group,token_address,symbol,status,pnl_pct,pnl_usd,position_usd,created_at"
        ).eq("chain", "solana").eq("source", "rt").eq("is_shadow", False).eq("strategy", strat).gte("created_at", SINCE_14D)
    ))

live_closed = closed_only(live_rows)
main_closed = closed_only(main_rows)
print(f"\nLIVE closed 14d: {len(live_closed)} | PAPER MAIN closed 14d: {len(main_closed)}")


# ──────────────────────────────────────────────────────────────────────────
# Step 4: pull shadow with rolling-24h dedup
# ──────────────────────────────────────────────────────────────────────────
shadow_rows = []
for strat in SOL_STRATS:
    shadow_rows.extend(pull_paginated(
        sb.table("paper_trades").select(
            "strategy,kol_group,token_address,symbol,pnl_pct,pnl_usd,status,created_at"
        ).eq("chain", "solana").eq("is_shadow", True).eq("strategy", strat).in_(
            "status", ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
        ).gte("created_at", SINCE_14D)
    ))
shadow_dedup = rolling_24h_dedup(shadow_rows, SOL_BL)
print(f"SHADOW raw: {len(shadow_rows)} → after rolling-24h dedup + BL: {len(shadow_dedup)}")


# ──────────────────────────────────────────────────────────────────────────
# Step 5: per-day breakdown (14d)
# ──────────────────────────────────────────────────────────────────────────
today_iso = NOW.date().isoformat()
all_days = [(NOW.date() - timedelta(days=13 - i)).isoformat() for i in range(14)]
N_STRATS = max(len(SOL_STRATS), 1)

per_strat_summary = {}
breakdown_lines = []

for strat in SOL_STRATS:
    by_day_live = defaultdict(list)
    by_day_main = defaultdict(list)
    by_day_shadow = defaultdict(list)
    for r in live_closed:
        if r["strategy"] == strat:
            by_day_live[r["created_at"][:10]].append(r)
    for r in main_closed:
        if r["strategy"] == strat:
            by_day_main[r["created_at"][:10]].append(r)
    for r in shadow_dedup:
        if r["strategy"] == strat:
            by_day_shadow[r["created_at"][:10]].append(r)

    breakdown_lines.append(f"\n*{strat}*")
    breakdown_lines.append("```")
    breakdown_lines.append(f"{'date':<12} {'N_l':>3} {'live$':>8} {'N_m':>3} {'main$':>9} {'proj$':>8}")
    cum_live = 0.0
    cum_main = 0.0
    for d in all_days:
        nl = len(by_day_live[d])
        live_usd = sum(float(r.get("pnl_usd") or 0) for r in by_day_live[d])
        nm = len(by_day_main[d])
        main_pct = sum(float(r.get("pnl_pct") or 0) for r in by_day_main[d])
        main_usd = main_pct * POSITION_MAIN
        proj_pct = sum(float(r.get("pnl_pct") or 0) for r in by_day_shadow[d])
        proj_usd = proj_pct * POSITION_MAIN / N_STRATS
        cum_live += live_usd
        cum_main += main_usd
        if nl + nm == 0:
            breakdown_lines.append(f"{d:<12} {'-':>3} {'-':>8} {'-':>3} {'-':>9} {proj_usd:>+8.2f}")
        else:
            breakdown_lines.append(f"{d:<12} {nl:>3} {live_usd:>+8.4f} {nm:>3} {main_usd:>+9.2f} {proj_usd:>+8.2f}")
    breakdown_lines.append(f"{'TOTAL':<12} {'':>3} {cum_live:>+8.4f} {'':>3} {cum_main:>+9.2f}")
    breakdown_lines.append("```")
    per_strat_summary[strat] = {"cum_live": cum_live, "cum_main": cum_main}


# ──────────────────────────────────────────────────────────────────────────
# Step 6: red-day streak vs historical 25d
# ──────────────────────────────────────────────────────────────────────────
hist_rows = []
for strat in SOL_STRATS:
    hist_rows.extend(pull_paginated(
        sb.table("paper_trades").select(
            "strategy,kol_group,token_address,pnl_pct,status,created_at"
        ).eq("chain", "solana").eq("is_shadow", True).eq("strategy", strat).in_(
            "status", ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
        ).gte("created_at", SINCE_25D)
    ))
hist_dedup = rolling_24h_dedup(hist_rows, SOL_BL)
agg = defaultdict(float)
for r in hist_dedup:
    agg[r["created_at"][:10]] += float(r.get("pnl_pct") or 0) * POSITION_MAIN / N_STRATS
days_sorted = sorted(agg.keys())
max_red_streak = 0
cur = 0
for d in days_sorted:
    if agg[d] < 0:
        cur += 1
        max_red_streak = max(max_red_streak, cur)
    else:
        cur = 0

# Current live red streak (today and going back)
live_agg = defaultdict(float)
for r in live_closed:
    live_agg[r["created_at"][:10]] += float(r.get("pnl_usd") or 0)
sorted_desc = sorted(live_agg.keys(), reverse=True)
current_red = 0
for d in sorted_desc:
    if live_agg[d] < 0:
        current_red += 1
    else:
        break

if current_red < max_red_streak:
    sigma_status = "NORMAL"
elif current_red == max_red_streak:
    sigma_status = "AT LIMIT"
else:
    sigma_status = "RECORD BROKEN"


# ──────────────────────────────────────────────────────────────────────────
# Step 7: drift live↔paper main (today's trades) + concentration + verdict
# ──────────────────────────────────────────────────────────────────────────
paired = []
for lr in live_closed:
    if lr["created_at"][:10] != today_iso:
        continue
    lt = datetime.fromisoformat(lr["created_at"].replace("Z", "+00:00"))
    for mr in main_closed:
        if mr["strategy"] != lr["strategy"] or mr["token_address"] != lr["token_address"]:
            continue
        mt = datetime.fromisoformat(mr["created_at"].replace("Z", "+00:00"))
        if abs((mt - lt).total_seconds()) < 3600:
            paired.append((lr, mr))
            break

mean_drift = 0.0
if paired:
    drifts = [(float(lr["pnl_pct"]) - float(mr["pnl_pct"])) * 100 for lr, mr in paired]
    mean_drift = statistics.mean(drifts)

# KOL concentration on today
kol_today = defaultdict(float)
for r in live_closed + main_closed:
    if r["created_at"][:10] != today_iso:
        continue
    pnl_usd = float(r.get("pnl_usd") or 0)
    if r in main_closed:
        pnl_usd = float(r.get("pnl_pct") or 0) * POSITION_MAIN
    kol_today[r.get("kol_group") or "unknown"] += pnl_usd

total_abs = sum(abs(v) for v in kol_today.values())
top_kol_pct = 0.0
top_kol_name = "-"
if total_abs > 0:
    top_kol_name, top_kol_pnl = max(kol_today.items(), key=lambda x: abs(x[1]))
    top_kol_pct = 100 * abs(top_kol_pnl) / total_abs

# Daily live loss (USD), compare vs daily_loss_limit_sol × SOL_PRICE (assume $86)
today_live_usd = live_agg.get(today_iso, 0.0)
SOL_PRICE_EST = 86.0
daily_loss_limit_usd = DAILY_LOSS_LIMIT_SOL * SOL_PRICE_EST
loss_pct = (abs(today_live_usd) / daily_loss_limit_usd * 100) if today_live_usd < 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Alert decision
# ──────────────────────────────────────────────────────────────────────────
alerts = []
if abs(mean_drift) > 15 and len(paired) >= 3:
    alerts.append(f"DRIFT {mean_drift:+.1f}pp on {len(paired)} paired trades today")
if current_red >= 3:
    alerts.append(f"RED STREAK {current_red} days (hist max {max_red_streak})")
if sigma_status == "RECORD BROKEN":
    alerts.append(f"STREAK RECORD BROKEN: {current_red} > {max_red_streak}d historical max")
if loss_pct > 50:
    alerts.append(f"DAILY LOSS {today_live_usd:+.2f}$ = {loss_pct:.0f}% of limit (${daily_loss_limit_usd:.2f})")
if top_kol_pct > 50 and total_abs > 0.5:
    alerts.append(f"KOL CONCENTRATION: {top_kol_name} = {top_kol_pct:.0f}% of today's PnL")

if abs(mean_drift) > 15 or current_red >= 3 or sigma_status == "RECORD BROKEN" or loss_pct > 50 or top_kol_pct > 50:
    verdict = "🔴 ALERT"
elif abs(mean_drift) > 8 or current_red == 2 or sigma_status == "AT LIMIT":
    verdict = "🟡 CAUTION"
else:
    verdict = "🟢 GO"


# ──────────────────────────────────────────────────────────────────────────
# Compose report
# ──────────────────────────────────────────────────────────────────────────
total_live = sum(s["cum_live"] for s in per_strat_summary.values())
total_main = sum(s["cum_main"] for s in per_strat_summary.values())

header = [
    f"*Live Perf Daily — {NOW.date().isoformat()}*",
    f"Live: {'ON' if LIVE_ENABLED else 'OFF'} | Strats: {len(SOL_STRATS)} | Verdict: {verdict}",
    "",
    f"14d totals: live=${total_live:+.4f} | paper_main=${total_main:+.2f} (@$50/trade)",
    f"Drift today (N={len(paired)}): {mean_drift:+.2f}pp",
    f"Red streak: {current_red}d (hist max 25d: {max_red_streak}d) → {sigma_status}",
    f"Today live PnL: ${today_live_usd:+.4f} | loss vs limit: {loss_pct:.0f}%",
    f"Top KOL today: {top_kol_name} ({top_kol_pct:.0f}% of today PnL)",
]
if alerts:
    header.append("")
    header.append("*ALERTS:*")
    for a in alerts:
        header.append(f"- {a}")

report = "\n".join(header + breakdown_lines)
print("\n" + report)


# ──────────────────────────────────────────────────────────────────────────
# Telegram send (only if alert condition met OR forced via env)
# ──────────────────────────────────────────────────────────────────────────
def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print("\n[telegram] TG creds missing — skipping send")
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    # Telegram message max 4096 chars; truncate body if needed
    if len(text) > 3900:
        text = text[:3900] + "\n...[truncated]"
    try:
        r = requests.post(url, data={
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }, timeout=10)
        if r.status_code == 200:
            print(f"[telegram] sent OK")
            return True
        print(f"[telegram] HTTP {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"[telegram] error: {e}")
        return False


force_send = os.environ.get("FORCE_TG_SEND", "").lower() in ("1", "true", "yes")
if alerts or force_send:
    send_telegram(report)
else:
    print("\n[telegram] no alert conditions met — skipping send (set FORCE_TG_SEND=1 to override)")

print("\n=== DONE ===")
