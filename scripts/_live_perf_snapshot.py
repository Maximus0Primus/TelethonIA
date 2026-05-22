"""Live perf snapshot — live vs paper main vs shadow 14d (rolling-24h dedup)."""
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

NOW = datetime.now(timezone.utc)
EXIT = ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]

# ---- Step 1: config ----
cfg = sb.table("scoring_config").select("paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
SOL_BL = set(cfg["paper_trade_config"].get("kol_chain_blacklist", {}).get("solana", []))
ALLOCS = cfg["rt_trade_config"].get("hybrid_strategy", {}).get("allocations", {})
LIVE = cfg["rt_trade_config"].get("live_trading", {})
LIVE_ENABLED = LIVE.get("enabled", False)
LIVE_KOL_BL = set(LIVE.get("kol_blacklist", []) or [])
MAX_POS_SOL = LIVE.get("max_position_sol")
SOL_STRATS = [k for k in ALLOCS if not k.startswith(("ETH_", "BSC_", "BASE_"))]
N_STRATS = len(SOL_STRATS)
print(f"LIVE_ENABLED={LIVE_ENABLED} | max_position_sol={MAX_POS_SOL} | SOL strats live={N_STRATS}: {SOL_STRATS}")
print(f"SOL blacklist={len(SOL_BL)} KOLs | live kol_blacklist={len(LIVE_KOL_BL)}")


def pull(filters, since, cols):
    rows = []
    for strat in SOL_STRATS:
        offset = 0
        while True:
            q = sb.table("paper_trades").select(cols).eq("chain", "solana").eq("strategy", strat).gte("created_at", since)
            for k, v in filters.items():
                q = q.eq(k, v)
            res = q.order("created_at").range(offset, offset + 999).execute()
            b = res.data or []
            rows.extend(b)
            if len(b) < 1000:
                break
            offset += 1000
    return rows


COLS = "id,strategy,kol_group,token_address,symbol,status,pnl_pct,pnl_usd,position_usd,created_at"
SINCE_7D = (NOW - timedelta(days=7)).isoformat().replace("+00:00", "Z")
SINCE_14D = (NOW - timedelta(days=14)).isoformat().replace("+00:00", "Z")
SINCE_25D = (NOW - timedelta(days=25)).isoformat().replace("+00:00", "Z")

# ---- Step 2/3: live + paper main ----
live_rows = pull({"source": "rt_live"}, SINCE_7D, COLS)
main_rows = pull({"source": "rt", "is_shadow": False}, SINCE_7D, COLS)

# Detect actual live window
live_dates = sorted(r["created_at"] for r in live_rows)
if live_dates:
    first_live = datetime.fromisoformat(live_dates[0].replace("Z", "+00:00"))
    age_h = (NOW - first_live).total_seconds() / 3600
    print(f"\nLive trades: N={len(live_rows)} | first={live_dates[0][:19]} | age={age_h:.0f}h | last={live_dates[-1][:19]}")
else:
    age_h = 0
    print("\nLive trades: NONE in last 7d")
print(f"Paper main trades (rt, non-shadow): N={len(main_rows)}")


def dedup_rolling(rows):
    rows = sorted(rows, key=lambda r: r["created_at"])
    last_seen = {}
    out = []
    for r in rows:
        if r.get("kol_group") in SOL_BL:
            continue
        ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        key = (r["strategy"], r["token_address"])
        prev = last_seen.get(key)
        if prev and (ts - prev) < timedelta(hours=24):
            continue
        last_seen[key] = ts
        out.append(r)
    return out


# ---- Step 4: shadow 14d dedup ----
shadow_raw = []
for strat in SOL_STRATS:
    offset = 0
    while True:
        res = sb.table("paper_trades").select(
            "strategy,kol_group,token_address,symbol,pnl_pct,pnl_usd,status,created_at"
        ).eq("chain", "solana").eq("is_shadow", True).eq("strategy", strat).in_("status", EXIT).gte("created_at", SINCE_14D).order("created_at").range(offset, offset + 999).execute()
        b = res.data or []
        shadow_raw.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
shadow = dedup_rolling(shadow_raw)
print(f"Shadow 14d: raw={len(shadow_raw)} -> dedup={len(shadow)}")

POSITION_MAIN = 50  # reference scale for paper main / shadow projection

# ---- Aggregate totals ----
def total_usd_live(rows):
    return sum(float(r.get("pnl_usd") or 0) for r in rows if r["status"] in EXIT)

def total_usd_pct(rows, pos):
    return sum(float(r["pnl_pct"]) for r in rows if r["status"] in EXIT) * pos

live_total = total_usd_live(live_rows)
main_total = total_usd_pct(main_rows, POSITION_MAIN)
shadow_total = sum(float(r["pnl_pct"]) for r in shadow) * POSITION_MAIN / max(N_STRATS, 1)

print(f"\n=== TOTALS (live window) ===")
print(f"Live actual $: {live_total:+.2f} (real position size)")
print(f"Paper main $ @${POSITION_MAIN}: {main_total:+.2f}")
print(f"Shadow 14d proj $ @${POSITION_MAIN}/{N_STRATS}: {shadow_total:+.2f}")

# ---- Per-strategy summary ----
print(f"\n=== PER-STRATEGY (live window, since {SINCE_7D[:10]}) ===")
print(f"{'strategy':<34} {'N_live':>6} {'live_$':>9} {'live_WR':>7} {'N_main':>6} {'main_$':>9} {'main_WR':>7}")
for strat in SOL_STRATS:
    lv = [r for r in live_rows if r["strategy"] == strat and r["status"] in EXIT]
    mn = [r for r in main_rows if r["strategy"] == strat and r["status"] in EXIT]
    lv_usd = sum(float(r.get("pnl_usd") or 0) for r in lv)
    mn_usd = sum(float(r["pnl_pct"]) for r in mn) * POSITION_MAIN
    lv_wr = (sum(1 for r in lv if float(r.get("pnl_usd") or 0) > 0) / len(lv) * 100) if lv else 0
    mn_wr = (sum(1 for r in mn if float(r["pnl_pct"]) > 0) / len(mn) * 100) if mn else 0
    print(f"{strat:<34} {len(lv):>6} {lv_usd:>+9.2f} {lv_wr:>6.0f}% {len(mn):>6} {mn_usd:>+9.2f} {mn_wr:>6.0f}%")

# ---- Step 7: drift paired ----
paired = []
for lr in live_rows:
    if lr["status"] not in EXIT:
        continue
    lt = datetime.fromisoformat(lr["created_at"].replace("Z", "+00:00"))
    for mr in main_rows:
        if mr["strategy"] != lr["strategy"] or mr["token_address"] != lr["token_address"]:
            continue
        if mr["status"] not in EXIT:
            continue
        mt = datetime.fromisoformat(mr["created_at"].replace("Z", "+00:00"))
        if abs((mt - lt).total_seconds()) < 3600:
            paired.append((lr, mr))
            break
if paired:
    drifts = [(float(lr["pnl_pct"]) - float(mr["pnl_pct"])) * 100 for lr, mr in paired]
    mean_drift = statistics.mean(drifts)
    med_drift = statistics.median(drifts)
    live_won = sum(1 for lr, mr in paired if float(lr["pnl_pct"]) > float(mr["pnl_pct"]))
    print(f"\n=== DRIFT live<->paper main (paired) ===")
    print(f"N paired={len(paired)} | mean drift={mean_drift:+.2f}pp | median={med_drift:+.2f}pp | live>paper {live_won}/{len(paired)}")
else:
    mean_drift = 0
    print(f"\n=== DRIFT: 0 paired trades ===")

# ---- Step 6: sigma ----
hist_raw = []
for strat in SOL_STRATS:
    offset = 0
    while True:
        res = sb.table("paper_trades").select(
            "strategy,kol_group,token_address,pnl_pct,status,created_at"
        ).eq("chain", "solana").eq("is_shadow", True).eq("strategy", strat).in_("status", EXIT).gte("created_at", SINCE_25D).order("created_at").range(offset, offset + 999).execute()
        b = res.data or []
        hist_raw.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
hist = dedup_rolling(hist_raw)
agg = defaultdict(float)
for r in hist:
    agg[r["created_at"][:10]] += float(r["pnl_pct"]) * POSITION_MAIN / max(N_STRATS, 1)
max_red = cur = 0
for d in sorted(agg.keys()):
    if agg[d] < 0:
        cur += 1
        max_red = max(max_red, cur)
    else:
        cur = 0
live_agg = defaultdict(float)
for r in live_rows:
    if r["status"] in EXIT:
        live_agg[r["created_at"][:10]] += float(r.get("pnl_usd") or 0)
current_red = 0
for d in sorted(live_agg.keys(), reverse=True):
    if live_agg[d] < 0:
        current_red += 1
    else:
        break
if current_red < max_red:
    sigma = "NORMAL"
elif current_red == max_red:
    sigma = "AT LIMIT"
else:
    sigma = "RECORD BROKEN"
print(f"\n=== SIGMA ===")
print(f"Hist max red streak (25d shadow)={max_red}d | current live red streak={current_red}d | {sigma}")

# ---- Live per-day ----
print(f"\n=== LIVE PER-DAY ===")
for d in sorted(live_agg.keys()):
    nn = sum(1 for r in live_rows if r["created_at"][:10] == d and r["status"] in EXIT)
    print(f"{d}: N={nn} ${live_agg[d]:+.2f}")

# ---- Verdict ----
if sigma == "RECORD BROKEN" or abs(mean_drift) > 15 or current_red >= 3:
    verdict = "ALERT"
elif sigma == "AT LIMIT" or abs(mean_drift) > 8 or current_red == 2:
    verdict = "CAUTION"
else:
    verdict = "GO"
print(f"\n=== VERDICT: {verdict} (age={age_h:.0f}h) ===")
