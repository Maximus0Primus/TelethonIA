"""Ground truth — validate shadow sim matches paper main reality, per strat."""
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("scraper/.env")
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
NOW = datetime.now(timezone.utc)
EXIT = ["tp_hit", "sl_hit", "timeout", "trail_stop", "be_stop"]
POS = 50  # reference scale

cfg = sb.table("scoring_config").select("paper_trade_config,rt_trade_config").eq("id", 1).execute().data[0]
SOL_BL = set(cfg["paper_trade_config"].get("kol_chain_blacklist", {}).get("solana", []))
ALLOCS = cfg["rt_trade_config"].get("hybrid_strategy", {}).get("allocations", {})
SOL_STRATS = [k for k in ALLOCS if not k.startswith(("ETH_", "BSC_", "BASE_"))]
print(f"SOL BL={len(SOL_BL)} | live SOL strats={SOL_STRATS}\n")

SINCE = (NOW - timedelta(days=7)).isoformat().replace("+00:00", "Z")


def pull_all(strat):
    rows = []
    offset = 0
    while True:
        res = sb.table("paper_trades").select(
            "id,source,is_shadow,kol_group,token_address,symbol,status,pnl_pct,pnl_usd,created_at"
        ).eq("chain", "solana").eq("strategy", strat).gte("created_at", SINCE).in_("status", EXIT).order("created_at").range(offset, offset + 999).execute()
        b = res.data or []
        rows.extend(b)
        if len(b) < 1000:
            break
        offset += 1000
    return rows


def dedup(shadow_rows):
    shadow_rows = sorted(shadow_rows, key=lambda r: r["created_at"])
    last_seen = {}
    out = []
    for r in shadow_rows:
        if r["kol_group"] in SOL_BL:
            continue
        ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        key = (r["token_address"],)
        prev = last_seen.get(key)
        if prev and (ts - prev) < timedelta(hours=24):
            continue
        last_seen[key] = ts
        out.append(r)
    return out


for strat in SOL_STRATS:
    rows = pull_all(strat)
    shadow = dedup([r for r in rows if r["is_shadow"]])
    main = [r for r in rows if not r["is_shadow"] and r["source"] == "rt"]
    print(f"{'='*70}\n## {strat}")
    print(f"shadow raw={sum(1 for r in rows if r['is_shadow'])} dedup={len(shadow)} | paper main={len(main)}")
    if not main:
        print("  (no paper main rows — skip reconciliation)\n")
        continue

    days = sorted(set(r["created_at"][:10] for r in rows))
    print(f"\n| day | sim shadow $/d | real main $/d | diff% | status |")
    print(f"|-----|-----|-----|-----|-----|")
    tot_sim = tot_real = 0
    matches = mism = 0
    for d in days:
        sim = sum(float(r["pnl_pct"]) for r in shadow if r["created_at"][:10] == d) * POS
        real = sum(float(r["pnl_pct"]) for r in main if r["created_at"][:10] == d) * POS
        tot_sim += sim
        tot_real += real
        if real == 0 and sim == 0:
            continue
        diff = abs(sim - real) / max(abs(real), 0.01) * 100
        ok = diff < 5
        if real != 0:
            (matches if ok else mism).__class__  # noop
            if ok:
                matches += 1
            else:
                mism += 1
        st = "MATCH" if ok else f"MISMATCH {diff:.0f}%"
        print(f"| {d} | {sim:+.2f} | {real:+.2f} | {diff:.0f}% | {st} |")

    drift = abs(tot_sim - tot_real) / max(abs(tot_real), 0.01) * 100
    print(f"\nAggregate: sim_shadow=${tot_sim:+.2f} | real_main=${tot_real:+.2f} | drift={drift:.0f}%")

    # concentration on shadow
    by_kol = defaultdict(float)
    for r in shadow:
        by_kol[r["kol_group"]] += float(r["pnl_pct"]) * POS
    by_day = defaultdict(float)
    for r in shadow:
        by_day[r["created_at"][:10]] += float(r["pnl_pct"]) * POS
    tot = sum(by_kol.values())
    if by_kol and tot != 0:
        tk = max(by_kol.items(), key=lambda x: abs(x[1]))
        td = max(by_day.items(), key=lambda x: abs(x[1]))
        print(f"Top KOL: {tk[0]} = {100*tk[1]/tot:.0f}% of shadow PnL (${tk[1]:+.2f})")
        print(f"Top day: {td[0]} = {100*td[1]/tot:.0f}% of shadow PnL (${td[1]:+.2f})")
        conc_kol = abs(tk[1]) > 0.3 * abs(tot)
        conc_day = abs(td[1]) > 0.3 * abs(tot)
    else:
        conc_kol = conc_day = False
        print("Concentration: n/a (no shadow PnL)")

    # verdict
    if drift > 15 or conc_kol or conc_day:
        if drift > 15:
            v = f"BLOCK (drift {drift:.0f}%)"
        else:
            v = "CAUTION (concentration >30%)"
    elif drift > 5:
        v = f"CAUTION (drift {drift:.0f}%)"
    else:
        v = "GO"
    print(f"VERDICT: {v}\n")
