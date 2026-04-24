"""Use token_snapshots outcomes (max_price_24h, did_2x_*) to estimate PnL
of relaxing the 12h age window."""
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

POSITION_USD = 1.70
FEE_RT = 0.002  # 20 bps round-trip (BUY_FEE + SELL_FEE)

LOG_PATH = "./kol_48h.log"


def parse_skips():
    pat = re.compile(
        r"^(\w{3} \d+ \d+:\d+:\d+).*RT SKIP: (\$\S+) — token too old \((\d+)h"
    )
    year = datetime.now(timezone.utc).year
    rows = []
    with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            try:
                ts = datetime.strptime(
                    f"{year} {m.group(1)}", "%Y %b %d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                rows.append((ts, m.group(2), int(m.group(3))))
            except ValueError:
                continue
    # dedup
    seen = set()
    uniq = []
    for ts, sym, age in rows:
        k = (sym, ts.replace(second=0, microsecond=0))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((ts, sym, age))
    return uniq


def main():
    unique = parse_skips()
    print(f"Unique too-old skips parsed: {len(unique)}")

    # Resolve CA via kol_mentions (±30 min window, same symbol)
    resolved = {}
    for ts, sym, age in unique:
        try:
            r = (sb.table("kol_mentions")
                 .select("resolved_ca,chain")
                 .eq("symbol", sym)
                 .gte("message_date", (ts - timedelta(minutes=30)).isoformat())
                 .lte("message_date", (ts + timedelta(minutes=30)).isoformat())
                 .not_.is_("resolved_ca", "null")
                 .limit(1).execute())
            if r.data:
                resolved[(ts, sym, age)] = (r.data[0]["resolved_ca"],
                                            r.data[0].get("chain") or "solana")
        except Exception as e:
            pass
    print(f"CA resolved: {len(resolved)} / {len(unique)}")

    buckets = [
        ("12-24h", 12, 24),
        ("24-48h", 24, 48),
        ("48-72h", 48, 72),
        ("72-168h", 72, 168),
        (">168h", 168, 9999999),
    ]

    agg = defaultdict(lambda: {"N": 0, "N_with_data": 0,
                                "did_2x_6h": 0, "did_2x_24h": 0,
                                "cum_max_ret": 0.0, "details": []})

    sol_ca = [(k, v) for k, v in resolved.items() if v[1] == "solana"]
    print(f"Fetching token_snapshots for {len(sol_ca)} Solana CAs...")

    for (ts, sym, age), (ca, chain) in sol_ca:
        bname = next((n for n, lo, hi in buckets if lo <= age < hi), ">168h")
        agg[bname]["N"] += 1

        # Get first snapshot >= call time within 2h after
        try:
            r = (sb.table("token_snapshots")
                 .select("snapshot_at,price_at_snapshot,max_price_24h,"
                         "price_after_6h,price_after_24h,did_2x_6h,did_2x_24h")
                 .eq("token_address", ca)
                 .gte("snapshot_at", ts.isoformat())
                 .lte("snapshot_at", (ts + timedelta(hours=2)).isoformat())
                 .order("snapshot_at")
                 .limit(1).execute())
        except Exception:
            continue
        if not r.data:
            continue
        s = r.data[0]
        entry = float(s.get("price_at_snapshot") or 0)
        max24 = float(s.get("max_price_24h") or 0)
        after6 = float(s.get("price_after_6h") or 0)
        after24 = float(s.get("price_after_24h") or 0)
        if entry <= 0 or max24 <= 0:
            continue
        max_ret = (max24 / entry - 1) * 100
        agg[bname]["N_with_data"] += 1
        agg[bname]["did_2x_6h"] += int(bool(s.get("did_2x_6h")))
        agg[bname]["did_2x_24h"] += int(bool(s.get("did_2x_24h")))
        agg[bname]["cum_max_ret"] += max_ret
        agg[bname]["details"].append({
            "sym": sym, "age": age, "max_ret": max_ret,
            "entry": entry, "max24": max24,
            "after6": after6, "after24": after24,
            "d2_6": bool(s.get("did_2x_6h")),
            "d2_24": bool(s.get("did_2x_24h")),
        })

    print(
        f"\n{'Age band':<12}{'N':>5}{'N_data':>8}{'cov%':>6}"
        f"{'did_2x_6h':>12}{'did_2x_24h':>12}{'avg max_ret':>14}"
    )
    for bname, _, _ in buckets:
        d = agg[bname]
        if d["N"] == 0:
            continue
        cov = d["N_with_data"] / d["N"] * 100 if d["N"] else 0
        avg = d["cum_max_ret"] / d["N_with_data"] if d["N_with_data"] else 0
        print(f"  {bname:<10}{d['N']:>5}{d['N_with_data']:>8}{cov:>5.0f}%"
              f"{d['did_2x_6h']:>12}{d['did_2x_24h']:>12}{avg:>+12.1f}%")

    # Naive PnL sim per bucket: approximate TP/SL/timeout outcomes via max_ret
    # and "after6h" / "after24h" signals.
    # Rule:
    #   max_ret >= 80% -> TP80 fired (+80% net after fees)
    #   max_ret in [50, 80) -> TP50 would have fired (for TP50 strat)
    #   max_ret < 0 -> SL25 fired (-25%) or SL30 (-30%)
    #   otherwise: timeout at ~max_ret/2 (rough mean-rev proxy, biased optimistic)

    print(f"\n=== Simulated PnL per bucket (TP80_SL25 live strat, ${POSITION_USD}/trade) ===")
    for bname, _, _ in buckets:
        d = agg[bname]
        if d["N_with_data"] == 0:
            continue
        n_tp, n_sl, n_to = 0, 0, 0
        pnl = 0.0
        for det in d["details"]:
            mr = det["max_ret"]
            if mr >= 80:
                pnl += POSITION_USD * (0.80 - FEE_RT)
                n_tp += 1
            elif mr >= -25:
                # Assume timeout at after6h or max_ret/2 (bearish bias)
                # if after6h data is available, use it
                if det["entry"] > 0 and det["after6"] > 0:
                    ret = (det["after6"] / det["entry"] - 1)
                else:
                    ret = mr / 200  # conservative: mean-rev by half
                pnl += POSITION_USD * (ret - FEE_RT)
                n_to += 1
            else:
                pnl += POSITION_USD * (-0.25 - FEE_RT)
                n_sl += 1
        n = n_tp + n_sl + n_to
        print(f"  {bname:<12} N={n:<3} TP={n_tp} SL={n_sl} TO={n_to}  "
              f"PnL=${pnl:+.2f}  avg=${pnl/n if n else 0:+.3f}/trade")

    # 12h -> 48h relaxation impact
    print(f"\n=== Impact de relaxer 12h -> 48h (bands 12-24h + 24-48h) ===")
    relax_pnl = 0.0
    relax_n = 0
    for bname in ("12-24h", "24-48h"):
        d = agg[bname]
        for det in d["details"]:
            mr = det["max_ret"]
            if mr >= 80:
                relax_pnl += POSITION_USD * (0.80 - FEE_RT)
            elif mr >= -25:
                ret = ((det["after6"] / det["entry"] - 1)
                       if det["entry"] > 0 and det["after6"] > 0 else mr / 200)
                relax_pnl += POSITION_USD * (ret - FEE_RT)
            else:
                relax_pnl += POSITION_USD * (-0.25 - FEE_RT)
            relax_n += 1
    print(f"  Added trades: {relax_n}")
    print(f"  PnL total:    ${relax_pnl:+.2f}")
    print(f"  Per 48h:      ${relax_pnl:+.2f}  (~ {relax_pnl*15:+.2f}$/mois extrapolé)")

    # 48h+ relaxation
    print(f"\n=== Impact full relax (12h -> no limit) ===")
    tot_pnl = 0.0
    tot_n = 0
    for bname, _, _ in buckets:
        for det in agg[bname]["details"]:
            mr = det["max_ret"]
            if mr >= 80:
                tot_pnl += POSITION_USD * (0.80 - FEE_RT)
            elif mr >= -25:
                ret = ((det["after6"] / det["entry"] - 1)
                       if det["entry"] > 0 and det["after6"] > 0 else mr / 200)
                tot_pnl += POSITION_USD * (ret - FEE_RT)
            else:
                tot_pnl += POSITION_USD * (-0.25 - FEE_RT)
            tot_n += 1
    print(f"  Added trades: {tot_n}")
    print(f"  PnL total:    ${tot_pnl:+.2f}")


if __name__ == "__main__":
    main()
