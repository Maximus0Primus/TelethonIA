"""v14e.41 — RECALL sweep extended.

Extends mega-sweep methodology to test RECALL entries alongside first_call.
For each candidate exit spec (TP/SL/timeout/BE/LOCK), replays it on three
universes:
  1. first_call    — dedup'd 1st mention per token (= classic mega sweep)
  2. recall_*      — rank≥2 mentions, segmented by drift_vs_first or
                     drift_vs_peak (matches RECALL_DIP* / RECALL_PEAK* filters)
  3. union_*       — first_call ∪ recall_* (bigger N for same exit hypothesis)

Output CSV is compatible with analyze_mega_sweep.py (extra column:
`entry_universe`). Run via cron alongside mega-sweep-48h to track whether
adding recalls to a given exit strategy improves or hurts its EV.

Usage:
  python scripts/_recall_sweep.py --since 2026-04-14 \\
      --output scraper/_recall_sweep.csv

Reuses sim.py replay engine + strategies.py STRATEGIES pool. No new mechanics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Path to import scraper modules
SCRAPER_DIR = Path(__file__).resolve().parent.parent / "scraper"
sys.path.insert(0, str(SCRAPER_DIR))

from sim import _replay_trade_on_ticks, sb_get  # noqa: E402
from strategies import STRATEGIES  # noqa: E402


# ---------------------------------------------------------------------------
# Universe builders
# ---------------------------------------------------------------------------

def fetch_recall_events(since: str, max_age_h: float = 72.0) -> list[dict]:
    """Build recall universe from kol_mentions.

    Returns one row per (token, mention rank≥2) with:
      token_address, message_date (= entry_at), first_at, first_price,
      entry_price (= price at recall), peak_between (post-1st-call ATH),
      hours_since_first, drift_first, drift_peak.

    Drops events with missing prices or where 1st call was <10 min before.
    """
    print(f"Fetching kol_mentions since {since}...")
    rows = sb_get(
        "kol_mentions",
        [
            ("select", "id,resolved_ca,message_date,kol_group,chain"),
            ("message_date", f"gte.{since}"),
            ("resolved_ca", "not.is.null"),
            ("order", "message_date"),
        ],
    )
    print(f"  {len(rows):,} mentions")

    # Group by token, sort, attach rank + first_at
    by_token: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_token[r["resolved_ca"]].append(r)

    recalls: list[dict] = []
    for ca, events in by_token.items():
        events.sort(key=lambda e: e["message_date"])
        if len(events) < 2:
            continue
        first_at = events[0]["message_date"]
        for rank, ev in enumerate(events[1:], start=2):
            t_first = datetime.fromisoformat(first_at.replace("Z", "+00:00"))
            t_now = datetime.fromisoformat(ev["message_date"].replace("Z", "+00:00"))
            hours = (t_now - t_first).total_seconds() / 3600.0
            if hours < (10 / 60):  # gate matches RT detect (600s)
                continue
            if hours > max_age_h:
                continue
            recalls.append(
                {
                    "token_address": ca,
                    "chain": ev.get("chain") or "solana",
                    "kol_group": ev["kol_group"],
                    "rank": rank,
                    "first_at": first_at,
                    "event_at": ev["message_date"],
                    "hours_since_first": hours,
                }
            )
    print(f"  {len(recalls):,} recall events (rank>=2, gap>=10min, gap<={max_age_h}h)")
    return recalls


def fetch_first_calls(since: str) -> list[dict]:
    """First-call universe: paper_trades.source=rt deduped to one row per CA.
    Mirrors sim.py mega sweep universe."""
    print(f"Fetching first_call universe since {since}...")
    rows = sb_get(
        "paper_trades",
        [
            ("select", "id,token_address,created_at,entry_price,rt_liquidity_usd,"
                       "rt_score,kol_group,chain"),
            ("source", "eq.rt"),
            ("is_shadow", "eq.false"),
            ("created_at", f"gte.{since}"),
            ("order", "created_at"),
        ],
    )
    seen: set[str] = set()
    universe: list[dict] = []
    for r in rows:
        if r["token_address"] in seen:
            continue
        seen.add(r["token_address"])
        universe.append(r)
    print(f"  {len(universe):,} first-call tokens")
    return universe


def fetch_ticks_for_tokens(tokens: list[str], since: str, end: str) -> dict[str, list[dict]]:
    """Pull price_ticks for each token in batch. Returns {ca: [ticks]}."""
    print(f"Fetching ticks for {len(tokens):,} tokens...")
    out: dict[str, list[dict]] = {}
    for i, ca in enumerate(tokens):
        rs = sb_get(
            "price_ticks",
            [
                ("select", "price_usd,fetched_at,source,chain"),
                ("token_address", f"eq.{ca}"),
                ("fetched_at", f"gte.{since}"),
                ("fetched_at", f"lte.{end}"),
                ("order", "fetched_at"),
            ],
        )
        if rs:
            out[ca] = rs
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tokens)}", flush=True)
    print(f"  {len(out):,} tokens with ticks")
    return out


def attach_recall_prices(recalls: list[dict], ticks: dict[str, list[dict]]) -> list[dict]:
    """For each recall event, find entry_price (closest tick) +
    first_price (closest tick to first_at) + peak_between (max price first→event).
    Compute drift_first + drift_peak. Drops events without tick data."""
    enriched: list[dict] = []
    for r in recalls:
        token_ticks = ticks.get(r["token_address"])
        if not token_ticks:
            continue
        t_first = datetime.fromisoformat(r["first_at"].replace("Z", "+00:00"))
        t_event = datetime.fromisoformat(r["event_at"].replace("Z", "+00:00"))

        # Closest tick to first_at (within ±15 min)
        first_price = None
        peak_between = 0.0
        event_price = None
        best_first_dt = 999_999
        best_event_dt = 999_999
        for tk in token_ticks:
            tk_t = datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00"))
            tk_p = float(tk["price_usd"]) if tk["price_usd"] else None
            if tk_p is None or tk_p <= 0:
                continue
            d_first = abs((tk_t - t_first).total_seconds())
            if d_first <= 15 * 60 and d_first < best_first_dt:
                first_price = tk_p
                best_first_dt = d_first
            if t_first <= tk_t <= t_event and tk_p > peak_between:
                peak_between = tk_p
            d_event = abs((tk_t - t_event).total_seconds())
            if d_event <= 15 * 60 and d_event < best_event_dt:
                event_price = tk_p
                best_event_dt = d_event

        if event_price is None or first_price is None:
            continue
        r2 = dict(r)
        r2["entry_price"] = event_price
        r2["first_price"] = first_price
        r2["peak_between"] = peak_between if peak_between > 0 else first_price
        r2["drift_first"] = (event_price / first_price) - 1.0
        r2["drift_peak"] = (event_price / r2["peak_between"]) - 1.0 if r2["peak_between"] > 0 else 0
        enriched.append(r2)
    return enriched


def attach_first_call_prices(universe: list[dict], ticks: dict[str, list[dict]]) -> list[dict]:
    """Attach ticks-derived entry_price for first_call universe (uses paper_trades.entry_price
    when present, fallback to closest tick)."""
    out = []
    for r in universe:
        if r.get("entry_price") and float(r["entry_price"]) > 0:
            out.append(r)
            continue
        token_ticks = ticks.get(r["token_address"])
        if not token_ticks:
            continue
        t_event = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        best = None
        best_dt = 999_999
        for tk in token_ticks:
            tk_t = datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00"))
            d = abs((tk_t - t_event).total_seconds())
            if d <= 15 * 60 and d < best_dt:
                best = float(tk["price_usd"])
                best_dt = d
        if best:
            r2 = dict(r)
            r2["entry_price"] = best
            out.append(r2)
    return out


# ---------------------------------------------------------------------------
# Recall bucket classifier
# ---------------------------------------------------------------------------

RECALL_BUCKETS = {
    "recall_dip10":   ("first", -0.30, -0.10, 0.0, 24.0),
    "recall_dip30":   ("first", -0.50, -0.30, 0.0, 24.0),
    "recall_dip50":   ("first", -0.85, -0.50, 0.0, 48.0),
    "recall_peak30":  ("peak",  -0.50, -0.30, 0.0, 6.0),
    "recall_peak50":  ("peak",  -0.85, -0.50, 0.0, 12.0),
    "recall_peak70":  ("peak",  -0.95, -0.70, 0.0, 24.0),
}


def classify_recall(r: dict) -> list[str]:
    """Return all buckets this recall event qualifies for."""
    out: list[str] = []
    for name, (mode, lo, hi, age_lo, age_hi) in RECALL_BUCKETS.items():
        d = r["drift_first"] if mode == "first" else r["drift_peak"]
        if d < lo or d > hi:
            continue
        if r["hours_since_first"] < age_lo or r["hours_since_first"] > age_hi:
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# Strategy spec extractor
# ---------------------------------------------------------------------------

def get_exit_specs(spec_filter: str | None = None) -> list[tuple[str, dict]]:
    """Pick exit specs to test. Filter to single-tranche, no-DTRAIL/DIP/SPLIT,
    no-RECALL (we add recall via universe, not via filter). Subset to ~30 if
    spec_filter not provided."""
    candidates: list[tuple[str, dict]] = []
    EXCLUDE_PREFIXES = (
        "DTRAIL", "PTRAIL", "TRAIL", "SPLIT_", "DIP30_", "DIP_", "BOND_",
        "TD2_", "MCAP_DTRAIL", "RECALL_", "ETH_RECALL_",
    )
    for name, tranches in STRATEGIES.items():
        if any(name.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if len(tranches) != 1:
            continue
        spec = tranches[0]
        if spec.get("tp_mult") is None:  # moonbag etc.
            continue
        candidates.append((name, spec))

    if spec_filter:
        wanted = set(spec_filter.split(","))
        candidates = [(n, s) for n, s in candidates if n in wanted]
    else:
        # Default subset: 30 most-likely-relevant — BE/LOCK/FAST/SLOW + basics
        priority_substrings = [
            "BE25_TP80_SL30", "BE15_TP100_SL50", "BE25_LOCK10",
            "BE30_LOCK10", "BE50_LOCK20",
            "FAST_TP50_SL30", "FAST_TP80_SL25", "FAST_TP100_SL20",
            "FAST_TP40_SL30", "FAST45_TP50", "FAST60_TP50", "FAST60_TP40",
            "SLOW4H_TP100_SL50", "SLOW6H_TP100_SL50", "SLOW6H_TP200_SL50",
            "TP50_SL30", "TP80_SL30", "TP100_SL50", "TP150_SL50", "TP200_SL50",
            "SCALP_TP15_SL15", "SCALP_TP20_SL15",
        ]
        # Filter out score gates (_S30, _S40), liquidity (_NZ), price-source
        # variants (_HYST/_DS/_BOTH/_JUPITER/_MED). These are filter-flavored
        # variants of the same exit math; the core spec without them suffices.
        EXCLUDE_SUFFIXES = ("_S30", "_S40", "_S35", "_NZ", "_HYST", "_DS",
                            "_BOTH", "_JUPITER", "_MED3", "_MCAP", "_NZS30",
                            "_LAZYMED", "_LAZY", "_COMBO")
        seen = set()
        out = []
        for name, spec in candidates:
            if name in seen:
                continue
            if any(s in name for s in EXCLUDE_SUFFIXES):
                continue
            if any(sub in name for sub in priority_substrings):
                out.append((name, spec))
                seen.add(name)
            if len(out) >= 30:
                break
        candidates = out

    print(f"Exit specs: {len(candidates)}")
    return candidates


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def build_fake_trade(event: dict, spec_name: str, spec: dict, idx: int) -> dict:
    """Construct a paper_trades-compatible dict for sim._replay_trade_on_ticks."""
    entry = float(event["entry_price"])
    tp_mult = spec.get("tp_mult")
    sl_mult = spec.get("sl_mult", 0.50)
    return {
        "id": f"sweep_{idx}",
        "token_address": event["token_address"],
        "created_at": event["event_at"] if "event_at" in event else event["created_at"],
        "entry_price": entry,
        "tp_price": entry * tp_mult if tp_mult else None,
        "sl_price": entry * sl_mult,
        "horizon_minutes": spec.get("horizon_min", 120),
        "strategy": spec_name,
        "tranche_label": "main",
        "position_usd": 50.0,
        "high_price_seen": entry,
        "be_activation": spec.get("be_activation"),
        "be_lock_pct": spec.get("be_lock_pct"),
        "tp_decay_end": spec.get("tp_decay_end"),
        "rt_liquidity_usd": event.get("rt_liquidity_usd") or 50_000,
        "dex_spot_price_at_entry": entry,
        "chain": event.get("chain") or "solana",
    }


def aggregate(results: list[dict]) -> dict:
    """Compute n, avg_pnl_pct, sum_pnl_usd, wr, sharpe, daily_pnl_json."""
    if not results:
        return {"n": 0, "avg_pnl_pct": 0.0, "sum_pnl_usd": 0.0, "wr_pct": 0.0,
                "sharpe": 0.0, "daily_pnl_json": "{}"}
    pnls = [r["pnl_pct"] for r in results]
    avg = sum(pnls) / len(pnls)
    wr = 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)
    sum_usd = sum(r.get("pnl_usd", r["pnl_pct"] * 50.0) for r in results)
    # sharpe = avg / std (no risk-free)
    if len(pnls) > 1:
        m = avg
        var = sum((p - m) ** 2 for p in pnls) / (len(pnls) - 1)
        std = var ** 0.5
        sharpe = avg / std if std > 0 else 0.0
    else:
        sharpe = 0.0
    # daily_pnl
    daily: dict[str, list[float]] = defaultdict(list)
    for r in results:
        d = r.get("entry_date") or "unknown"
        daily[d].append(r["pnl_pct"])
    daily_avg = {d: sum(v) / len(v) for d, v in daily.items()}
    return {
        "n": len(results),
        "avg_pnl_pct": round(avg, 6),
        "sum_pnl_usd": round(sum_usd, 2),
        "wr_pct": round(wr, 2),
        "sharpe": round(sharpe, 4),
        "daily_pnl_json": json.dumps(daily_avg),
    }


def run_sweep(specs: list[tuple[str, dict]],
              first_calls: list[dict],
              recalls: list[dict],
              ticks: dict[str, list[dict]]) -> list[dict]:
    """For each spec × universe combination, replay and aggregate."""
    csv_rows: list[dict] = []
    universes: dict[str, list[dict]] = {"first_call": first_calls}
    # bucket recalls
    for bucket in RECALL_BUCKETS:
        universes[bucket] = []
    for r in recalls:
        for b in classify_recall(r):
            universes[b].append(r)
    for k, v in universes.items():
        print(f"  universe[{k}] = {len(v)} events")

    # Union universes — first_call merged with each recall bucket
    union_universes: dict[str, list[dict]] = {}
    for bucket in ("recall_dip30", "recall_dip50", "recall_peak30", "recall_peak50"):
        if not universes.get(bucket):
            continue
        # Convert recall events to first_call-compatible dicts (entry_price + created_at)
        recall_events = [
            {"token_address": e["token_address"],
             "created_at": e["event_at"],
             "entry_price": e["entry_price"],
             "rt_liquidity_usd": 50_000,
             "chain": e.get("chain") or "solana"}
            for e in universes[bucket]
        ]
        union_universes[f"union_first_{bucket}"] = first_calls + recall_events
    universes.update(union_universes)

    print(f"\nReplaying {len(specs)} specs × {len(universes)} universes...")
    for spec_idx, (spec_name, spec) in enumerate(specs, start=1):
        for uni_name, events in universes.items():
            if not events:
                continue
            results = []
            for idx, ev in enumerate(events):
                # For recall buckets, ev has event_at; for first_call ev has created_at
                ev_time_key = "event_at" if "event_at" in ev else "created_at"
                token_ticks = ticks.get(ev["token_address"])
                if not token_ticks:
                    continue
                # Trim to ticks AFTER entry
                fake = build_fake_trade(ev, spec_name, spec, idx)
                # Use timestamp of created_at to derive entry_date
                t = datetime.fromisoformat(fake["created_at"].replace("Z", "+00:00"))
                fake["_entry_date"] = t.date().isoformat()
                fwd_ticks = [
                    tk for tk in token_ticks
                    if datetime.fromisoformat(tk["fetched_at"].replace("Z", "+00:00")) >= t
                ]
                if not fwd_ticks:
                    continue
                res = _replay_trade_on_ticks(fake, fwd_ticks, disable_lazy=True)
                if res:
                    res["entry_date"] = fake["_entry_date"]
                    results.append(res)
            agg = aggregate(results)
            csv_rows.append({
                "strategy": spec_name,
                "entry_universe": uni_name,
                "tp_mult": spec.get("tp_mult"),
                "sl_mult": spec.get("sl_mult", 0.50),
                "horizon_min": spec.get("horizon_min", 120),
                "be_activation": spec.get("be_activation") or "",
                "be_lock_pct": spec.get("be_lock_pct") or "",
                **agg,
            })
        if spec_idx % 5 == 0:
            print(f"  spec {spec_idx}/{len(specs)} done", flush=True)
    return csv_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="ISO date (default: 14 days ago)")
    ap.add_argument("--output", default="scraper/_recall_sweep.csv")
    ap.add_argument("--specs", default=None,
                    help="Comma-separated list of strategy names (default: top 30)")
    ap.add_argument("--max-age-h", type=float, default=72.0,
                    help="Max hours_since_first for recall events (default 72)")
    args = ap.parse_args()

    since = args.since or (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    print(f"Recall sweep from {since}")

    first_calls = fetch_first_calls(since)
    recall_events = fetch_recall_events(since, max_age_h=args.max_age_h)

    all_tokens = list({r["token_address"] for r in first_calls}
                      | {r["token_address"] for r in recall_events})
    ticks = fetch_ticks_for_tokens(all_tokens, since, end)

    first_calls = attach_first_call_prices(first_calls, ticks)
    recall_events = attach_recall_prices(recall_events, ticks)
    print(f"Enriched: {len(first_calls)} first_call + {len(recall_events)} recall events")

    specs = get_exit_specs(args.specs)
    rows = run_sweep(specs, first_calls, recall_events, ticks)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No rows produced.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Quick summary: top 10 by EV per universe
    print("\nTop 5 by EV per universe:")
    by_uni: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_uni[r["entry_universe"]].append(r)
    for uni, items in sorted(by_uni.items()):
        items.sort(key=lambda x: -x["avg_pnl_pct"])
        print(f"\n  [{uni}]  ({len(items)} specs, n_min={min((i['n'] for i in items), default=0)})")
        for it in items[:5]:
            print(f"    {it['strategy']:<40} N={it['n']:<4} EV={it['avg_pnl_pct']:+.3f}  WR={it['wr_pct']:>5.1f}%  $sum={it['sum_pnl_usd']:+.0f}")


if __name__ == "__main__":
    main()
