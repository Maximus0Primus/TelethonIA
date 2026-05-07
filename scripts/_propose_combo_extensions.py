"""v14e.56f: propose new BE/LOCK combos based on mega-sweep top performers.

This script DOES NOT modify any code or DB. It reads the latest mega-sweep
output (`_mega_sweep_top_robust.csv` or DB `mega_sweep_runs`), identifies
the top N performers, then proposes BE×LOCK combos that don't yet exist
in STRATEGIES.

Output: human-readable report with copy-paste ready Python snippets for
strategies.py. The user reviews and selects which combos to deploy as
new shadows.

Why standalone (not auto-injected):
- Zero runtime risk: nothing is added to STRATEGIES at runtime
- Reviewable: every proposed combo has a justification
- Reversible: if a combo turns out bad, just don't paste it
- Avoids registry bloat (20 combos × N runs = 100s of dead strats)

Usage:
    python scripts/_propose_combo_extensions.py
    python scripts/_propose_combo_extensions.py --top 5 --max-proposals 10
    python scripts/_propose_combo_extensions.py --csv _mega_sweep_top_robust.csv

Output example:
    === PROPOSAL 1 ===
    Base: SLOW4H_TP50_SL30 ($94/d, N=122, robust)
    Suggested combos (4 missing):
      SLOW4H_BE25_TP50_SL30
      SLOW4H_BE25_LOCK10_TP50_SL30
      SLOW4H_BE15_LOCK5_TP50_SL30
      SLOW4H_BE35_LOCK15_TP50_SL30

    Code snippet to paste in strategies.py after _SLOW6H block:
    [paste-ready Python]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scraper"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / "scraper" / ".env")
except ImportError:
    pass


# Default BE × LOCK matrix to explore
BE_VALUES = [None, 15, 25, 35]    # 4 options
LOCK_VALUES = [None, 5, 10, 15]   # 4 options
# Valid combos: BE without LOCK is fine. LOCK requires BE >= LOCK + 5.
# Skip nonsense: LOCK without BE, LOCK >= BE.


def parse_strat_name(name: str) -> dict | None:
    """Parse a strat name like 'SLOW4H_TP50_SL30' or 'BE25_LOCK10_TP80_SL30'.
    Returns dict {family, horizon_min, tp_pct, sl_pct, be_pct, lock_pct} or None.
    """
    horizon_match = re.match(r"^(SLOW4H|SLOW6H|FAST60|FAST45|FAST)_", name)
    horizon_min = None
    family = None
    if horizon_match:
        prefix = horizon_match.group(1)
        family = prefix
        horizon_min = {
            "SLOW4H": 240, "SLOW6H": 360,
            "FAST": 30, "FAST45": 45, "FAST60": 60,
        }.get(prefix)
        rest = name[len(prefix) + 1:]
    else:
        rest = name

    be_match = re.match(r"BE(\d+)_", rest)
    be_pct = int(be_match.group(1)) if be_match else None
    if be_match:
        rest = rest[len(be_match.group(0)):]

    lock_match = re.match(r"LOCK(\d+)_", rest)
    lock_pct = int(lock_match.group(1)) if lock_match else None
    if lock_match:
        rest = rest[len(lock_match.group(0)):]

    tp_match = re.search(r"TP(\d+)", rest)
    sl_match = re.search(r"SL(\d+)", rest)
    if not tp_match or not sl_match:
        return None
    tp_pct = int(tp_match.group(1))
    sl_pct = int(sl_match.group(1))

    return {
        "family": family,
        "horizon_min": horizon_min,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "be_pct": be_pct,
        "lock_pct": lock_pct,
    }


def build_strat_name(parts: dict) -> str:
    """Reconstruct canonical strat name from parsed parts."""
    name_parts = []
    if parts["family"]:
        name_parts.append(parts["family"])
    if parts["be_pct"]:
        name_parts.append(f"BE{parts['be_pct']}")
    if parts["lock_pct"]:
        name_parts.append(f"LOCK{parts['lock_pct']}")
    name_parts.append(f"TP{parts['tp_pct']}")
    name_parts.append(f"SL{parts['sl_pct']}")
    return "_".join(name_parts)


def load_existing_strats() -> set[str]:
    """Read STRATEGIES dict from strategies.py (module-level keys)."""
    try:
        from strategies import STRATEGIES
        return set(STRATEGIES.keys())
    except Exception as e:
        print(f"WARN: failed to import STRATEGIES ({e}), using empty set")
        return set()


def load_top_from_csv(csv_path: Path, limit: int) -> list[dict]:
    """Load top N strats from a mega-sweep CSV (col `strategy`, `pnl_per_day`)."""
    import csv
    rows = []
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            strat = r.get("strategy") or r.get("strat") or r.get("name")
            pnl_d = r.get("pnl_per_day") or r.get("pnl_d") or r.get("d14") or r.get("d8")
            n = r.get("n") or r.get("n_trades")
            if strat and pnl_d:
                try:
                    rows.append({"strategy": strat, "pnl_per_day": float(pnl_d),
                                 "n": int(float(n)) if n else 0})
                except (ValueError, TypeError):
                    continue
    rows.sort(key=lambda x: x["pnl_per_day"], reverse=True)
    return rows[:limit]


def load_top_from_db(limit: int) -> list[dict]:
    """Fallback: query mega_sweep_runs latest run."""
    try:
        from supabase import create_client
        url = os.environ["SUPABASE_URL"]
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
        client = create_client(url, key)
        # Latest run_id
        runs = client.table("mega_sweep_runs").select(
            "run_id, strategy, pnl_per_day, n_sim_trades, run_at"
        ).order("run_at", desc=True).limit(500).execute()
        rows = runs.data or []
        if not rows:
            return []
        latest_run_id = rows[0].get("run_id")
        same_run = [r for r in rows if r.get("run_id") == latest_run_id]
        same_run.sort(key=lambda x: float(x.get("pnl_per_day") or 0), reverse=True)
        return [
            {"strategy": r["strategy"],
             "pnl_per_day": float(r.get("pnl_per_day") or 0),
             "n": int(r.get("n_sim_trades") or 0)}
            for r in same_run[:limit]
        ]
    except Exception as e:
        print(f"DB lookup failed: {e}")
        return []


def propose_combos(base_strat: str, existing: set[str]) -> list[dict]:
    """For a given base strat, propose BE × LOCK extensions that don't exist."""
    parts = parse_strat_name(base_strat)
    if not parts or not parts["horizon_min"]:
        return []

    proposals = []
    for be in BE_VALUES:
        for lock in LOCK_VALUES:
            # Skip nonsense combos
            if lock is not None and be is None:
                continue  # LOCK without BE makes no sense
            if be is not None and lock is not None and lock >= be:
                continue  # LOCK must be lower than BE
            # Build candidate name
            candidate_parts = {
                "family": parts["family"],
                "horizon_min": parts["horizon_min"],
                "tp_pct": parts["tp_pct"],
                "sl_pct": parts["sl_pct"],
                "be_pct": be,
                "lock_pct": lock,
            }
            candidate_name = build_strat_name(candidate_parts)
            if candidate_name == base_strat:
                continue  # same as base
            if candidate_name in existing:
                continue  # already exists
            proposals.append({
                "name": candidate_name,
                "be_pct": be,
                "lock_pct": lock,
                "tp_pct": parts["tp_pct"],
                "sl_pct": parts["sl_pct"],
                "horizon_min": parts["horizon_min"],
            })
    return proposals


def render_python_snippet(prop: dict) -> str:
    """Generate paste-ready Python for strategies.py."""
    name = prop["name"]
    tp_mult = 1 + prop["tp_pct"] / 100
    sl_mult = 1 - prop["sl_pct"] / 100
    horizon = prop["horizon_min"]
    extras = []
    if prop["be_pct"]:
        extras.append(f'"be_activation": {prop["be_pct"] / 100:.2f}')
    if prop["lock_pct"]:
        extras.append(f'"be_lock_pct": {prop["lock_pct"] / 100:.2f}')
    extras_str = (", " + ", ".join(extras)) if extras else ""
    return (
        f'STRATEGIES["{name}"] = [\n'
        f'    {{"pct": 1.0, "tp_mult": {tp_mult:.2f}, "sl_mult": {sl_mult:.2f}, '
        f'"horizon_min": {horizon}{extras_str}, "label": "main"}},\n'
        f']\n'
        f'SHADOW_STRATEGIES.append("{name}")'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5,
                    help="Top N base strats to extend (default: 5)")
    ap.add_argument("--max-proposals", type=int, default=8,
                    help="Max combos to propose per base strat (default: 8)")
    ap.add_argument("--csv", type=str,
                    default=str(REPO_ROOT / "_mega_sweep_top_robust.csv"),
                    help="CSV path (default: _mega_sweep_top_robust.csv at repo root)")
    args = ap.parse_args()

    # Load top from CSV first, fallback to DB
    csv_path = Path(args.csv)
    top = load_top_from_csv(csv_path, args.top)
    if not top:
        print(f"CSV empty or missing, trying DB fallback...")
        top = load_top_from_db(args.top)
    if not top:
        print("ERROR: no top strats found in CSV or DB. Did the mega-sweep run?")
        return 1

    existing = load_existing_strats()
    print(f"Loaded {len(existing)} existing strategies from STRATEGIES dict.\n")

    print(f"=== TOP {args.top} BASE STRATS (by pnl_per_day) ===\n")
    for r in top:
        print(f"  {r['strategy']:45s}  pnl/d={r['pnl_per_day']:.1f}  N={r['n']}")
    print()

    all_proposals = []
    for r in top:
        proposals = propose_combos(r["strategy"], existing)
        proposals = proposals[:args.max_proposals]
        if proposals:
            all_proposals.append((r, proposals))

    if not all_proposals:
        print("No new combos to propose — all variants already exist.")
        return 0

    print(f"=== {sum(len(p) for _, p in all_proposals)} COMBOS PROPOSED ===\n")
    for base, props in all_proposals:
        print(f"--- BASE: {base['strategy']} (pnl/d={base['pnl_per_day']:.1f}, N={base['n']}) ---")
        for prop in props:
            print(f"  → {prop['name']}")
        print()

    print("=== PASTE-READY PYTHON SNIPPETS ===\n")
    print("# Add this block to strategies.py after the existing SLOW6H section.\n")
    print("# v14e.56f: combos auto-proposed from mega-sweep top performers.\n")
    for base, props in all_proposals:
        print(f"# Extensions of {base['strategy']} ($/d {base['pnl_per_day']:.1f})")
        for prop in props:
            print(render_python_snippet(prop))
            print()

    print(f"\n=== SUMMARY ===")
    print(f"Total proposed: {sum(len(p) for _, p in all_proposals)} new shadows")
    print(f"Cost: ~{sum(len(p) for _, p in all_proposals) * 24840:,} extra configs in next sweep")
    print(f"Action: review the snippets, paste the ones you want into strategies.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
