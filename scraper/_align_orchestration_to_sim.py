"""
v142 (Apr 18) — Align production orchestration (polling + price_source) with
mega sweep optimal config per strategy.

Reads scraper/_mega_sweep_v142.csv, extracts the top $/day config for each
target strategy, and applies:
  (1) LAZY_STRATEGIES additions in strategies.py   — for polling_mode='lazy'
  (2) rt_trade_config.strategy_overrides JSONB     — for polling_sec +
      price_source (merged with existing, no reset)

Safe to re-run. Dry-run by default: pass --apply to write.

Maps:
  polling_mode 'fast'         -> polling_sec=30,  NOT in LAZY_STRATEGIES
  polling_mode 'static_60'    -> polling_sec=60
  polling_mode 'static_120'   -> polling_sec=120
  polling_mode 'static_240'   -> polling_sec=240
  polling_mode 'lazy'         -> polling_sec=30 + add to LAZY_STRATEGIES

  smoothing 'raw'         -> price_source='jupiter' (or 'ds' if source=dexscreener)
  smoothing 'hysteresis'  -> price_source='hysteresis'
  smoothing 'median_3/5'  -> price_source='median_3/5'
  smoothing 'winsor_p95'  -> price_source='winsor_p95'
  smoothing 'dual_confirm'-> price_source='dual_confirm'
  smoothing 'ema_fast/slow'-> price_source='ema_fast/slow'
"""
import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from supabase import create_client

SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")

CSV_PATH = SCRAPER_DIR / "_mega_sweep_v142.csv"
STRATEGIES_PY = SCRAPER_DIR / "strategies.py"

# Active mains + v142 shadows — scope for orchestration config
TARGET_STRATS = [
    # Active mains (21)
    "BE25_TP80_SL30", "BE25_TP80_SL30_DS",
    "FAST_TP100_SL20", "FAST_TP80_SL25", "FAST_TP50_SL30", "FAST_TP40_SL30",
    "TP50_SL15", "BE15_TP100_SL50",
    "NOZEROLIQ_TP200_SL40", "HIGHSCORE_TP200_SL40",
    "FAST_TP100_SL20_HYST", "FAST_TP80_SL25_HYST", "BE25_TP80_SL30_HYST",
    "FAST_TP50_SL30_HYST", "BE25_TP80_SL30_S30_HYST",
    "BE15_TP70_SL50_NZ", "BE25_TP80_SL30_NZS30_HYST", "BE15_TP300_SL50_MCAP",
    "FAST_TP70_SL50", "BE15_TP200_SL40_4H", "MCAP_MID_DTRAIL5_ACT25_SL50_2H",
    # v142 shadows (6 new + 3 v2)
    "TD2_BE5_TP120_SL44_T25",
    "PTRAIL_V2_T10-18-30-45_SL30_T60",
    "BOND_FAST_TP50_SL20_T20",
    "SCORE40_FAST_TP50_SL30_30M",
    "FAST_TP200_SL40_60M",
    "DIP30_B10_T10_A20_SL60_120m",
    "BE15_TP150_SL40_2H",
    "FAST_TP500_SL40_60M",
]

POLLING_MAP = {
    "fast": 30,
    "static_60": 60,
    "static_120": 120,
    "static_240": 240,
    "lazy": 30,   # LAZY handled via LAZY_STRATEGIES set, need fine polling_sec
}

# smoothing identity: the sim's smoothing name maps directly to price_source
# in the production _decision_price dispatcher (already supports these).
SMOOTH_TO_SRC = {
    "raw": None,  # raw uses pure source (jupiter or ds)
    "hysteresis": "hysteresis",
    "median_3": "median_3",
    "median_5": "median_5",
    "winsor_p95": "winsor_p95",
    "dual_confirm": "dual_confirm",
    "ema_fast": "ema_fast",
    "ema_slow": "ema_slow",
}

# Manual defaults for strats added AFTER the mega sweep (not in CSV).
# Extrapolated from similar-family sweep winners.
MANUAL_DEFAULTS = {
    # MCAP_MID + trail family: best sweep config for MCAP_MID filter + DTRAIL5
    # was median_5 + static_120 + jupiter
    "MCAP_MID_DTRAIL5_ACT25_SL50_2H": {
        "price_source": "median_5", "polling_sec": 120, "needs_lazy": False,
    },
    # SCORE40 + FAST: sweep best per SCORE40 filter = FAST_TP50_SL30 + median_3
    # + lazy + jupiter (+34.5% avg on N=18)
    "SCORE40_FAST_TP50_SL30_30M": {
        "price_source": "median_3", "polling_sec": 30, "needs_lazy": True,
    },
    # Moonshot 60min FAST: similar to BE15_TP200_SL40_4H = hysteresis + static_60 + ds
    "FAST_TP200_SL40_60M": {
        "price_source": "hysteresis", "polling_sec": 60, "needs_lazy": False,
    },
    # DIP variant: existing DIP strats in prod use 30s + jupiter default
    "DIP30_B10_T10_A20_SL60_120m": {
        "price_source": "jupiter", "polling_sec": 30, "needs_lazy": False,
    },
    # BE medium horizon: mirror BE15_TP200_SL40_4H
    "BE15_TP150_SL40_2H": {
        "price_source": "hysteresis", "polling_sec": 60, "needs_lazy": False,
    },
    # Moonshot TP500: patient strategy, use lazy like FAST_TP70_SL50
    "FAST_TP500_SL40_60M": {
        "price_source": "winsor_p95", "polling_sec": 30, "needs_lazy": True,
    },
}


def pick_best_per_strat(df, targets):
    """For each target strat, find the highest $/day row (any filter, source,
    smoothing, polling). Returns dict strat -> best_row."""
    best = {}
    for s in targets:
        sub = df[df["strategy"] == s]
        if sub.empty:
            best[s] = None
            continue
        row = sub.sort_values("dollars_per_day", ascending=False).iloc[0]
        best[s] = row
    return best


def build_config(row):
    """Translate a sweep row to (price_source, polling_sec, needs_lazy)."""
    polling_mode = row["polling_mode"]
    smoothing = row["smoothing"]
    source = row["source"]  # "jupiter" or "dexscreener"

    polling_sec = POLLING_MAP.get(polling_mode, 30)
    needs_lazy = (polling_mode == "lazy")

    # price_source: if smoothing != raw, use smoothing name directly.
    # Else use the source (jupiter/ds — map dexscreener->ds).
    smooth_src = SMOOTH_TO_SRC.get(smoothing)
    if smooth_src is not None:
        price_source = smooth_src
    else:
        price_source = "ds" if source == "dexscreener" else "jupiter"

    return price_source, polling_sec, needs_lazy


def update_lazy_strategies_file(to_add, dry_run):
    """Add strategy names to the LAZY_STRATEGIES set in strategies.py.
    Matches the existing multi-line set literal. Idempotent."""
    content = STRATEGIES_PY.read_text(encoding="utf-8")
    # Locate the LAZY_STRATEGIES block
    anchor = "LAZY_STRATEGIES: set[str] = {"
    idx = content.find(anchor)
    if idx < 0:
        raise SystemExit("LAZY_STRATEGIES anchor not found in strategies.py")
    end_idx = content.find("}", idx)
    block = content[idx:end_idx + 1]
    # Extract existing strat names (quoted)
    import re as _re
    existing = set(_re.findall(r'"([^"]+)"', block))
    new_add = [s for s in to_add if s not in existing]
    if not new_add:
        print("[LAZY] no new additions, all already present")
        return False

    print(f"[LAZY] to add ({len(new_add)}): {new_add}")
    if dry_run:
        return True

    # Insert before the closing brace. Use simple append line.
    insert_line = "\n    # v142 (auto-aligned to mega sweep optimal):\n"
    for name in new_add:
        insert_line += f'    "{name}",\n'
    new_content = content[:end_idx] + insert_line + content[end_idx:]
    STRATEGIES_PY.write_text(new_content, encoding="utf-8")
    print(f"[LAZY] wrote {len(new_add)} additions to {STRATEGIES_PY}")
    return True


def update_strategy_overrides(overrides_to_apply, dry_run):
    """Read rt_trade_config, merge new overrides (per-strategy, preserving any
    existing keys), write back. Idempotent."""
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    c = create_client(url, key)

    resp = c.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
    if not resp.data:
        raise SystemExit("scoring_config id=1 not found")
    rt_cfg = resp.data[0]["rt_trade_config"]
    if isinstance(rt_cfg, str):
        rt_cfg = json.loads(rt_cfg)

    existing_ov = dict(rt_cfg.get("strategy_overrides") or {})
    print(f"[OV] existing strategy_overrides keys: {len(existing_ov)}")
    for s, ocfg in existing_ov.items():
        print(f"     {s:<42} {ocfg}")

    changed = []
    for strat, new_cfg in overrides_to_apply.items():
        current = dict(existing_ov.get(strat) or {})
        merged = dict(current)
        merged.update(new_cfg)  # new takes precedence
        if merged != current:
            existing_ov[strat] = merged
            changed.append((strat, current or "NEW", merged))

    if not changed:
        print("[OV] no changes, all already aligned")
        return False

    print(f"\n[OV] {len(changed)} strategies to update:")
    for s, old, new in changed:
        print(f"     {s:<42} {old}  =>  {new}")

    if dry_run:
        return True

    rt_cfg["strategy_overrides"] = existing_ov
    c.table("scoring_config").update({"rt_trade_config": rt_cfg}).eq("id", 1).execute()
    print(f"[OV] wrote {len(changed)} overrides to Supabase")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run)")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        print("=== DRY RUN (pass --apply to write) ===\n")
    else:
        print("=== APPLY MODE — writing changes ===\n")

    if not CSV_PATH.exists():
        raise SystemExit(f"Mega sweep CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} sweep rows from {CSV_PATH.name}\n")

    best = pick_best_per_strat(df, TARGET_STRATS)

    # Build the per-strat configs
    to_lazy = []
    overrides = {}
    missing = []
    print(f"{'Strategy':<44} {'polling':<10} {'smoothing':<13} {'src':<12} | {'polling_sec':>12} {'price_source':<14} lazy?")
    print("-" * 130)
    for s in TARGET_STRATS:
        row = best.get(s)
        if row is None:
            md = MANUAL_DEFAULTS.get(s)
            if md is None:
                missing.append(s)
                print(f"{s:<44} [NOT FOUND — no manual default]")
                continue
            overrides[s] = {"polling_sec": int(md["polling_sec"]), "price_source": md["price_source"]}
            if md["needs_lazy"]:
                to_lazy.append(s)
            print(f"{s:<44} [manual-default] {'':<12} {'':<12} | "
                  f"{md['polling_sec']:>12} {md['price_source']:<14} {'YES' if md['needs_lazy'] else ''}")
            continue
        ps, poll_sec, needs_lazy = build_config(row)
        overrides[s] = {"polling_sec": int(poll_sec), "price_source": ps}
        if needs_lazy:
            to_lazy.append(s)
        print(f"{s:<44} {row['polling_mode']:<10} {row['smoothing']:<13} {row['source']:<12} | "
              f"{poll_sec:>12} {ps:<14} {'YES' if needs_lazy else ''}")

    if missing:
        print(f"\nWARNING: {len(missing)} target strats not found in sweep CSV: {missing}")

    print(f"\nSummary: {len(overrides)} overrides, {len(to_lazy)} LAZY additions")

    # Apply
    update_lazy_strategies_file(to_lazy, dry_run)
    print()
    update_strategy_overrides(overrides, dry_run)

    if dry_run:
        print("\n[DRY RUN] No changes written. Re-run with --apply to persist.")


if __name__ == "__main__":
    main()
