"""
v142 REVERT — restore original orchestration for the 18 existing mains.
Keep the v142 configs for the 3 NEW mains + 9 NEW shadows.

Mistake: _align_orchestration_to_sim.py applied sim-best config to ALL existing
strats, overwriting the A/B test structure (base strat vs _HYST variant). This
made e.g. FAST_TP100_SL20 identical to FAST_TP100_SL20_HYST (both hysteresis),
destroying the experiment.

Fix: restore pre-v142-align overrides for the 18 existing mains. Leave the
11 new strat overrides (3 mains + 8 shadows, including MCAP_MID which is both)
untouched.

Also revert LAZY_STRATEGIES additions for BE25_TP80_SL30 + filtered variants
(these were NOT originally LAZY; their _HYST counterparts were).
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

SCRAPER_DIR = Path(__file__).resolve().parent
load_dotenv(SCRAPER_DIR / ".env")

# ORIGINAL strategy_overrides values (snapshot from dry-run before v142 align).
# These are the pre-v142-align configs for the 18 existing mains (A/B test state).
ORIGINAL_OVERRIDES = {
    "TP50_SL15":                    {"polling_sec": 30,  "price_source": "jupiter"},
    "BE25_TP80_SL30":               {"polling_sec": 240, "price_source": "median_5"},
    "FAST_TP40_SL30":               {"polling_sec": 30,  "price_source": "hysteresis"},
    "FAST_TP50_SL30":               {"polling_sec": 30,  "price_source": "median_3"},
    "FAST_TP80_SL25":               {"polling_sec": 30,  "price_source": "ds"},
    "BE15_TP100_SL50":              {"polling_sec": 30,  "price_source": "ds"},
    "FAST_TP100_SL20":              {"polling_sec": 30,  "price_source": "ds"},
    "BE15_TP70_SL50_NZ":            {"polling_sec": 240, "price_source": "jupiter"},
    "BE25_TP80_SL30_DS":            {"polling_sec": 30,  "price_source": "ds"},
    "BE25_TP80_SL30_HYST":          {"polling_sec": 30,  "price_source": "hysteresis"},
    "FAST_TP50_SL30_HYST":          {"polling_sec": 30,  "price_source": "hysteresis"},
    "FAST_TP80_SL25_HYST":          {"polling_sec": 30,  "price_source": "hysteresis"},
    "BE15_TP300_SL50_MCAP":         {"polling_sec": 30,  "price_source": "ds"},
    "FAST_TP100_SL20_HYST":         {"polling_sec": 30,  "price_source": "hysteresis"},
    "HIGHSCORE_TP200_SL40":         {"polling_sec": 120, "price_source": "jupiter"},
    "NOZEROLIQ_TP200_SL40":         {"polling_sec": 120, "price_source": "jupiter"},
    "BE25_TP80_SL30_S30_HYST":      {"polling_sec": 240, "price_source": "hysteresis"},
    "BE25_TP80_SL30_NZS30_HYST":    {"polling_sec": 240, "price_source": "hysteresis"},
}

# Keep these (added in v142, never had prior config) — do NOT touch:
KEEP_V142 = {
    # 3 new mains
    "FAST_TP70_SL50", "BE15_TP200_SL40_4H", "MCAP_MID_DTRAIL5_ACT25_SL50_2H",
    # 9 new shadows
    "TD2_BE5_TP120_SL44_T25", "PTRAIL_V2_T10-18-30-45_SL30_T60",
    "BOND_FAST_TP50_SL20_T20", "SCORE40_FAST_TP50_SL30_30M",
    "FAST_TP200_SL40_60M", "DIP30_B10_T10_A20_SL60_120m",
    "BE15_TP150_SL40_2H", "FAST_TP500_SL40_60M",
}

# LAZY additions from v142 align that conflict with A/B test — remove these
# so they're NOT LAZY anymore (revert to their original "not in LAZY" state).
LAZY_TO_REMOVE = {
    "BE25_TP80_SL30",
    "BE25_TP80_SL30_S30_HYST",
    "BE25_TP80_SL30_NZS30_HYST",
}
# KEEP in LAZY (new strats — never had a prior state to revert to):
# FAST_TP70_SL50, PTRAIL_V2_T10-18-30-45_SL30_T60, SCORE40_FAST_TP50_SL30_30M,
# FAST_TP500_SL40_60M

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
c = create_client(url, key)

# ---- STEP 1: revert rt_trade_config.strategy_overrides ----
resp = c.table("scoring_config").select("rt_trade_config").eq("id", 1).execute()
rt_cfg = resp.data[0]["rt_trade_config"]
if isinstance(rt_cfg, str):
    rt_cfg = json.loads(rt_cfg)

cur = dict(rt_cfg.get("strategy_overrides") or {})
print(f"[BEFORE] strategy_overrides: {len(cur)} entries")

new_ov = {}
reverted = []
kept_v142 = []
for strat, cfg in cur.items():
    if strat in ORIGINAL_OVERRIDES:
        orig = ORIGINAL_OVERRIDES[strat]
        if cfg != orig:
            reverted.append((strat, cfg, orig))
        new_ov[strat] = orig
    elif strat in KEEP_V142:
        new_ov[strat] = cfg
        kept_v142.append(strat)
    else:
        # Unknown strat override — keep as-is
        new_ov[strat] = cfg

print(f"\n[REVERT] {len(reverted)} strats being reverted:")
for s, was, now in reverted:
    print(f"  {s:<42} {was} -> {now}")
print(f"\n[KEEP v142] {len(kept_v142)} new strats untouched:")
for s in kept_v142:
    print(f"  {s:<42} {new_ov[s]}")

rt_cfg["strategy_overrides"] = new_ov
c.table("scoring_config").update({"rt_trade_config": rt_cfg}).eq("id", 1).execute()
print(f"\n[OK] Supabase updated: {len(new_ov)} overrides ({len(reverted)} reverted, {len(kept_v142)} new-kept)")

# ---- STEP 2: revert LAZY_STRATEGIES set in strategies.py ----
STRAT_PY = SCRAPER_DIR / "strategies.py"
text = STRAT_PY.read_text(encoding="utf-8")
changed = False
for name in LAZY_TO_REMOVE:
    needle = f'    "{name}",'
    if needle in text:
        text = text.replace(needle + "\n", "")
        print(f"[LAZY REMOVE] {name}")
        changed = True
if changed:
    STRAT_PY.write_text(text, encoding="utf-8")
    print(f"[OK] strategies.py rewritten")
else:
    print("[LAZY] nothing to remove (already absent)")
