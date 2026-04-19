"""Full audit of strategy alignment across sim/paper/live.

Detects:
- Strategies defined but never activated (no shadow, no main, no live)
- Orchestration override referencing non-existent strategy
- LAZY_STRATEGIES membership on strat not in hybrid/main
- STRATEGY_FILTERS entry without corresponding STRATEGIES entry
- Live allocations pointing to strat not in hybrid/mains
- Shadows in SHADOW_STRATEGIES but not in STRATEGIES dict
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scraper"))
from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", "scraper", ".env"))
from supabase import create_client
from strategies import STRATEGIES, SHADOW_STRATEGIES, STRATEGY_FILTERS, LAZY_STRATEGIES
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# 1. Live config
r = sb.table("scoring_config").select("rt_trade_config").eq("id",1).execute()
cfg = r.data[0]["rt_trade_config"]
if isinstance(cfg, str): cfg = json.loads(cfg)

hybrid = set(cfg["hybrid_strategy"]["allocations"].keys())
live = set(cfg["live_trading"]["allocations"].keys())
overrides = cfg.get("strategy_overrides", {})

print(f"{'='*78}")
print(f"STRATEGY AUDIT — {len(STRATEGIES)} defined, {len(SHADOW_STRATEGIES)} shadow, {len(hybrid)} main paper, {len(live)} live, {len(overrides)} overrides")
print(f"{'='*78}\n")

# Unique set of shadow-listed
shadow_set = set(SHADOW_STRATEGIES)

# 2. Categorize each STRATEGY
cat_live, cat_main, cat_shadow, cat_orphan = [], [], [], []
for s in sorted(STRATEGIES):
    if s in live:
        cat_live.append(s)
    elif s in hybrid:
        cat_main.append(s)
    elif s in shadow_set:
        cat_shadow.append(s)
    else:
        cat_orphan.append(s)

print(f"[1] STRATEGIES by role")
print(f"  LIVE+main:       {len(cat_live):3d}  {cat_live}")
print(f"  MAIN paper only: {len(cat_main):3d}")
for s in cat_main: print(f"      {s}")
print(f"  SHADOW only:     {len(cat_shadow):3d}  (summary below)")
print(f"  ORPHAN:          {len(cat_orphan):3d}  (defined but never opened)")
for s in cat_orphan[:20]:
    print(f"      {s}")
if len(cat_orphan) > 20: print(f"      ... +{len(cat_orphan)-20} more")

# 3. Alignment issues
print(f"\n[2] ALIGNMENT ISSUES")
issues = []

# (a) override references unknown strat
for s in overrides:
    if s not in STRATEGIES:
        issues.append(f"  override {s!r} → strat not in STRATEGIES dict")

# (b) STRATEGY_FILTERS entry without STRATEGIES
for s in STRATEGY_FILTERS:
    if s not in STRATEGIES:
        issues.append(f"  filter {s!r} → strat not in STRATEGIES")

# (c) LAZY_STRATEGIES entries not in hybrid/live
for s in LAZY_STRATEGIES:
    if s not in STRATEGIES:
        issues.append(f"  LAZY {s!r} → strat not in STRATEGIES")
    elif s not in hybrid and s not in live:
        issues.append(f"  LAZY {s!r} → shadow/orphan (LAZY has no effect on position_usd=0)")

# (d) SHADOW_STRATEGIES entry without STRATEGIES
for s in shadow_set:
    if s not in STRATEGIES:
        issues.append(f"  SHADOW {s!r} → strat not in STRATEGIES")

# (e) Live allocation without orch override
for s in live:
    if s not in overrides:
        issues.append(f"  LIVE {s!r} → no orch override (uses defaults polling_sec=30, source=jupiter)")

# (f) Hybrid main without orch override
for s in hybrid:
    if s not in overrides:
        issues.append(f"  MAIN {s!r} → no orch override (uses defaults)")

# (g) Overrides with deprecated price_source (legacy enum not split)
for s, ov in overrides.items():
    if "price_source" in ov and "source" not in ov:
        # legacy style — still works but flagged
        pass  # not an issue, just info

if issues:
    print(f"  {len(issues)} issues found:")
    for i in issues: print(i)
else:
    print("  None — all alignments OK.")

# 4. Newly added v144 shadows — verify orch is set
print(f"\n[3] v144 NEW SHADOWS — orch status")
v144_new = [
    "FAST_TP50_SL30_NOLAZY", "FAST_TP80_SL25_NOLAZY", "FAST_TP40_SL30_NOLAZY", "TP50_SL15_NOLAZY",
    "BE25_TP80_SL30_S30", "BE25_TP80_SL30_S40", "FAST_TP80_SL25_S40",
    "FAST_TP50_SL30_S40", "FAST_TP100_SL20_S40",
    "FAST_TP50_SL30_LAZYSLOW", "FAST_TP80_SL25_LAZYSLOW", "BE25_TP80_SL30_LAZYSLOW",
    "FAST_TP50_SL30_LAZYFAST", "FAST_TP50_SL30_LAZYMED", "FAST_TP50_SL30_LAZYXSLOW",
    "FAST_TP50_SL30_BOTH", "FAST_TP50_SL30_JUPITER",
    "BE25_TP80_SL30_BOTH", "BE25_TP80_SL30_JUPITER",
    "FAST_TP80_SL25_BOTH", "FAST_TP100_SL20_BOTH",
    "FAST_TP50_SL30_MCAP_S40",
    "FAST_TP100_SL20_COMBO", "BE25_TP80_SL30_COMBO",
    "FAST_TP80_SL25_COMBO", "FAST_TP50_SL30_COMBO",
]
print(f"  {'Shadow':<32} STR SHDW ORCH FILTER")
missing_orch = []
for s in v144_new:
    in_str = s in STRATEGIES
    in_shd = s in shadow_set
    ov = overrides.get(s)
    flt = STRATEGY_FILTERS.get(s)
    ok_str = "Y" if in_str else "N"
    ok_shd = "Y" if in_shd else "N"
    ok_ov = "Y" if ov else "-"
    ok_flt = "Y" if flt else "-"
    print(f"  {s:<32} {ok_str:>3} {ok_shd:>4} {ok_ov:>4} {ok_flt:>6}")
    if not ov and s not in live and s not in hybrid:
        missing_orch.append(s)

if missing_orch:
    print(f"\n  ! {len(missing_orch)} v144 shadows without explicit orch override (will use _DEFAULT_ORCH):")
    for s in missing_orch: print(f"    {s}")

# 5. Live vs hybrid check
print(f"\n[4] LIVE ↔ MAIN consistency")
for s in live:
    if s not in hybrid:
        print(f"  WARN  {s} in LIVE but NOT in hybrid main paper")
    else:
        print(f"  OK    {s}: LIVE alloc={cfg['live_trading']['allocations'][s]}, hybrid weight={cfg['hybrid_strategy']['allocations'][s]}")

# 6. Counts summary
print(f"\n[5] Summary counts")
print(f"  STRATEGIES defined   : {len(STRATEGIES)}")
print(f"  SHADOW_STRATEGIES    : {len(SHADOW_STRATEGIES)}")
print(f"  Hybrid allocations   : {len(hybrid)}")
print(f"  Live allocations     : {len(live)}")
print(f"  Strategy overrides   : {len(overrides)}")
print(f"  LAZY_STRATEGIES      : {len(LAZY_STRATEGIES)}")
print(f"  STRATEGY_FILTERS     : {len(STRATEGY_FILTERS)}")
