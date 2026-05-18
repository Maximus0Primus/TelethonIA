# Dedup Rules — Canonical Reference

## Why this doc exists

Over multiple sessions (Feb 14, Feb 15, May 18 2026), we kept re-discovering the same five mistakes
about deduplication of paper trades / snapshots / labels. Each mistake produced inflated metrics,
wrong correlations, or fake "winning strategies" that lost money in paper main.

This doc is the **single source of truth** for dedup rules. Every script touching shadow data,
ML training, KOL stats, or strategy ranking MUST follow them.

`tasks/lessons.md` now points here for the dedup-specific lessons (the timestamp of original
discovery is kept inline).

---

## The 5 rules

### Rule 1 — ML training data must be deduplicated to one snapshot per token

**Discovered:** 2026-02-14

**Why:** `train_model.py` once loaded ALL snapshots (470 for 12h horizon) but many were the same
token appearing 3-7× across cycles. Same token = same outcome = correlated observations. The 470
"snapshots" were really 69 unique tokens. The 6.2% winner rate was actually 11.6%. Feature
correlations were inflated.

**How:** Sort by `snapshot_at`, keep first row per `token_address` (fallback `symbol`), filter
zombies >48h. Apply at both `auto_train()` and CLI entry points.

```python
def deduplicate_snapshots(rows):
    rows.sort(key=lambda r: r["snapshot_at"])
    seen = set()
    out = []
    for r in rows:
        key = r.get("token_address") or r.get("symbol")
        if not key or key in seen: continue
        seen.add(key)
        out.append(r)
    return out
```

```sql
-- SQL equivalent
SELECT DISTINCT ON (token_address) *
FROM token_snapshots
ORDER BY token_address, snapshot_at;
```

---

### Rule 2 — Feature correlations on duplicate data are unreliable

**Discovered:** 2026-02-14

**Why:** `price_action_score` showed +0.252 correlation with `did_2x_12h` on raw data (470
snapshots). After deduplication (69 unique tokens), it collapsed to +0.041 — practically noise.
Yet it had 55% weight in scoring. The duplicate tokens amplified PA's apparent correlation because
similar tokens had similar PA scores.

**How:** All correlation analysis (Pearson, Spearman, mutual info, SHAP) MUST be done on
deduplicated data per Rule 1. Post-dedup rankings: `risk_count +0.335` (#1), `entry_premium -0.180`,
`age -0.149`, `mentions -0.131`, `PA +0.041` (noise).

```python
df = deduplicate_snapshots(df)  # FIRST
corr = df.select_dtypes(include="number").corr()["did_2x_12h"]  # THEN
```

---

### Rule 3 — NEVER dedup by symbol — always use token_address

**Discovered:** 2026-02-15

**Why:** Across 7 files, deduplication used `symbol` (ticker like `$LUNA`) instead of
`token_address` (contract address). The same ticker can map to 3+ different contracts — `$LUNA`
had 3, `$ROCK` had 3, `$WIF` had 3. This caused:
- `auto_backtest.py` merged different tokens' outcomes (12 functions × 13 instances)
- `kol_scorer.py` collapsed 45 real token-KOL pairs into 32
- `backtester.ts` inflated hit rates
- snapshots route dropped tokens from API responses

**How:** Replace ALL `drop_duplicates(subset=["symbol"])` with `subset=["token_address"]`, all
`seen_symbols` sets with address-keyed sets, all `DISTINCT ON (kol, symbol)` with
`DISTINCT ON (kol, token_address)` in RPC. Always fall back to symbol when `token_address` is null
(rare edge case).

```python
# ❌ NEVER
key = r["symbol"]  # display name, NOT unique

# ✅ ALWAYS
key = r.get("token_address") or r.get("symbol")
```

When writing new code that touches token identity, grep for `symbol` in dedup/groupby/set contexts
and flag it.

---

### Rule 4 — Shadow data MUST be deduped before combo / ranking

**Discovered:** 2026-05-18

**Why:** Shadow trades fire on EVERY RT event with a matching strategy, regardless of an existing
open position. Paper main, by contrast, blocks re-entry while a position is open. Consequently,
shadow rows have 1.5-3× more duplicate (strategy, token) pairs than paper main.

If you rank strategies on raw shadow PnL, the strategies that get more re-entries on pumping tokens
look 2-3× better than they actually would be in paper main. This makes `/best-combo` and
`/strategy` recommend strategies that LOSE money once promoted to paper main.

**How:** Before any aggregation / ranking on shadow, apply Rule 5 (rolling-24h). Validation: sum
of dedup shadow PnL on a given day should be within ~5pp of paper main PnL for the same
(strategy, day) — see "Validation rule" below.

---

### Rule 5 — Dedup MUST be rolling 24h on timestamps, not calendar day

**Discovered:** 2026-05-18

**Why:** Calendar-day dedup (`r["created_at"][:10]` as the dedup window) leaks re-entries across
the midnight boundary. A token called at 23:50 UTC on day A and again at 00:10 UTC on day B counts
as TWO trades (because day strings differ) even though they're 20 minutes apart — a clear
re-entry.

This was found on May 18 2026 during the dual-pick deployment audit. After fixing to rolling 24h,
shadow PnL aligned within 5pp of paper main on the validation day.

**How:** Sort all rows by timestamp ASC, keep a `last_seen[(strategy, token_address)]` dict.
For each row: if `(now - last_seen)` < 24h → drop, else → keep + update `last_seen`.

```python
# Canonical implementation — copy this exactly
from datetime import datetime, timedelta

all_rows.sort(key=lambda r: r["created_at"])
last_seen = {}
allowed = []
for r in all_rows:
    if r.get("kol_group") in SOL_BLACKLIST:
        continue
    ts = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
    key = (r["strategy"], r.get("token_address"))
    prev = last_seen.get(key)
    if prev and (ts - prev) < timedelta(hours=24):
        continue
    last_seen[key] = ts
    allowed.append(r)
```

```sql
-- SQL equivalent using NOT EXISTS lookback
SELECT s.*
FROM paper_trades s
WHERE s.is_shadow = true
  AND NOT EXISTS (
    SELECT 1 FROM paper_trades p
    WHERE p.is_shadow = true
      AND p.strategy = s.strategy
      AND p.token_address = s.token_address
      AND p.created_at < s.created_at
      AND p.created_at >= s.created_at - INTERVAL '24 hours'
  );
```

⚠️ **NEVER use** `created_at[:10]`, `date_trunc('day', created_at)`, or any
"group by calendar day" pattern for dedup. They all leak across midnight.

---

## Validation rule (run on every analysis)

Pick the most recent day where BOTH shadow and paper main fired for the same strategy.
Sum `pnl_pct` for shadow (with rolling-24h dedup applied) and paper main.

```python
# For each strategy:
shadow_d = sum(float(r["pnl_pct"]) for r in shadow_dedup if r["created_at"][:10] == d)
main_d = sum(float(r["pnl_pct"]) for r in main_rows if r["created_at"][:10] == d)
ratio = shadow_d / main_d
```

**Expected:**
- If only 1 strategy active: ratio ≈ 1.0 (paper main blocks re-entry; shadow with dedup matches)
- With N strats active: shadow ≈ paper main × N_STRATS_share, but never more than ~3× main

**Abort criterion:** if `abs(shadow_d) > abs(main_d) * 5` → dedup is broken or there's a re-entry
bug. Do NOT trust the analysis. Investigate before continuing.

---

## When each rule applies

| Context | R1 (1/token) | R2 (corr on dedup) | R3 (address not symbol) | R4 (shadow dedup) | R5 (rolling 24h) |
|---------|:------------:|:------------------:|:-----------------------:|:-----------------:|:----------------:|
| ML training (train_model.py) | ✓ | ✓ | ✓ | n/a | n/a |
| Feature correlation analysis | ✓ | ✓ | ✓ | n/a | n/a |
| Backtest (auto_backtest.py) | ✓ | ✓ | ✓ | n/a | n/a |
| KOL leaderboard / KOL stats | n/a | n/a | ✓ | n/a | ✓ |
| Strategy ranking (/best-combo, /strategy) | n/a | n/a | ✓ | ✓ | ✓ |
| Shadow projection (live-perf, simulate) | n/a | n/a | ✓ | ✓ | ✓ |
| Per-day P&L breakdown | n/a | n/a | ✓ | ✓ | ✓ |
| Ground truth validation | n/a | n/a | ✓ | ✓ | ✓ |
| Paper main aggregation | n/a | n/a | ✓ | ✗ (already deduped) | ✗ |
| Live trades aggregation | n/a | n/a | ✓ | ✗ (no shadow) | ✗ |

Paper main and live rows are NEVER re-deduped — `paper_trader.py` and `live_trader.py` already
prevent re-entry via the open-position-block mechanism.

---

## Pointers in skills

- `live-perf-snapshot` — applies R5 on shadow projection
- `ground-truth-strat-perf` — applies R5 on shadow rows pre-comparison
- `best-combo` — applies R5 on shadow before ranking
- `kol-stats` — applies R3 (token_address) + R5
- `simulate` — applies R4 + R5
- `strategy` — applies R4 + R5

Any new skill or script that touches shadow data MUST link to this doc in its header comment.
