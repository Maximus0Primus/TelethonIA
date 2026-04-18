# Pipeline Status — Updated Apr 19, 2026 (v143.6 deployed)

## Current state

**Live (50/50)** — `BE25_TP80_SL30` (median_5/240s) + `BE15_TP100_SL50` (ds/30s). Position ~$1.70/trade, max 3 open. Configs live identiques au paper post-revert (A/B base préservé).

**Paper Telegram — 21 strats active × $1000 bankroll ($21K seed post-v142) + 9 shadows v142**. Orchestration per-strat via `rt_trade_config.strategy_overrides` + `LAZY_STRATEGIES`.

### 21 Mains actives — stats 7d (27,223 trades total)

| Strat | Orch | N 7d | WR | avg% | **$ 7d** | bankroll |
|---|---|---|---|---|---|---|
| **FAST_TP80_SL25** ⭐ | ds/30s + LAZY | 54 | 37% | +5.83% | **+$157** | $1029 |
| **FAST_TP40_SL30** ⭐ | hysteresis/30s + LAZY | 11 | 27% | +27.52% | **+$151** | $1158 |
| **BE25_TP80_SL30** | median_5/240s | 77 | 32% | +2.10% | **+$87** | $979 |
| **FAST_TP100_SL20** | ds/30s + LAZY | 65 | 31% | +1.61% | **+$52** | $1024 |
| **FAST_TP50_SL30** | median_3/30s + LAZY | 123 | 41% | +1.85% | **+$47** | $1100 |
| **TP50_SL15** | jupiter/30s | 11 | 45% | +4.74% | **+$26** | $1027 |
| BE25_TP80_SL30_DS | ds/30s + LAZY | — | — | — | — | $1031 |
| BE15_TP70_SL50_NZ (NOZEROLIQ) | jupiter/240s | — | — | — | — | $975 |
| BE15_TP300_SL50_MCAP (MCAP_MID) | ds/30s | — | — | — | — | $970 |
| BE15_TP100_SL50 | ds/30s | — | — | — | — | $904 |
| FAST_TP100_SL20_HYST | hysteresis/30s + LAZY | 8 | 12% | −18.25% | **−$73** | $927 |
| FAST_TP80_SL25_HYST | hysteresis/30s + LAZY | 8 | 12% | −19.67% | **−$79** | $921 |
| FAST_TP50_SL30_HYST | hysteresis/30s + LAZY | 8 | 25% | −15.81% | **−$63** | $936 |
| BE25_TP80_SL30_HYST | hysteresis/30s + LAZY | 8 | 12% | −17.66% | **−$71** | $929 |
| BE25_TP80_SL30_S30_HYST (SCORE30) | hysteresis/240s | — | — | — | — | $989 |
| BE25_TP80_SL30_NZS30_HYST (NZ+S30) | hysteresis/240s | — | — | — | — | $987 |
| HIGHSCORE_TP200_SL40 (SCORE30) | jupiter/120s | — | — | — | — | $956 |
| NOZEROLIQ_TP200_SL40 (liq>0) | jupiter/120s | — | — | — | — | $906 |
| FAST_TP70_SL50 🆕 (Apr 18) | winsor_p95/30s + LAZY | 0 | — | — | — | $1000 |
| BE15_TP200_SL40_4H 🆕 | hysteresis/60s | 0 | — | — | — | $1000 |
| MCAP_MID_DTRAIL5_ACT25_SL50_2H 🆕 | median_5/120s | 0 | — | — | — | $1000 |

**Totaux 7d** :
- **6 bases non-HYST** : N=341, **+$520 / 7j** = **+$74/jour** ✅
- **4 HYST variants** : N=32, **−$286 / 7j** = **−$41/jour** (N=8 chacun, activées Apr 17)
- **Net book** : ~+$234 / 7j sur bases + HYST combinés

**Global rt_bankroll** : $17,750 current / $18,722 peak.

### Paired A/B comparison HYST vs base (même token/date, 7d)

| Paire | N | mean Δ (HYST−base) | median Δ |
|---|---|---|---|
| FAST_TP100_SL20 | 8 | −0.47% | −0.64% |
| **FAST_TP80_SL25** | 8 | **−6.61%** | −2.75% |
| FAST_TP50_SL30 | 8 | −1.28% | −0.45% |
| BE25_TP80_SL30 | 8 | −2.07% | −1.11% |

**4/4 paires : HYST perd**. Direction consistante, mais N=8/pair. Need N≥30 pour verdict définitif (ETA Apr 22-23).

### Live vs Paper same-strat (depuis 2026-04-17 13:50, excl. pump.fun outliers)

| Strat | Matched | Avg diff | Median | Max | within_10pp | Same status | Entry div | Exit div |
|---|---|---|---|---|---|---|---|---|
| **BE25_TP80_SL30** | 15 | +2.90pp | +1.54pp | 23pp | **12/15 (80%)** | **100%** | 2.9% | +1.8% |
| **BE15_TP100_SL50** | 12 | +16.17pp | −1.16pp | 215pp | 6/12 (50%) | 10/12 | 2.6% | **−10.4%** |

BE25 bien aligné, BE15 a 1-2 outliers qui écrasent la moyenne.

### 9 Shadows v142 (bankroll $0)
TD2_BE5_TP120_SL44_T25, PTRAIL_V2_T10-18-30-45_SL30_T60, BOND_FAST_TP50_SL20_T20, SCORE40_FAST_TP50_SL30_30M, FAST_TP200_SL40_60M, DIP30_B10_T10_A20_SL60_120m, BE15_TP150_SL40_2H, FAST_TP500_SL40_60M.

---

## 📋 Reste à faire

### ⏳ Data wait
- **3 mains v142** (FAST_TP70, BE15_TP200_4H, MCAP_MID_DTRAIL5) — N≥15 ~ Apr 20-21
- **9 shadows v142** — N≥20 ~ Apr 21-22
- **HYST verdict** (paired N≥30) — Apr 22-23
- **v143.5 exit shadow-sync validation** — 48h, query `exit_div_pct` BE15 doit tomber <2% — Apr 20-21
- **v143.6 paper_sim_pnl_pct populate** — 24h, joint analysis `pnl_pct vs paper_sim_pnl_pct` isolera slip/routing — Apr 20
- **Slip calibration v144** par liq_bucket — N≥15 live/bucket — Apr 22-24

### 🔴 Open bugs (need data)
- **P3** : slip model sur-pénalise liq>$50K, sous-pénalise bondings. N≥30 par bucket.
- **S5** : NOZEROLIQ/HIGHSCORE filtres continuent de perdre. N≥50 par bucket.
- **BE15 exit divergence** : -10.4% median. v143.5 devrait résoudre — re-measure Apr 20.

### 🟡 Améliorations alignment identifiées (low-priority)
- **Tick logging 15-30s → 5-10s** : gain DTRAIL outliers maxabs −36pp → <5pp, coût Jupiter RPC 2-3x. À envisager SI #v143.5/v143.6 ne résolvent pas tout.
- **Shadow-sync entry pour trades non-hybrid** : v142E ne sync que hybrid. Low-stakes si on reste 100% hybrid.
- **Update 4 autres callsites `_replay_with_intervals`** pour passer `dex_ticks` systématiquement (actuellement seul mega-sweep le fait). Utile si un jour on active confirm/twin_confirm/hybrid sur une stratégie non-mega.

### 🔒 Bloqué sur scale-up live
- **Jupiter Trigger V2** — 0 fills historiques. Débloquer quand live_pos > $10.

### 🧠 Gotcha
Supabase PostgREST cap 1000 rows même avec `.limit(10000)`. Toujours paginer via `.range(off, off+999)`. Pattern : `sim.py::sb_get`.

---

## Sim ↔ Live/Paper coherence (v143 = aligned)

### Status post-v143.6
- **Avg sim-live diff** : +0.09pp (centered on zero)
- **BE25 within_10pp** : 13/19 (68%)
- **BE15 within_10pp** : 10/15 (67%)
- **Outliers restants** : structurels (polling cadence + tick 15-30s resolution), pas bugs logiques

### v143 changelog (Apr 18-19)
- **v143 + 143.1** (`3c6cfee` + `5c35a0d`) — verify_sim_live_alignment utilise `_decision_price` + reset `high_price_seen` dans fake trade (bug critique : BE armait au tick 1)
- **v143.2** (`7ab6f6a`) — sim.py port single-stream modes (jp_sampled_60s/180s, vwap_5min, ohlc_burst_60s)
- **v143.3** (`07c62f6`) — sim.py port dual-stream modes (confirm, twin_confirm, hybrid) via `dex_ticks` param
- **v143.4 + 143.5** (`0aeac2d`) — mega-sweep passe `dex_ticks`, live_trader exit shadow-sync (symétrique à v142 E)
- **v143.6** (`be44422`) — DS cache 5s TTL dans `_fetch_prices_batch`, colonne `paper_sim_pnl_pct` persistée par live_trader, CI gate nightly `sim-align-gate.yml`
- **chore** (`54b10d5`) — suppression 6 scripts obsolètes (check_alerts, query_trades, _apply_v140_*, _rt_score_v2_audit, cleanup_hybrid_sell_pollution)

### Tools
- `--from-eval-history` (v138) = 0% bias mathématique
- `--from-trades` = ground truth historique
- `--from-ticks jupiter` = +4pp residual sur trail-heavy
- `scripts/verify_sim_live_alignment.py` = audit sim vs live (tourne en CI nightly)

**Thresholds CI** : avg |diff| ≤ 5pp ET within_10pp ≥ 80% sinon fail + Telegram alert.

---

## Architecture summary

**Scoring :** 40.5/13.5/40.5/5.4 (consensus/conviction/breadth/PA), 16-multiplier chain.
**Trading :** Paper slip dynamic, live Jupiter Ultra RFQ ~10bps, position reconciliation sibling-aware (v133-D), loss limit 0.5 SOL/jour.
**Alerting :** ML disabled, RT listener uncapped, GH Actions failures, daily summary 8am UTC.

## Workflow sim

| Mode | Flag | Use case |
|---|---|---|
| Grid focused | `--from-ticks` | Ranking rapide par strategy |
| Ground truth | `--from-trades` | Vérité terrain historique |
| 0% bias | `--from-eval-history` | Perfect replay post-v138 |
| Mega sweep | `--mega-sweep` | Full matrix 134K configs |

```bash
python scraper/sim.py --from-ticks --since 2026-04-13 --top 30
python scraper/sim.py --from-trades --since 2026-04-13
python scraper/sim.py --from-eval-history --since 2026-04-17
python scraper/sim.py --mega-sweep  # flags: --mega-workers N, --mega-csv-out, --mega-since
```

## Historique récent

- **v143.6** (Apr 19) ✅ DS cache TTL + `paper_sim_pnl_pct` column + CI nightly gate
- **v143.5** (Apr 19) ✅ Live exit shadow-sync : force-close paper match au fill Jupiter
- **v143.1-4** (Apr 18-19) ✅ Sim alignment fixes (`_decision_price`, `high_price_seen` reset, 7 smoothing modes ports)
- **v142 E** (Apr 18) ✅ Entry shadow-sync : paper reuse live `execution_price` via `_rt_force_entry_price`
- **v142 A-D** (Apr 18) ✅ Mega sweep 134K configs → 3 new mains + 9 shadows + 3 smoothing modes + OHLC burst port
- **v141** (Apr 17) ✅ rt_score +3 bonuses data-driven (corr +0.207 → +0.236)
- **v140** (Apr 17) ✅ 8 new strats, `_BE_RE` regex relaxé, bankroll reset $18K
- **v138.5** (Apr 17) ✅ Slip recalibration (sl_hit 435bps, trail 250bps, tp +300bps)
- **v138** (Apr 17) ✅ `eval_history` JSONB + `cache_snapshots` table + `--from-eval-history`
