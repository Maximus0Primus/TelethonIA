# Pipeline Status — v14e.56b (May 3, 2026 — LIVE PAUSED)

## 🛑 LIVE TRADING PAUSED

`live_trading.enabled = False` en DB depuis May 2 22:12 UTC. Verdict acté (v14e.50 audit, N=175 paired pairs) :
- Drift médian = -1.46pp = slip Jupiter Ultra normal (~146 bps round-trip)
- 5 strats homogènes (drift med -1.20 à -2.36pp), pas de divergence par mécanique
- Tail risk rugs = +1pp avg vs paper, structurel

Backup config : `data/rt_trade_config_pre_pause_20260501T221242Z.json`. Allocations + position size conservés. Pivot 100% **shadow expansion** jusqu'à émergence d'un winner clair.

---

## 🎯 OBJECTIF — Best strategy via shadow A/B grid

14d audit a révélé pattern OPPOSÉ entre SOL et ETH sur l'âge des tokens :

```
            SOL (14d)         ETH (14d)
[ 0- 1h]    -$45K  saigne     +$5K   positif
[ 1- 3h]    +$10K  sweet 1    -$5K   saigne
[ 3- 6h]    -$32K  saigne     +$7K   sweet 1
[ 6-12h]    -$25K  saigne     +$4K   sweet 2
[12-24h]    -$21K  saigne     -$1K   saigne (les deux)
[24-48h]    +$28K  sweet 2    -$3K   saigne
[48-72h]    -$13K  saigne     +$0.6K positif (N petit)

Retrade gap [6-12h]: SOL +22% / ETH +31%  (les deux profitable)
```

**Implication** : pas de strat one-size-fits-all. Filters age/RECALL doivent être **chain-aware**.

---

## 🟢 Cette session — v14e.56 → v14e.56b deployed (May 3)

### v14e.56b (~13:50 UTC) — mega-sweep 6-shard split + KCO orphan resurrection + zcallz ban

**Mega-sweep SOL 6-shard split** (résout les 5/6 cancelled runs Apr 27 → May 1) :
- `sim.py` : flag `--mega-strat-shard X/N` (default `1/1` = no split, zéro régression). Strat split déterministe par index sorted modulo N.
- `mega-sweep-48h.yml` : matrix `source[3] × strat[2] = 6 shards parallel` (~1.5-2h chacun, vs ~5-6h cancelled au cap GH 6h).
- Smoke test : 697 strats → 349/348 disjoint complet déterministe.
- **À monitorer** : prochain cron May 4 02:00 UTC ou `gh workflow run mega-sweep-48h.yml` manuel.
- ETH workflow inchangé (118-168 strats, 3-shard tient).

**KCO orphan resurrection** (sans toucher `outcome_tracker.py`) :
- `scripts/_kco_resurrect_dead.py` idempotent dry-run-default. Force-mark `dead_no_ohlcv` rows NULL outcome_status >72h. Backup table `_backup_kco_resurrect_20260503` (1089 rows).
- Run réel : NULL **1507 → 484** (61.3% → 19.7%), dead_no_ohlcv 890 → 1912 (36.2% → 77.8%).
- Bug racine : Phase C silently skip fresh-mint pump.fun rugs (entry_price filled au spawn mais pas d'OHLCV historique). Patch côté code laissé en backlog (ML training depends, hors session).

**Ban zcallz SOL chain** :
- Pattern `pump.fun + age<2h + liq/mcap>1.0` 14d → 2 tokens (UPEG + MEN). MEN callé par `zcallz`.
- 7d shadow N=346 **WR 0%** -$8.9K. 3d shadow N=1014 -$10,145. SOL blacklist 17 → **18 KOLs**.

### v14e.56 (~10:50 UTC) — UPEG ticker-hijack incident

**Incident** : 8 paper main + 343 shadow trades fired sur fake `$UPEG` SOL (`Bwx2Rqsh...`, fresh-mint pump.fun 1.7h). Net -$44 (7 winners +$56 vs 2 rugs -$100). Le vrai $UPEG est sur ETH (`0x44b28991...`, 128x). TheReaperGems a posté **directement** le CA SOL scam (logs VPS confirmés : `Resolved CA solana:Bwx2Rqsh… → $UPEG`, msg→detect=27.46s). Pas un bug d'extraction.

**Cause racine** : TheReaperGems shadow 3d a basculé en rug magnet (N=703, WR 18.6%, rug rate 30.2%, vs 14d avant WR 81.6%, 0 rugs).

**Actions** :
- Ban TheReaperGems SOL chain (16 → 17 KOLs). Backup `_backup_ban_reapergems_20260503`.
- Fix #1 RT shape gate (`safe_scraper.py:2089-2100`) : skip explicite si `detect_chain(ca) is None` (au lieu du fallback silencieux `or "solana"`). Honors `chain_detect.py:47-48` warn. Zéro régression (downstream rejetait déjà).

---

## 🔬 Decision rules — verdict shadow grid (~7-14j)

À N≥15 par shadow, paired-test vs baseline same-token :

| Hypothèse | Si confirmée → action |
|---|---|
| `*_A1to3` > baseline +5pp paired | resserrer live filter `min_age=1, max_age=3` |
| `*_A24to48` positif sur 3+ mécaniques | green-light AGE48 family promotion |
| `AGE3H_*` top performer | promote en paper main, candidate live |
| `RECALL_DIP30_AGE6to12_*` +30%+ | confirme sweet spot retrade |
| **`*_A3to12` positif sur 9+ mécaniques** | **NEW v14e.56b** : confirme A3to12 capture RECALL window (signal early Δ +7 à +17pp paired sur N=4-8). Si tient à N=15 → promote 3-5 top en paper main, candidate live. Inversion vs verdict 14d audit (qui disait 3-12h saigne) — probablement le filter A3to12 capture le 2e+ entry, pas le 1er call. |

---

## 🟡 Wait sur data — N en cours d'accumulation (audit May 3 ~14:00 UTC)

| Item | N actuel/strat | N cible | Note |
|---|---:|---:|---|
| **51 SOL age-band shadows v14e.52/53** | 4.8-7.3 | 15 | Early signal A3to12 positif (+7 à +17pp), A1to3 mixte, A24to48 trop tôt |
| **6 AGE3H_* SOL shadows** | 7 | 15 | |
| **6 RECALL_AGE6to12_* SOL shadows** | 5.9 | 10 | |
| **12 RECALL_DIP10/ANY SOL shadows v14e.51** | 5.9 | 15 | |
| **36 ETH age-band shadows v14e.53b** | 2 | 15 | ETH volume KOL en chute -75% (Apr 27 24/d → May 3 2/d). ETA verdict éloigné |
| **5 AGE6H_ETH_* shadows** sweet spot 3-6h | — | 15 | Idem ETH volume |
| **9 ETH_RECALL relaxed** | — | 10 | Idem ETH volume |
| **BSR_MCAP_SOL (4 strats `_BSR_MCAP`)** | 49.3 | 30 | ✅ **VERDICT NÉGATIF DÉCISIF** : Δ -12 à -16pp paired vs baseline (N=65-73). À KILL |
| **BSR55_ETH (5 strats)** | 22 | 30 | 4/5 positifs sur paired Δ +2.5 à +3pp MAIS $/d net inférieur à baseline ($26 vs $33 sur top). Filter coupe trop de volume → **PAS de promote**. Garder en shadow telemetry. |
| KW_AB SOL (5 strats `_KW34`) | 8.4 | 30 | |
| KW_AB ETH (3 strats `_KW26`) | 7 | 30 | |
| **SCORE_V3_AB** (rt_score_v3 v14e.55) | accumule | walk-forward AUC ≥ V1 sur 7-14j (Mai 9-16) | |
| W1-W5 paired-tests (SCALP, AGE, LOCK, ETH cluster) | varies | 30 | |
| W9b RECALL family verdict par bucket | varies | 30 | |
| T2 sell slip drift twin pairs | varies | 200 | |

---

## 🔧 Backlog technique

- [ ] **Fix #2 verb-proximity ticker filter** — `$X (whales|holders|dev) (buying|aping) $Y` patterns. Demande N≥30 messages historiques pour calibrer sans casser les calls légitimes.
- [ ] **Fix #3 fresh-mint anti-rug RT gate** — `pump.fun + age<2h + liq/mcap>1.0` → shadow-only. Pattern 14d audité = 2 tokens (UPEG + MEN), zéro faux positif. Risque très faible mais demande seconde validation user avant deploy.
- [ ] **outcome_tracker Phase C patch côté code** — fix permanent du Phase C silently-skip (entry_price filled mais pas d'OHLCV historique → reste NULL indefiniment). Le script v14e.56b résout le stock historique ; le flow reste cassé pour les nouveaux orphans. Demande audit ML training pre/post.
- [ ] **Score V3 walk-forward audit** (Mai 9-16) — sur N≥30 trades fermés post-v14e.55. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter. Script à créer : `scripts/_score_v3_walk_forward.py`.
- [ ] **R2** Profiler `process_and_push` si lag >30s revient.
- [ ] **T3** `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.
- [ ] **T8** Auto-apply JSONB diff via PR si signal stable 4 sem K1/K2.
- [ ] **LOCK polling alignment** — appliquer `polling_sec=60 + median_5` aux LOCK SOL/ETH si BE25 confirme post-bump.

---

## 📋 Backlog post-shadow-verdict (séquencé)

1. **Resume live** quand un winner shadow émerge (N≥15 + paired-test +5pp vs baseline). Backup config : `data/rt_trade_config_pre_pause_20260501T221242Z.json`.
2. **Promote AGE48 top 3** en paper main si confirmé sur N élargi.
3. **AGE24/AGE48 unification** → si A1to3 = sweet spot SOL, refactorer les AGE existants.

---

## ⏸️ Tâches en suspens

- [ ] **Test BE15_LOCK5 hypothèse slip-sensitivity** — verdict v14e.50 N=28 drift -2.36pp = 2× LOCK10 (-1.20pp) mais pas disqualifiant. Re-check à N=50.
- [ ] **Rolling windows feature review** — committed v14e.49i. Revoir `analyze_mega_sweep.py` pour validation user.

---

## 📌 Rappels persistants (rule-encoded)

### Filter combos verdict (v14e.56b audit)
- **BSR_MCAP SOL** (`BSR≥0.53 AND mcap≥$45K`) — validé v14e.43b walk-forward (+$20-26/d sur SLOW/TP100). **Inversé à N=65-73** : Δ paired -12 à -16pp vs baseline. Cause probable : regime change marché + le filter MCAP coupe les moonshots SOL qui font le profit. **Action : kill 4 strats `_BSR_MCAP` du SHADOW_STRATEGIES** (non encore appliqué code, à valider).
- **BSR55 ETH** (`BSR≥0.55` ETH only) — gardé v14e.49b. À N=22 paired-test 4/5 positifs Δ +2.5 à +3pp MAIS $/d net 7d inférieur à baseline (ex: ETH_BE25_LOCK15_TP100_SL40_T2H_BSR55 +$26.63/d vs base sans filter +$32.73/d). **Le filter monte l'avg% en coupant trop de volume → $/d net plus bas**. Règle rule-encoded : **avg% supérieur n'implique PAS $/d supérieur** quand le filter réduit le call rate. Toujours auditer $/d réel sur 7d en plus du paired test. Pas de promote BSR55 en paper main tant que ETH KOL volume ne rebondit pas (chute -75% Apr 27 → May 3).

### Méthode statistique
- **Paired-test apples-to-apples** sur tokens intersection — JAMAIS aggregate avg quand N diffère.
- **N≥30 par strat** avant verdict, N≥15 = "probable" pour décisions intermédiaires.
- Bootstrap CI 95% + sign test obligatoires pour KOL ban/unban.
- Filtrer artefacts (DTRAIL/DIP/SPLIT/TRAIL).
- **Drift live** : paper main twin OU shadow twin via paired-test, **PAS `paper_sim_pnl_pct` companion** (inflate 2-3×).

### Slippage v14e.49b (single source = strategies.py)
- **SOL** : `BUY_SLIPPAGE_BPS = 225` (median empirique 216).
- **ETH** : `BUY 350 / SELL 650` (recalibré v14e.49b sur N=10).
- JSONB miroir ETH : `paper_trade_config.eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`.
- Live tx tolerance ETH : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`.

### Drift live↔paper (v14e.50)
- Drift médian global = -1.46pp sur N=175 pairs = slip Jupiter Ultra normal.
- Drift moyen -4.84pp tiré par 5.7% rugs (slip exec 5000-85000 bps).
- 31% trades : live > paper (positive slippage).
- Tail risk rug non modélisé dans le sim → coût hidden +1pp avg.

### Solana fees (v14e.50)
- Fee per round-trip : median **$0.018**, mean **$0.035** à $1/trade = 1.8-3.5% capital.
- Seuil rentabilité = avg PnL >+3.5% par trade.
- Inline tracking actif depuis v14e.50.

### KOL routing (v14e.56b)
- Per-chain blacklist : **18 SOL**, 3 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- `mad_apes_gambles` : SOL ban / ETH allow.
- `ryoshigamble` : flat unban v14e.49i (96.2% WR sur N=371 SOL).
- `ryoshikdegen` : SOL ban / ETH allow.
- `bagcalls`, `batman_gem` : double ban (flat live + chain).
- Live KOL blacklist (flat all-chain) : 6 KOLs.
- KOL whitelist : DISABLED.

### Bankroll
- Total : ~$60K (starting_capital v14e.49 $37K + cumul +$23K).
- **SOL live : PAUSED** depuis v14e.53. allocations conservées pour reprise rapide.
- ETH live : DÉSACTIVÉ.

### Mega-sweep (v14e.56b)
- SOL : cron `02:00 UTC tous les 2 jours`, matrix split **6 shards** (3 sources × 2 strat-halves), ~1.5-2h chacun.
- ETH : cron `22:00 UTC tous les 2 jours`, single job (3-shard hold).
- Persist top-30+50 dans `mega_sweep_runs` Supabase.

### Shadow registry — 676 strats actives (post-v14e.55 RECALL_PEAK kill)
| Family | Count | Sweet spot tested |
|--------|---:|---|
| BE | ~33 SOL / 9 ETH | A1to3, A3to12, A24to48 |
| LOCK | ~33 SOL / 20 ETH | A1to3, A3to12, A24to48 |
| FAST | ~73 SOL / 18 ETH | A1to3, A3to12, A24to48 |
| SLOW | ~13 SOL / 5 ETH | A1to3, A3to12, A24to48 |
| SCALP | ~36 SOL / 17 ETH | A1to3, A3to12, A24to48 |
| AGE clones (existing AGE24/48/72) | 38 | — |
| AGE3H_* (v14e.52) | 6 | sweet spot 1-3h direct |
| AGE6H_ETH_* (v14e.53b) | 5 | sweet spot 3-6h ETH |
| RECALL DIP30/50 | 22 + 4 | — |
| RECALL_DIP10/ANY (v14e.51) | 12 | drift relaxed |
| RECALL_AGE6to12 (v14e.52) | 6 | sweet spot retrade gap 6-12h |
| ETH_RECALL relaxed (v14e.53b) | 9 | bypass default ETH 12h |
| ETH age-bands (v14e.53b) | 36 | A3to6, A6to12, A12to24 |
| BSR_MCAP combo (SOL only) | 4 | — |
| KW34/26 (KOL win-rate filter) | 8 | — |
| BSR55 ETH only | 5 | — |
| Score V3 A/B | persisted col `rt_score_v3` | walk-forward Mai 9-16 |

---

## 🗂️ Historique des sessions

Détails complets dans git log. Synthèse :
- **v14e.50** (May 1) — Inline Solana fees + drift audit infra (`_drift_audit_full.py`, `_drift_by_exit_type.py`, `_wallet_5h_forensic.py`)
- **v14e.51** (May 1) — `max_age_hours=12` sur 5 live + 12 RECALL relaxed shadows
- **v14e.52** (May 2) — 27 age-band probes + AGE3H + RECALL×AGE6to12
- **v14e.53** (May 2) — LIVE PAUSED + age-band grid 17 mécaniques × 3 bands = 51 shadows
- **v14e.53b** (May 2) — ETH age-band grid + RECALL relaxed (50 ETH shadows, total 168 ETH strats)
- **v14e.54** (May 2) — Promote 5 robust shadows en paper main (SOL BE_LOCK10, FAST_MCAP, 2 moonshots, 1 ETH cluster)
- **v14e.55** (May 2 23:00) — Score V3 shadow A/B + RECALL_PEAK kill (22 strats) + DB index `idx_paper_trades_strategy_status_created`
- **v14e.56** (May 3 ~10:50) — UPEG hijack + RT shape gate + ban TheReaperGems
- **v14e.56b** (May 3 ~13:50) — Mega-sweep 6-shard split + KCO orphan resurrection + ban zcallz
