# Pipeline Status — v14e.53 (May 2, 2026 — LIVE PAUSED)

## 🛑 LIVE TRADING PAUSED (May 2 22:12 UTC)

Le live test a rempli son objectif initial : **mesurer le slippage Jupiter Ultra et caractériser le drift live↔paper**. Verdict acté (v14e.50 audit, N=175 paired pairs depuis Apr 29) :
- **Drift médian = -1.46pp** = slip Jupiter Ultra normal (~146 bps round-trip)
- **5 strats homogènes** (drift med -1.20 à -2.36pp) → pas de divergence par mécanique
- **Tail risk rugs** = +1pp avg vs paper, structurel (slip 5000-85000 bps quand un token meurt entre buy et sell)

**Action** : `live_trading.enabled = False` en DB (backup `data/rt_trade_config_pre_pause_20260501T221242Z.json`). Allocations + position size conservés pour reprise rapide quand un winner shadow émergera.

**Pivot stratégique** : focus 100% sur **shadow expansion** pour identifier la meilleure strat possible. Plus de live tant qu'un candidat clair n'a pas émergé du registry.

---

## 🎯 OBJECTIF SESSION — Best strategy via shadow A/B grid

Le 14d audit (`scripts/_what_is_profitable.py`) a révélé 2 sweet spots et 3 dead zones :
```
[ 0- 1h]  N=36818  avg=-1.92%  -$45,238   (pump.fun spam)
[ 1- 3h]  N=14030  avg=+1.06%  +$10,626   *** SWEET SPOT 1 ***
[ 3- 6h]  N= 8300  avg=-5.61%  -$32,260
[ 6-12h]  N= 4923  avg=-12.55% -$25,518   (worst zone)
[12-24h]  N=10279  avg=-4.09%  -$20,783   (AGE24 dead)
[24-48h]  N= 6762  avg=+8.69%  +$28,621   *** SWEET SPOT 2 (AGE48) ***
[48-72h]  N= 2668  avg=-10.57% -$13,429
```

Plus un sweet spot retrade **gap [6-12h]** (avg +22.68% WR 76% +$1999/14d).

---

## 🟢 Cette session — v14e.50 → v14e.53 deployed

### v14e.50 (May 1) — Inline Solana fees + drift audit infrastructure
- ✅ `live_trader.py` : `_fetch_tx_fee_lamports()` + `_tx_fee_usd()` inline → chaque trade enregistre `gas_usd_buy/sell` à l'insert (Helius RPC si `HELIUS_API_KEY`)
- ✅ `scripts/_backfill_solana_fees.py` standalone (223 trades historiques backfillés, fee médian $0.018, mean $0.035 par round-trip)
- ✅ `scripts/_drift_audit_full.py` paired Wilcoxon test live vs paper
- ✅ `scripts/_drift_by_exit_type.py` segments drift par type d'exit
- ✅ `scripts/_wallet_5h_forensic.py` audit forensique PnL+gas+slip

### v14e.51 (May 1) — `max_age_hours=12` sur 5 live + 12 RECALL relaxed shadow
- ✅ Filter `max_age_hours=12` ajouté aux 5 STRATEGY_FILTERS des strats live (block retrades >12h)
- ✅ 12 nouvelles `RECALL_DIP10_*` / `RECALL_ANY_*` / `RECALL_AGE6H/12H/24H_ANY_*` shadow (filters relâchés vs DIP30 strict)
- ✅ Tests integration mis à jour pour le nouveau contrat

### v14e.52 (May 2) — 27 age-band probes des 5 live strats + AGE3H + RECALL×AGE6to12
- ✅ 15 age-band clones des 5 live (A1to3 / A3to12 / A24to48)
- ✅ 6 AGE3H_* shadows sur sweet spot 1-3h
- ✅ 6 RECALL_DIP30_AGE6to12_* sur sweet spot retrade

### v14e.53 (May 2) — LIVE PAUSED + age-band grid étendu à 17 mécaniques
- ✅ Live paused via `scripts/_pause_live_trading.py` (DB JSONB flip)
- ✅ Age-band grid étendu : **17 mécaniques × 3 bands = 51 shadows** (était limité aux 5 live, étendu à TP100/150/200, FAST_TP100_SL20, BE25_LOCK10/15, SCALP, SLOW4H/6H, etc.)
- ✅ Total **SHADOW_STRATEGIES : 648** (était 612 v14e.52, 585 pré-session)

### v14e.53b (May 2) — ETH age-band grid + RECALL relaxed (mirror SOL avec ajustements)
14d audit ETH (`scripts/_eth_profitable.py`) révèle pattern **INVERSE de SOL** :
```
ETH 1er call par age:
[ 0- 1h]  N=2618  avg= +2.53%  +$4977   (SOL était NEGATIF -$45K !)
[ 1- 3h]  N= 717  avg= -7.52%  -$4814   (SOL était sweet spot +$10K)
[ 3- 6h]  N= 261  avg=+21.67%  +$7002   *** SWEET SPOT 1 ETH ***
[ 6-12h]  N= 357  avg=+15.64%  +$3848   *** SWEET SPOT 2 ETH ***
[12-24h]  N=  15  avg=-23.77%   -$713
[24-48h]  N=  97  avg=-16.50%  -$3200   (SOL était sweet spot +$28K !)

ETH retrades par GAP:
[ 6-12h]  N=68  avg=+31.04%  +$1207     *** SWEET SPOT RETRADE ETH ***
```
ETH RECALL : **0 trades en 14d** sur 11 strats existantes (default `max_age_hours=12` ETH bloque tout recall, qui par def est >30min après 1er call).

**50 nouveaux ETH shadows v14e.53b** :
- ✅ **36 ETH age-bands** : 12 mécaniques × 3 bands (A3to6, A6to12, A12to24)
  - Mécaniques : ETH_TP200_SL40_2H_NZ_S40, ETH_BE50_TP150_SL40_T2H, ETH_BE30_TP100_SL40, ETH_FAST_TP200_SL40_60M_MCAP_S40, ETH_TP80_SL40_T2H, ETH_FAST60_TP100_SL50_NZ_S40, ETH_TP100_SL50, ETH_FAST_TP100_SL20, ETH_BE25_LOCK10_TP100_SL20, ETH_BE25_LOCK15_TP100_SL40_T2H, ETH_BE25_LOCK20_TP100_SL30, ETH_BE50_LOCK25_TP200_SL40
- ✅ **5 AGE6H_ETH_*** sur sweet spot 3-6h (BE30, BE25_LOCK15_T2H, FAST_TP100_SL20, TP200_SL40_2H, BE50_TP150_SL40_T2H)
- ✅ **9 ETH_RECALL relaxed** : DIP10×2, ANY×2, AGE6to12×5 (override max_age_hours=72 pour bypass le default ETH 12h)

Différences vs SOL (pattern miroir) :
- ETH age bands sur **3-6h, 6-12h, 12-24h** (PAS 1-3h ni 24-48h qui saignent ETH)
- ETH AGE6H = sweet spot **3-6h** (vs SOL AGE3H = 1-3h)
- ETH RECALL avec `max_age_hours=72` explicite (override le default ETH 12h qui bloquait tout)

Total **ETH strats : 118 → 168**. Total **SHADOW_STRATEGIES : 648 → 698**.

---

## 🔬 Decision rules — verdict shadow grid (~7-14 jours)

À N≥15 par shadow, paired-test vs baseline same-token :

| Hypothèse | Si confirmée → action |
|---|---|
| `*_A1to3` > baseline +5pp paired | → resserrer live filter à `min_age=1, max_age=3` |
| `*_A24to48` positif sur 3+ mécaniques | → green-light AGE48 family promotion |
| `AGE3H_*` top performer | → promote en paper main, candidate live |
| `RECALL_DIP30_AGE6to12_*` +30%+ | → confirme sweet spot retrade, override la version sans age band |
| `*_A3to12` négatif partout | → confirme l'observation 14d, resserrer le filter live |
| `RECALL_PEAK*` toujours 0 trade | → kill les 15 strats mortes (cleanup code) |

---

## 🟡 Wait sur data — N en cours d'accumulation

| Item | N actuel | N cible | ETA |
|---|---:|---:|---|
| **51 SOL age-band shadows v14e.52/53** | 0 | 15 par strat | ~7-14j (Mai 9-16) |
| **6 AGE3H_* SOL shadows** | 0 | 15 | ~7-10j |
| **6 RECALL_AGE6to12_* SOL shadows** | 0 | 10 | ~10-14j (depend retrade frequency) |
| **12 RECALL_DIP10/ANY SOL shadows v14e.51** | 0 | 15 | ~10j |
| **36 ETH age-band shadows v14e.53b** (A3to6/A6to12/A12to24) | 0 | 15 par strat | ~7-14j (Mai 9-16) |
| **5 AGE6H_ETH_* shadows** sweet spot 3-6h | 0 | 15 | ~7-14j |
| **9 ETH_RECALL relaxed** (DIP10/ANY/AGE6to12) | 0 | 10 | ~10-14j (recall ETH rare) |
| BSR_MCAP_AB SOL | 11 | 30 | ~7-10j |
| KW_AB SOL | 4 | 30 | ~10j |
| SCORE_V2_AB | data collection | AUC vs rt_score sur N=30 | ~3j |
| ETH BSR55 (5 strats) | 11 | 30 | ~Mai 7-10 |

---

## 🔴 User action requise

_(rien d'urgent — live paused, shadow gather data tranquillement)_

---

## 📋 Backlog post-shadow-verdict

1. **Resume live** quand un winner shadow émerge (N≥15 + paired-test +5pp vs baseline)
2. **Kill RECALL_PEAK*** (15 strats mortes en 14d) → cleanup code
3. **Promote AGE48 top 3** en paper main si confirmé sur N élargi
4. **AGE24/AGE48 unification** → si A1to3 = sweet spot, refactorer les AGE48 existants
5. **Schema migration** : index sur `paper_trades(token_address, source, status)` pour accélérer les audits cluster (queries 14d-30d devient lent à >100K rows)

---

## 📌 Rappels persistants (rule-encoded)

### Méthode statistique
- **Paired-test apples-to-apples** sur tokens intersection — JAMAIS aggregate avg quand N diffère.
- **N≥30 par strat** avant verdict, N≥15 = "probable" pour décisions intermédiaires.
- Bootstrap CI 95% + sign test obligatoires pour KOL ban/unban.
- Filtrer artefacts (DTRAIL/DIP/SPLIT/TRAIL).
- **Drift live** : paper main twin OU shadow twin via paired-test, **PAS `paper_sim_pnl_pct` companion** (inflate 2-3×).

### Slippage v14e.49b (single source = strategies.py)
- **SOL** : `BUY_SLIPPAGE_BPS = 225` (median empirique 216)
- **ETH** : `BUY 350 / SELL 650` (recalibré v14e.49b sur N=10)
- JSONB miroir ETH : `paper_trade_config.eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`
- Live tx tolerance ETH : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600` (live ≠ paper sim)

### Drift live↔paper (v14e.50 acté)
- **Drift médian global = -1.46pp** sur N=175 pairs = slip Jupiter Ultra normal
- Drift moyen -4.84pp tiré par 5.7% rugs (slip exec 5000-85000 bps)
- **31% trades : live > paper** (positive slippage)
- **Tail risk rug** non modélisé dans le sim → coût hidden +1pp avg

### Solana fees (v14e.50 mesurés)
- Fee per round-trip : median **$0.018**, mean **$0.035** à $1/trade = 1.8-3.5% capital
- Seuil rentabilité = avg PnL >+3.5% par trade
- Inline tracking actif depuis v14e.50, backfill via `scripts/_backfill_solana_fees.py`

### KOL routing v14e.49b
- Per-chain blacklist : **16 SOL** (post v14e.49i), 3 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- `mad_apes_gambles` : SOL ban / ETH allow.
- `ryoshigamble` : flat unban v14e.49i (96.2% WR sur N=371 SOL)
- `ryoshikdegen` : SOL ban / ETH allow.
- `bagcalls`, `batman_gem` : double ban (flat live + chain).
- Live KOL blacklist (flat all-chain) : 6 KOLs.
- KOL whitelist : DISABLED.

### Bankroll
- Total : ~$60K (starting_capital v14e.49 $37K + cumul +$23K).
- **SOL live : PAUSED** depuis v14e.53. allocations conservées pour reprise rapide.
- ETH live : DÉSACTIVÉ.

### Mega-sweep
- SOL : cron `02:00 UTC tous les 2 jours`, matrix split 3 shards.
- ETH : cron `22:00 UTC tous les 2 jours`, single job.
- Persist top-30+50 dans `mega_sweep_runs` Supabase.

### Pattern critique : SOL ↔ ETH sont OPPOSÉS sur l'âge des tokens

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

**Implication** : pas de strat one-size-fits-all. Les filters age/RECALL doivent être **chain-aware**. Le default `max_age_hours=12` ETH (paper_trader.py:282) est bien calibré ETH. Sur SOL le filter `max_age_hours=12` v14e.51 sur les 5 live laisse passer les 3-12h qui saignent — verdict shadow A1to3 va dire si on doit resserrer encore.

### Shadow registry — 698 strats actives (v14e.53b)
| Family | Count | Sweet spot tested |
|--------|---:|---|
| BE | ~33 SOL / 9 ETH | A1to3, A3to12, A24to48 (BE25 5 mecaniques) |
| LOCK | ~33 SOL / 20 ETH | A1to3, A3to12, A24to48 (BE25_LOCK10/15) |
| FAST | ~73 SOL / 18 ETH | A1to3, A3to12, A24to48 (FAST_TP50/100, FAST60) |
| SLOW | ~13 SOL / 5 ETH | A1to3, A3to12, A24to48 (SLOW4H/6H) |
| SCALP | ~36 SOL / 17 ETH | A1to3, A3to12, A24to48 (SCALP_TP15/20) |
| AGE clones (existing AGE24/48/72) | 38 | — |
| AGE3H_* (NEW v14e.52) | 6 | sweet spot 1-3h direct |
| RECALL DIP30/50 | 22 + 4 | — |
| RECALL_DIP10/ANY (NEW v14e.51) | 12 | drift relaxed |
| RECALL_AGE6to12 (NEW v14e.52) | 6 | sweet spot retrade gap 6-12h |
| RECALL_PEAK* | 15 | **0 trade 14d → mortes, à kill** |
| BSR_MCAP combo (SOL only) | 4 | — |
| KW34/26 (KOL win-rate filter) | 8 | — |
| BSR55 ETH only | 5 | — |
