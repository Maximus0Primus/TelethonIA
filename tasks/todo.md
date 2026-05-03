# Pipeline Status — v14e.55 (May 2, 2026 — LIVE PAUSED)

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

## 🟢 Cette session — v14e.50 → v14e.56 deployed

### v14e.56 (May 3 ~10:50 UTC) — UPEG ticker-hijack incident + RT shape gate + TheReaperGems SOL ban

**Incident** : 8 paper main + 343 shadow trades fired sur `$UPEG` (CA SOL `Bwx2Rqsh1UmzV1ZfKFxDBoeRBTnzocCcJJcuTpGx8H84`, fresh-mint pump.fun 1.7h, top5_pct=68.97%, bundle_pct=34.83%, lp_locked=0%) sur trigger TheReaperGems. Net: 7 winners +$56 vs 2 rugs -$100 = **−$44 net**. Le vrai $UPEG est sur ETH (`0x44b28991...`, fait 128x). Le bot a acheté un scam SOL fresh-mint qui parasite le ticker.

**Cause identifiée** :
- TheReaperGems shadow 3d a basculé en rug magnet : N=703, avg -44%, **WR 18.6%**, **rug rate 30.2%** (vs 14d avant: avg +37%, WR 81.6%, 0 rugs). KOL hacké ou tournique en pump-and-dump fresh-mints.
- Pas de gate anti-rug en RT (top10/bundle/lp_locked sont fetchés en batch post-buy, trop tard).
- Le `chain_detect.py:47-48` warn "treat None as skip" est violé par `safe_scraper.py:2089` `or "solana"` fallback silencieux.

**Actions appliquées** :
- ✅ **Ban TheReaperGems SOL chain blacklist** (DB JSONB, 16 → 17 KOLs SOL banned). Backup table `_backup_ban_reapergems_20260503`. Ne ban PAS ETH (stats ETH KOL non auditées). Stop saignement immédiat.
- ✅ **Fix #1 RT shape gate** (`safe_scraper.py:2089-2100`) : `detect_chain(ca) or "solana"` → `detect_chain(ca)` + skip explicite si `None`. Garbage CA / 0x mal-formé / extraction buggée → skip plus tôt avec log `RT SKIP: <symbol> — unrecognized CA shape`. Zéro régression : `_fetch_dexscreener_by_address` rejetait déjà ces CAs (enrich.py:540-543), on déplace juste le skip + log clair.

**Mystère non résolu** : le message Telegram qui a déclenché le buy (`paper_trades.message_ts = 09:23:10`) n'est pas en `kol_mentions`. Les seuls TheReaperGems messages voisins sont 09:17:30 ("$UPEG whales buying $UCORE") et 09:52:17 — mais le RT loop skip les tickers sans CA (`if not ca: continue` ligne 2060), donc impossible que ce message ait fired le buy. Hypothèse forte : un message à 09:23:10 contenait soit une URL DexScreener/pump.fun avec le CA SOL scam (parsée via `msg.entities`), soit le CA en clair. Logs VPS nécessaires pour confirmer.

**Backlog fix #2 / #3 (à valider avant déploiement, non urgent vu le ban)** :
- [ ] **Fix #2 verb-proximity** : filtrer les tickers en position passive (`$X (whales|holders|dev|chart) (buying|aping|holding) $Y`, `better than $X`, `vs $X`). Risque: faux positifs sur calls légitimes. Demande N≥30 messages historiques pour calibrer les patterns sans casser.
- [ ] **Fix #3 fresh-mint anti-rug RT** : ajouter un check "if `is_pump_fun=1` AND `token_age<2h` AND `liq/mcap>1.0` AND extraction_method='ticker'" → shadow-only (pas paper main). Demande audit historique pour quantifier faux positifs (combien de fresh-mint pump.fun avec liq>mcap sont en réalité légitimes ?).
- [ ] **Audit logs VPS** pour identifier exact contenu message 09:23:10 — informe la calibration des fix #2/#3.



### v14e.55 (May 2 23:00 UTC) — Score V3 + RECALL_PEAK kill + DB index
- ✅ **Score V3** (`safe_scraper._rt_compute_score_v3`) : V1 base + binned bonuses sur sweet spots SOL 7j (MCAP 50-100k +15, AGE 1-3h +10, AGE 24-72h +8, MCAP ≥1M -10, AGE 3-6h -8, LIQ <10k -10). In-sample AUC 0.562 vs V1 0.534 (+0.028). Filter ≥35 : WR +1.92pp et avg PnL 2× sur 3j. **Persisté en parallèle** (col `rt_score_v3` ajoutée à `paper_trades`), pas utilisé en filter, walk-forward attendu Mai 9-16.
- ✅ **V2 dépréciée** (kept for backward compat) : audit N=37K paired montre AUC 0.486 (anti-prédictif). Les 4 features V2 (kol_wr, BSR, vol, mcap linéaires) sont individuellement anti-prédictives.
- ✅ **RECALL_PEAK kill** : 22 strats supprimées (`kol_call_outcomes.ath_after_call` fill rate 5%). SHADOW_STRATEGIES 698 → 676.
- ✅ **DB index** `idx_paper_trades_strategy_status_created` (strategy, status, created_at DESC) — accélère audits per-strat 14d-30d.



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

### v14e.54 (May 2) — Promote 5 robust shadows en paper main (alertes Telegram)
Cross-window robustness audit (`scripts/_best_strats_robustness.py`) sur 14d/7d/3d a identifié les strats positives sur les 3 fenêtres avec N≥20 et WR≥45%.

**5 strats promues paper main** (hybrid_strategy.allocations 28 → 33) :
- **SOL safe BE_LOCK10** : `BE25_LOCK10_TP80_SL30_S40` (N=95, WR 56%, +$18.47/d → +$36.94/d 7d → +$62.50/d 3d)
- **SOL FAST + MCAP filter** : `FAST_TP50_SL30_MCAP_S40` (N=97, WR 51%, +$19.90/d → +$53/d → +$92/d)
- **SOL moonshot WR 60%** : `BE25_LOCK15_TP200_SL40_4H_NZ_S40` (N=80, WR 60%, +$24.29/d, accelerating)
- **SOL moonshot top 3d** : `BE50_LOCK25_TP200_SL40_4H_MCAP` (N=86, WR 48%, +$21/d → **+$112/d 3d**)
- **ETH top WR all chains** : `ETH_BE25_LOCK15_TP100_SL40_T2H` (N=62, **WR 69%**, +$18/d → +$78/d 3d)

Bankrolls seedés $1000 chacune (4 nouvelles, 1 existante avec balance $1046). starting_capital $37K → $41K. Backups : `data/rt_trade_config_pre_promote_*` + `data/rt_bankroll_pre_promote_*`.

**Note feedback** : paper main = même calcul que shadow, juste alertes Telegram en plus. Pas de "wait pour valider" — la donnée shadow était suffisante.

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
| **51 SOL age-band shadows v14e.52/53** | 0 | 15 par strat | |
| **6 AGE3H_* SOL shadows** | 0 | 15 | |
| **6 RECALL_AGE6to12_* SOL shadows** | 0 | 10 | (depend retrade frequency) |
| **12 RECALL_DIP10/ANY SOL shadows v14e.51** | 0 | 15 |  |
| **36 ETH age-band shadows v14e.53b** (A3to6/A6to12/A12to24) | 0 | 15 par strat | 
| **5 AGE6H_ETH_* shadows** sweet spot 3-6h | 0 | 15 | |
| **9 ETH_RECALL relaxed** (DIP10/ANY/AGE6to12) | 0 | 10 | |
| BSR_MCAP_AB SOL (4 strats `_BSR_MCAP`) | 11 | 30 | |
| KW_AB SOL (5 strats `_KW34`) | 4 | 30 | 
| KW_AB ETH (3 strats `_KW26`) | 3 | 30 | 
| **SCORE_V3_AB** (rt_score_v3 v14e.55) | 0 (deploy pending) | walk-forward AUC ≥ V1 sur 7-14j out-of-sample | Mai 9-16 |
| ETH BSR55 (KEPT v14e.49b, 5 strats) | 11 | 30 | 
| W1-W5 paired-tests (SCALP, AGE, LOCK, ETH winner cluster) | varies | 30 | 
| W9b RECALL family verdict par bucket | varies | 30 | 
| T2 sell slip drift twin pairs | varies | 200 | 

---

## 🔴 User action requise

### MEGA-SWEEP SOL en panne (timeout cap 6h GH free-tier)
- **Statut** : 5 runs cancelled sur 6 derniers (Apr 27 → May 1). Dernier success = Apr 29 15:05 (run 25116811803). ETH workflow OK (last success May 1 22:53).
- **Run dispatch test** : `25261805651` lancé May 2 21:02 UTC pour confirmer/infirmer le timeout. À checker au prochain retour de session.
  - Si ✅ success → fluke (queue GH chargée), pas d'action immédiate. Pull top robust pour décisions promote.
  - Si ❌ cancelled → option B : ajouter flag `--mega-strat-shard` dans `sim.py` pour passer de 3 à 4 shards (split jupiter en 2 par range de strats). ~30 lignes, 0 perte de précision (juste parallélisation plus fine).
- **NOT possible** : self-hosted runner sur VPS (rejeté user).
- Workflow : `.github/workflows/mega-sweep-48h.yml`

---

## 🔧 Backlog technique (pas urgent)

- [ ] **Score V3 walk-forward audit** (Mai 9-16) — sur N≥30 trades fermés post-déploiement v14e.55, recalculer AUC V3 vs V1 sur out-of-sample. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter en `min_rt_score_v3` sur le live deck. Si AUC V3 < V1 → garder V1 et investiguer pourquoi les bonuses MCAP/AGE n'ont pas tenu (regime change ?). Script à créer : `scripts/_score_v3_walk_forward.py`.
- [ ] **R2** — Profiler `process_and_push` si lag >30s revient.
- [ ] **T3** — `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.
- [ ] **T8** — Auto-apply JSONB diff via PR si signal stable 4 sem K1/K2.
- [ ] **LOCK polling alignment (parqué)** — appliquer `polling_sec=60 + median_5` aux LOCK SOL/ETH si BE25 confirme post-bump.
- [ ] **outcome_tracker fill rate** — diagnostiquer pourquoi 61% des `kol_call_outcomes` ont `outcome_status=NULL` 14d et 36% sont `dead_no_ohlcv`. Si fill remonte à ≥50%, ré-ajouter 3-5 RECALL_PEAK shadows (mécanique $PARANOID +220% encore intéressante en théorie).

---

## 📋 Backlog post-shadow-verdict (séquencé)

1. **Resume live** quand un winner shadow émerge (N≥15 + paired-test +5pp vs baseline). Backup config disponible : `data/rt_trade_config_pre_pause_20260501T221242Z.json`
2. ~~Kill RECALL_PEAK*~~ — ✅ DONE May 2 22:45 UTC. 22 PEAK strats supprimées (5 SOL initial + 1 ETH initial + 13 SOL _add_recall + 3 ETH _add_recall). SHADOW_STRATEGIES: 698 → 676. Cause root: `kol_call_outcomes.ath_after_call` fill rate = 5.0% sur 14d (1667/2720 NULL outcome_status, 976 dead_no_ohlcv) → 0 trades 14d sur toute la family. Re-add seulement si outcome_tracker fill rate ≥50%.
3. **Promote AGE48 top 3** en paper main si confirmé sur N élargi
4. **AGE24/AGE48 unification** → si A1to3 = sweet spot SOL, refactorer les AGE existants
5. ~~Schema migration index `paper_trades`~~ — ✅ DONE May 2 22:30 UTC. Index `idx_paper_trades_strategy_status_created` créé sur `(strategy, status, created_at DESC)` (changement vs todo originale `(token_address, source, status)` qui était sous-optimal — `source` n'a que 2 valeurs distinctes, et `(token_address, status)` existe déjà via `idx_paper_trades_token`). Vérifié EXPLAIN sur audit 6 strats × 14d : index utilisé, 6.0ms pour 1281 rows.

---

## ⏸️ Tâches en suspens (à reprendre quand contexte le permet)

- [ ] **Test BE15_LOCK5 hypothèse slip-sensitivity** (parked v14e.49e) — LOCK 5% très proche entrée → SL fire en cascade sur slip wobble. Si N≥10 et drift > -8pp vs shadow → kill. Si drift ≈ -3pp comme BE25_LOCK10 → garder. **Verdict v14e.50 N=28** : drift médian -2.36pp = 2× LOCK10 (-1.20pp) mais pas disqualifiant. Garder en observation.
- [ ] **Rolling windows feature review** — committed v14e.49i. Revoir `analyze_mega_sweep.py` pour validation user.

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

### KOL routing v14e.49b → v14e.56
- Per-chain blacklist : **17 SOL** (post v14e.56 +TheReaperGems), 3 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
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
