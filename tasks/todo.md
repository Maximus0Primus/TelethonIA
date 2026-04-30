# Pipeline Status — v14e.49e (Apr 30, 2026 19:20 UTC)

## CURRENT — Live Deck SOL (5 strats post v14e.49e swap)

**5 strats live** alloc=1 chacune, position **$1.00/trade** (`max_position_sol=0.012` à SOL $83) :
1. `BE25_TP80_SL30` — baseline BE family
2. `FAST_TP50_SL30` — baseline FAST family
3. `BE25_LOCK10_TP100_SL30_NZ_S40` — seul positif live (+6.6%) sur audit 21h
4. **`FAST45_TP40_SL30_S30`** ★ NEW v14e.49e — WR 48.2% (top du panel), 14d +$11.90/d, N_main=67 paper main calibré
5. **`BE15_LOCK5_TP50_SL30`** ★ NEW v14e.49e — signal fort 14d +$13.65/d / 3d +$115/d, **MAIS LOCK 5% slip-sensitive (test hypothèse user)**. À surveiller en priorité.

`max_open_positions=24` (bumped v14e.49f, était 12 — cap touché 4× en 24h + 16 events ≥10), `daily_loss_limit_sol=0.5`, `min_sol_reserve=0.05`. Exposure max 0.338 SOL (24 × 0.012 + 0.05), wallet 0.56 SOL = 1.66× marge.

**ETH live : DÉSACTIVÉ** (`eth_live_enabled=false`).

**Killed live cette session** :
- `SCALP_TP20_SL10_S30` (v14e.49c) — drift −17.63pp, SL=10% inviable à slip live 822 bps
- `TP200_SL40_2H_NZ_S40` (v14e.49e) — 0/8 TP en 21h, sim −$54/d, dpd_3d aussi neg. Continue paper/shadow.

---

## ✅ Cette session (Apr 30) — toutes complétées

- ✅ **v14e.49 promote 3 SOL shadows → paper main** : `BE15_LOCK5_TP50_SL30` (3d +$108/d), `FAST_TP100_SL20_S35` (3d +$98/d), `BE15_LOCK15_TP80_SL30` (3d +$62/d, WR 51.5%). Bankrolls seedés $1000 chacun, starting_capital $34k → $37k.
- ✅ **v14e.49b kill SOL BSR52** (5 strats) — verdict EARLY confirmé négatif (avg −5.4 à −5.7%, WR 27-33% sur N=33). ETH BSR55 KEPT (5/5 positive, +5 à +19%).
- ✅ **v14e.49b ETH slip recalib** 500/800 → 350/650 (N=10 post-MUSK, empirical 21.5% drag). JSONB + strategies.py.
- ✅ **v14e.49b un-blacklist `CarnagecallsGambles` SOL** — recovered N=2076, avg +4.05%, WR 50.5%, +$596/d shadow. 14 → 13 KOLs banned SOL.
- ✅ **v14e.49c kill SCALP live** + audit drift apples-to-apples (paper/shadow twin, pas companion sim).
- ✅ **v14e.49d bump position 0.006 → 0.012 SOL** ($0.50 → $1.00) pour tester hypothèse "drift slip = position trop petite".
- ✅ **Mega-sweep ETH workflow fix** (e1cb73e) — skip empty shards au lieu de fail le merge. SOL workflow fix mirror.
- ✅ **Sim mega-sweep rolling windows 14d/7d/3d** ajouté (`sim.py` + `analyze_mega_sweep.py`) — pas committé encore (review user).
- ✅ **Forensic v141→v14e.48 SL-blocked** : net delta +$2.62 sur 12 suspects, **pas de recompute requis**.

---

## 🟢 Faisable maintenant — bloqué sur rien

- [ ] **Re-mesurer drift apples-to-apples post-position-bump + 2 nouvelles strats** (24-48h) — si drift converge vers −1pp et buy_slip vers 225 bps, hypothèse position size confirmée. Sinon investiguer (priorité fee, MEV, polling).
- [ ] **Test BE15_LOCK5 hypothèse slip-sensitivity** — LOCK 5% est très proche de l'entrée, attendu : SL fire en cascade sur slip wobble. Si N≥10 et drift > -8pp vs shadow → kill. Si drift ≈ -3pp comme BE25_LOCK10 → garder.
- [x] ~~**Commit rolling windows feature** (`sim.py` + `analyze_mega_sweep.py`)~~ — committed v14e.49i.

---

## 🟡 Wait sur data — N en cours d'accumulation

| Item | N actuel | N cible | ETA |
|---|---:|---:|---|
| BSR_MCAP_AB SOL (4 strats `_BSR_MCAP`) | 11 | 30 | ~7-10j |
| KW_AB SOL (5 strats `_KW34`) | 4 | 30 | ~10j |
| KW_AB ETH (3 strats `_KW26`) | 3 | 30 | ~14j |
| SCORE_V2_AB (rt_score_v2 collecté) | data collection | AUC vs rt_score sur N=30 | ~3j (cible Mai 02) |
| ETH BSR55 (KEPT v14e.49b, 5 strats) | 11 chacune | 30 | ~Mai 7-10 |
| **Live drift re-mesure** post-bump position $1 | 0 | 15 trades/strat | ~24-48h |
| W1-W5 paired-tests (SCALP, AGE, LOCK, ETH winner cluster) | varies | 30 | Mai 03-10 |
| W9b RECALL family verdict par bucket | varies | 30 | Mai 26-Juin 5 |
| T2 sell slip drift twin pairs | varies | 200 | ~Mai 03-05 |

---

## 🔴 User action requise

- [x] ~~**K3** — `jadendegens` triple ban v14e.49g (Apr 30 19:40 UTC)~~ — N=36 ETH paper main WR 0% (p<10⁻¹¹), N=116 SOL paper main avg -30.65%, net 14d -$3,243.
- [x] ~~**K3 part 2** — `aliensalphacalls` ban ETH+live v14e.49h (Apr 30 19:50 UTC)~~ — ETH N=10 WR 0% -$387, SOL live 0/4, SOL shadow saigne -$4.5K. SOL chain déjà banné v14e.37.

---

## 🔧 Backlog (pas urgent)

- [ ] **R2** — Profiler `process_and_push` si lag >30s revient.
- [ ] **T3** — `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.
- [ ] **T8** — Auto-apply JSONB diff via PR si signal stable 4 sem K1/K2.
- [ ] **T9-T10** — Dead-day filter (`_compute_day_regime`) — priorité basse.
- [ ] **T11-T16** — Idées mécaniques nouvelles : DELAY entry, CIRCUIT BREAKER, VOLUME drop exit, LIQ-pull exit, MULTI-KOL confirmation, TIME-based BE.
- [ ] **LOCK polling alignment (parqué)** — si BE25 vérif post-bump confirme `polling_sec=60 + median_5` est le bon réglage, appliquer même override aux LOCK SOL/ETH.

---

## 📌 Rappels persistants (rule-encoded)

### Méthode statistique
- **Paired-test apples-to-apples** sur tokens intersection — JAMAIS aggregate avg quand sample sizes diffèrent.
- **N≥30 par strat** avant verdict, N≥30 par (KOL, chain) avant blacklist reliable (15-29 = probable, observer 1 sem).
- Bootstrap CI 95% + sign test obligatoires sur tout verdict KOL.
- Filtrer artefacts (DTRAIL/DIP/SPLIT/TRAIL) avec `--exclude-artifact-strats`.
- **Drift live** : utiliser paper main twin OU shadow twin via paired-test, **PAS `paper_sim_pnl_pct` companion** (inflate de 2-3×). Leçon v14e.49c.

### Slippage v14e.49b (single source = strategies.py)
- **SOL** : `BUY_SLIPPAGE_BPS = 225` (median empirique 216, valeur OK au global)
- **ETH** : `BUY 350 / SELL 650` (recalibré v14e.49b sur N=10 empirique)
- JSONB miroir : `paper_trade_config.eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`
- Live tx tolerance ETH : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600` (live ≠ paper sim)

### Drift sim ↔ live
- **SOL** : drift global aggregate ~−0.15pp ✓. POST-04-22 BE25 par exit_reason : be_stop −20pp, sl_hit −13pp, timeout stable.
- Drift = fonction directe de `SL_pct` × buy_slip_actual. Strats à SL <15% inviables à position $0.50 (cf. SCALP). À tester à $1.00.

### KOL routing v14e.49b
- Per-chain blacklist active : **13 SOL** (Carnage retiré), 0 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- `mad_apes_gambles` : SOL ban / ETH allow (RELIABLE_WINNER N=104).
- Live KOL blacklist (flat all-chain) : MaestrosDegen, bagcalls, batman_gem, ryoshigamble, ryoshikdegen, venom_gambles.
- KOL whitelist : DISABLED.

### Bankroll
- Total : ~$60K (v14e.49 starting_capital $37K + cumul +$23K).
- SOL live : `max_position_sol = 0.012` = $1/trade. Cap 12 simultané. Exposure max 0.194 SOL.
- ETH live : DÉSACTIVÉ (`eth_live_enabled=false`).

### Mega-sweep
- SOL : cron `02:00 UTC tous les 2 jours`, matrix split 3 shards (jupiter/dexscreener/both).
- ETH : cron `22:00 UTC tous les 2 jours`, single job, fix shard-empty appliqué (v14e.49 commit e1cb73e).
- Persist top-30+50 dans `mega_sweep_runs` Supabase + `_mega_sweep_calibration.py`.

### Strats deck (594 dont 5 SOL BSR52 retirés v14e.49b)
| Family | SOL | ETH |
|--------|-----|-----|
| BE | 33 | 9 |
| LOCK | 33 | 20 |
| FAST | 73 | 18 |
| SLOW | 13 | 5 |
| SCALP | 36 | 17 |
| Other | 80 | 10 |
| AGE clones | 38 | 18 |
| RECALL DIP+PEAK | 45 | 15 |
| BSR_MCAP combo (SOL only) | 4 | 0 |
| KW34/26 (KOL win-rate filter) | 5 | 3 |
| BSR55 ETH only (KEPT v14e.49b) | 0 | 5 |
