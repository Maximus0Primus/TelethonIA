# Pipeline Status — v14e.56c (May 3, 2026, refresh May 7)

## 🎯 Paired-test verdict 14j (May 7, audit après accumulation)

**WINNER: filtre `_S40` (min_score=40)** — battu sa baseline sur 4+ strats apples-to-apples (token intersection) :

| Variant | Baseline | N | diff_pp | v/b won | $ diff 14j |
|---|---|---:|---:|---|---:|
| `FAST_TP50_SL30_S40` | FAST_TP50_SL30 | 228 | +0.07 | 141/78 (64%) | **+$755** |
| `TP50_SL15_S40` | TP50_SL15 | 172 | +0.07 | 91/55 (62%) | +$638 |
| `FAST_TP50_SL30_MCAP_S40` | FAST_TP50_SL30 | 146 | +0.08 | 85/53 (62%) | +$598 |
| `BE25_TP80_SL30_S35` | BE25_TP80_SL30 | 313 | +0.06 | 147/127 (54%) | +$507 |
| `BE25_TP80_SL30_S40` | BE25_TP80_SL30 | 255 | +0.05 | 130/94 (58%) | +$329 |

→ **Promote candidate 1: `FAST_TP50_SL30_S40`** (déjà shadow, faut alloc paper main + bankroll seed).
→ **Promote candidate 2: `BE25_TP80_SL30_S35`** (N=313, dollar-positive, low-variance).

## ❌ Hypothèses testées et rejetées (May 7)

- **AGE3H_SOL (WR 58% aggregate)** — paired-test 6/6 strats : 5/6 neutres-négatives (jusqu'à -0.12pp). WR aggregate = effet sélection-token, pas valeur du filter. **PAS de promote.**
- **BSR55_ETH** — paired-test 5/5 : avg_diff = 0.01-0.03pp (zéro). BSR55 fire au même moment que baseline → coupe juste du volume. Confirme verdict initial todo.
- **BE15_LOCK5 slip-sensitivity** — drift live↔paper = **0.00pp** (N=19, fenêtre 30 avr→1 mai avant pause live). Cross-LOCK shadow : LOCK5 = même SL_hit rate que LOCK10/15/20. **Hypothèse rejetée.**

---

# Pipeline Status — v14e.56c (May 3, 2026)

## 🛑 LIVE PAUSED (May 2 22:12 UTC)

`live_trading.enabled = False`. Backup config : `data/rt_trade_config_pre_pause_20260501T221242Z.json`. Allocations + position size conservés pour reprise rapide quand un winner shadow émerge.

## 🎯 Pattern critique SOL ↔ ETH (chain-aware filters obligatoires)

```
            SOL (14d)         ETH (14d)
[ 0- 1h]    -$45K  saigne     +$5K   positif
[ 1- 3h]    +$10K  sweet 1    -$5K   saigne
[ 3- 6h]    -$32K  saigne     +$7K   sweet 1
[ 6-12h]    -$25K  saigne     +$4K   sweet 2
[12-24h]    -$21K  saigne     -$1K   les deux
[24-48h]    +$28K  sweet 2    -$3K   saigne

Retrade gap [6-12h]: SOL +22% / ETH +31%  (les deux profitable)
```

---

## 🔬 Decision rules — verdict shadow grid (~7-14j, à N≥15 paired)

| Hypothèse | Si confirmée → action |
|---|---|
| `*_A1to3` > baseline +5pp paired | resserrer live filter `min_age=1, max_age=3` SOL |
| `*_A24to48` positif sur 3+ mécaniques | green-light AGE48 family promotion |
| `AGE3H_*` top performer | promote en paper main, candidate live |
| `RECALL_DIP30_AGE6to12_*` +30%+ | confirme sweet spot retrade |
| **`*_A3to12` positif sur 9+ mécaniques** | signal early Δ +7 à +17pp (N=4-8). Si tient à N=15 → promote 3-5 top en paper main. Probablement capture le RECALL window (2e+ entry), pas le 1er call. |

---

## 🟡 Wait sur data (audit May 3 ~14:00 UTC)

| Item | N actuel/strat | Note |
|---|---:|---|
| 51 SOL age-band shadows v14e.52/53 | 4.8-7.3 | Early signal A3to12 positif, A1to3 mixte, A24to48 trop tôt |
| 6 AGE3H_* SOL shadows | 31 | ❌ Paired-test May 7 : 5/6 neutres-négatives, pas de promote |
| 6 RECALL_AGE6to12_* SOL shadows | 5.9 | |
| 12 RECALL_DIP10/ANY SOL shadows | 5.9 | |
| 36 ETH age-band shadows v14e.53b | 2 | ETH KOL volume -75% (Apr 27 24/d → May 3 2/d) |
| 5 AGE6H_ETH_* + 9 ETH_RECALL relaxed | — | Idem ETH volume |
| **BSR_MCAP_SOL (4 strats)** | 49.3 | ✅ **VERDICT KILL** : Δ -12 à -16pp paired (N=65-73) |
| **BSR55_ETH (5 strats)** | 51 | ❌ Paired-test May 7 : avg_diff = 0.01-0.03pp (zéro edge), BSR55 fire au même moment que baseline. **PAS de promote** (confirmé). |
| KW34_SOL / KW26_ETH | 7-8.4 | |
| SCORE_V3_AB | accumule | walk-forward AUC ≥ V1 sur 7-14j (Mai 9-16) |

---

## 🔧 Backlog technique

- [ ] **Fix #2 verb-proximity ticker filter** — `$X (whales|holders|dev) (buying|aping) $Y` patterns. Demande N≥30 messages historiques pour calibrer.
- [ ] **Score V3 walk-forward audit** (Mai 9-16) — sur N≥30 trades fermés post-v14e.55. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter. Script à créer : `scripts/_score_v3_walk_forward.py`.
- [ ] **Kill 4 SOL `_BSR_MCAP`** du `SHADOW_STRATEGIES` (verdict décisif, à appliquer côté code).
- [x] ~~**Companion shadow post-promote**~~ — May 7 v14e.57 : `paper_trader.py:1644` patché. `is_promoted` flag bypass cooldown/open_combos dedup (reserved to real) mais ré-emet le shadow row companion. ML training (train_model.py:766 `.eq("is_shadow",True)`) enrichit son dataset, auto_backtest filtre `.eq("is_shadow",False)` donc pas de double-count. Vérif post-deploy : `SELECT strategy, COUNT(*) FROM paper_trades WHERE strategy IN (<promoted_strats>) AND is_shadow=true AND created_at > <deploy_ts> GROUP BY 1;` doit retourner >0 par strat promue.
- [x] ~~**Test BE15_LOCK5 hypothèse slip-sensitivity**~~ — May 7 : drift live↔paper = 0.00pp à N=19, cross-LOCK shadow LOCK5 = même SL rate. Hypothèse rejetée.
- [ ] **R2** Profiler `process_and_push` si lag >30s revient.
- [ ] **T3** `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.
- [ ] **LOCK polling alignment** — appliquer `polling_sec=60 + median_5` aux LOCK SOL/ETH si BE25 confirme post-bump.

---

## 📋 Backlog post-shadow-verdict (séquencé)

1. **Promote `FAST_TP50_SL30_S40`** en paper main (winner paired-test May 7, N=228, +$755/14j vs base, 64% v_winrate). Alloc + bankroll seed $1000.
2. **Promote `BE25_TP80_SL30_S35`** en paper main (winner paired-test, N=313, +$507/14j, 54% v_winrate, low-variance). Alloc + bankroll seed $1000.
3. **Resume live** une fois 2 promus stables (N≥30 paper main, drift contrôlé sur 7j).
4. ~~Promote AGE48 top 3~~ — paired-test verdict pas convaincant May 7, à re-tester si N grossit.
5. **AGE24/AGE48 unification** → si A1to3 = sweet spot SOL, refactorer les AGE existants.

---

## 📌 Référence rapide

### Méthode statistique
- Paired-test apples-to-apples sur tokens intersection — JAMAIS aggregate avg.
- N≥30 = verdict, N≥15 = "probable" pour décisions intermédiaires.
- **avg% > base ≠ $/d > base** quand un filter coupe le volume. Toujours auditer $/d 7d en plus du paired test.
- Drift live = paper twin paired-test, **PAS `paper_sim_pnl_pct` companion** (inflate 2-3×).
- ⚠️ **Shadow companion s'éteint au promote** : `paper_trader.py:1644` skip toute SHADOW_STRATEGIES dans `real_strats`. Conséquence : les 5 strats promues v14e.54 n'ont **PLUS de shadow companion** depuis May 2. Audit "shadow vs main 3d" sur ces strats compare en réalité **pre-promote vs post-promote** sur des fenêtres temporelles différentes — apples-to-oranges. Pour ces strats, comparer aux strats SHADOW-only (jamais promues) sur les mêmes jours, pas à leur propre shadow gelé.

### Slippage (single source = strategies.py)
- SOL : `BUY_SLIPPAGE_BPS = 225`.
- ETH paper : `BUY 350 / SELL 650`. JSONB miroir : `eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`.
- ETH live tx tolerance : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`.

### KOL routing
- Per-chain blacklist : **18 SOL**, 3 ETH. JSONB `paper_trade_config.kol_chain_blacklist`.
- Splits : `mad_apes_gambles` SOL ban / ETH allow. `ryoshikdegen` SOL ban / ETH allow. `ryoshigamble` flat unban (96.2% WR N=371).
- Double ban (flat + chain) : `bagcalls`, `batman_gem`, `TheReaperGems`, `zcallz`.
- Live KOL flat blacklist : 6 KOLs. Whitelist : DISABLED.

### Mega-sweep
- SOL : cron `02:00 UTC tous les 2 jours`, matrix **6 shards** (3 sources × 2 strat-halves), ~1.5-2h chacun.
- ETH : cron `22:00 UTC tous les 2 jours`, single job (3-shard tient).

### Shadow registry — 676 strats actives
| Family | Count |
|--------|---:|
| BE / LOCK / FAST / SLOW / SCALP | ~188 SOL / 69 ETH |
| AGE clones (AGE24/48/72) | 38 |
| AGE3H_* SOL / AGE6H_ETH_* | 6 / 5 |
| RECALL DIP30/50 + DIP10/ANY + AGE6to12 + ETH relaxed | 22+4 + 12 + 6 + 9 |
| ETH age-bands v14e.53b (A3to6/A6to12/A12to24) | 36 |
| BSR_MCAP SOL (à kill) / BSR55 ETH | 4 / 5 |
| KW34 SOL / KW26 ETH | 5 / 3 |
| Score V3 A/B (col `rt_score_v3`) | walk-forward Mai 9-16 |
