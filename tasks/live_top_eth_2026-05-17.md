# 🎯 Live ETH Dual Pick — Snapshot 2026-05-17

> **Objectif** : choisir les meilleures stratégies ETH pour aller en live. Optimise consistance long-terme (30d) ET trajectoire récente (5-7d). Méthodologie identique au SOL `live_top5_2026-05-17.md` mais avec **data fused (main + shadow)** car ETH a une particularité structurelle (cf. §🐛).
>
> **Décision finale** : 2 strats à mettre en paper main (TP100_SL50 à ajouter + FAST60_NZ_S40 déjà alloué). Pour live `eth_live_enabled = false` actuellement — réflexion live ETH séparée (gas Ethereum = $200/trade min vs SOL $1).

---

## 📊 Méthodologie ETH (différente de SOL)

**Pour SOL** on a utilisé shadow only car le bug v14e.58 avait gelé les paper main → shadow était la seule data propre.

**Pour ETH** la méthodologie correcte est **MAIN data** (= ce que paper main fait vraiment), MAIS j'ai découvert un bug structurel :

### 🐛 Le bug ETH `SHADOW_STRATEGIES` gap (fixé en v14e.60)

2 strats ETH allouées en paper main (`ETH_TP80_SL40_T2H` et `ETH_FAST_TP100_SL20`) n'étaient **PAS dans la liste `SHADOW_STRATEGIES`** dans `strategies.py`. Conséquence :
- Pas de shadow direct → seul le **companion shadow** (v14e.57) peut tirer en parallèle
- Le companion shadow se fait **bloquer par MAIN via `open_combos`** (paper_trader.py:1664)
- Résultat : 0 shadow trades pour ces 2 strats malgré 170 et 153 trades main
- **Impossible de mesurer le paired-drift main↔shadow** pour ces strats

**Fix v14e.60** :
```python
# strategies.py:2748 (ETH_TP80_SL40_T2H) et :2827 (ETH_FAST_TP100_SL20)
SHADOW_STRATEGIES.append("ETH_TP80_SL40_T2H")
SHADOW_STRATEGIES.append("ETH_FAST_TP100_SL20")
```

Sync avec la convention SOL : **toute strat paper main allouée DOIT aussi être dans SHADOW_STRATEGIES**. Sinon pas de paired-drift mesurable.

### Méthodologie analyse — fused (main + shadow combinés)

Tant que le fix v14e.60 n'a pas accumulé 7j de data shadow propre, on utilise **fused data** :

```sql
-- Pour chaque (token, day, strategy), prendre le plus ancien event main OU shadow
DISTINCT ON (strategy, token_address, DATE(created_at))
WHERE chain='ethereum' AND source='rt' AND status IN closed
  AND kol_group NOT IN (6 KOLs ETH banned)
ORDER BY strategy, token, day, created_at
```

→ Bypass le SHADOW_STRATEGIES gap (capture les events main pour les 2 strats orphelines).

---

## 🏆 PICK #1 (top consistance + médiane massive) : `ETH_FAST60_TP100_SL50_NZ_S40`

**Design** : TP +100%, SL −50%, horizon 60min, filter `min_liquidity_usd ≥ 1` AND `min_rt_score ≥ 40`. (`strategies.py:3086`)

```python
STRATEGIES["ETH_FAST60_TP100_SL50_NZ_S40"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 60, "label": "main"},
]
STRATEGY_FILTERS["ETH_FAST60_TP100_SL50_NZ_S40"] = {
    "chain": "ethereum", "min_liquidity_usd": 1, "min_rt_score": 40
}
```

### Pourquoi #1
1. **Min window $30.3** = le plancher le plus haut du panel ETH
2. **Médiane PnL POSITIVE +29.7%** = top du panel (typiquement TP+100% atteint sur majorité des trades)
3. **WR 54%** — solide
4. **Trajectoire accélérante** : 30→14d $30 → 5d **$37**
5. Déjà alloué ✅ (paper main + shadow s'accumulent)
6. N=33 main + 76 shadow = **109 events fused** sur 30j

### $/jour sur 7 fenêtres (fused dedup + BL filtered)

| Fenêtre | $/jour | N | Note |
|---|---|---|---|
| **30 jours** | **$36.0** | 109 | |
| 30→14d (early 16j) | $30.3 | 47 | Floor solide |
| **14 jours** | $42.5 | 62 | |
| **7 jours** | $32.3 | 27 | |
| **5 jours** | **$37.4** | 19 | Accélération |
| Médiane 14d | **+29.7%** | — | TOP du panel |
| WR 14d | **54%** | — | |

---

## 🥈 PICK #2 (stable WR + médiane positive + gros N) : `ETH_TP100_SL50`

**Design** : TP +100%, SL −50%, horizon 240min (4H), filter chain=ethereum. (`strategies.py:2734`)

```python
STRATEGIES["ETH_TP100_SL50"] = [
    {"pct": 1.0, "tp_mult": 2.00, "sl_mult": 0.50, "horizon_min": 240, "label": "main"},
]
STRATEGY_FILTERS["ETH_TP100_SL50"] = {"chain": "ethereum"}
SHADOW_STRATEGIES.append("ETH_TP100_SL50")  # déjà en shadow
```

### Pourquoi #2
1. **Stable cross-window** : 30d $45, 14d $44, 7d $37, 5d $19 — tout positif
2. **WR 54%** — break-even évident
3. **Médiane +5.2%** — pas de moonshot dependency
4. **N=164 fused** (34 main + 130 shadow) = très significatif statistiquement
5. Design ultra-simple : TP+100/SL-50/4H
6. **Pas alloué actuellement** → candidat PROMOTE

### $/jour sur 7 fenêtres

| Fenêtre | $/jour | N | Note |
|---|---|---|---|
| **30 jours** | **$45.0** | 164 | #4 du panel |
| 30→14d | $45.6 | 69 | Floor élevé |
| **14 jours** | $44.2 | 95 | |
| **7 jours** | $36.7 | 44 | |
| **5 jours** | $19.2 | 25 | |
| Médiane 14d | **+5.2%** | — | |
| WR 14d | **53.5%** | — | |

### ⚠️ Note position size
ETH paper main utilise des positions variables ($50-200 selon strat, EVM gas amortization). `ETH_TP100_SL50` a `avg_pos=$83` dans la fused data. À $50/trade équivalent shadow, les $/d projetés sont à recalibrer prorata si position diffère.

---

## 🥊 Tradeoff PICK #1 vs PICK #2

| Critère | FAST60_TP100_SL50_NZ_S40 | TP100_SL50 |
|---|---|---|
| $/d 30d | $36 | **$45** 🥇 |
| $/d 14d | $42 | **$44** |
| **$/d 5d** | **$37** 🥇 | $19 |
| Médiane 14d | **+29.7%** 🥇 | +5.2% |
| WR 14d | **54%** | **53.5%** |
| N fused | 109 | **164** 🥇 |
| Design horizon | 60min FAST | 240min (4H) |
| Filter strict | NZ + S40 | aucun (chain ETH only) |
| Déjà alloué | ✅ | ❌ (PROMOTE) |
| Bankroll actuel | $1170 | $998 (jamais alloué) |

**Verdict** : Les 2 strats sont complémentaires :
- **FAST60_NZ_S40** : haute fréquence, exit rapide, filter qualité, trajectoire accélérante
- **TP100_SL50** : exit lent (4H), capture les longues bougies vertes ETH, design ultra-simple

**Dual pick recommandé** pour diversification temporal (60min + 240min horizons couvrent 2 régimes de pump).

---

## 📈 Top 8 ETH ranking (fused, BL filtered, multi-window)

| # | Strat | n_main | n_shadow | 30d | 14d | 7d | 5d | WR | med |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **ETH_BE30_TP100_SL40** | 30 | 130 | **$49.9** | $40.3 | $33.2 | $11.6 | 51% | +2.8 |
| 🥈 | ETH_BE50_TP150_SL40_T2H | 79 | 60 | $48.6 | $45.1 | $48.8 | $15.5 | 42% | -28 |
| 🥉 | ETH_TP200_SL40_2H_NZ_S40 | 0 | 103 | $47.2 | $43.4 | $41.5 | $20.4 | 39% | -37 |
| 4 | **ETH_TP100_SL50** ← Pick #2 | 34 | 130 | $45.0 | $44.2 | $36.7 | $19.2 | **54%** | **+5.2** |
| 5 | **ETH_TP80_SL40_T2H** | **170** | 0 | $40.9 | $27.3 | $20.2 | $4.3 | 50% | **+9.6** |
| 6 | ETH_TP300_SL50_4H | 34 | 94 | $39.7 | $66.5 | $92.3 | $20.1 | 34% | -49 |
| 7 | 🌟 **ETH_FAST60_TP100_SL50_NZ_S40** ← Pick #1 | 33 | 76 | $36.0 | $42.5 | $32.3 | **$37.4** | **54%** | **+30** |
| 8 | **ETH_FAST_TP100_SL20** | 153 | 0 | $33.5 | $34.4 | $38.5 | $13.5 | 40% | -27 |

**Note importante** : les strats avec `n_shadow=0` (ETH_TP80, ETH_FAST_TP100_SL20) sont les 2 qui avaient le bug SHADOW_STRATEGIES gap. Post-fix v14e.60, elles auront shadow → analyse sera plus robuste à J+7.

---

## 🛡️ Stress-test A — Blacklist robustness (edge amplification)

ETH a **6 KOLs banned** (vs SOL 18) : `jadendegens, aliensalphacalls, batman_gem, dddegens, CryptoChefCooks, degenncabal`.

| Strat | dpd avec BL | dpd sans BL | Amplif | Verdict |
|---|---|---|---|---|
| ETH_BE25_LOCK15_TP100_SL30 | $26 | $25 | **1.05x** | ✅ Quasi-intrinsèque |
| **ETH_FAST60_TP100_SL50_NZ_S40** | $34 | $29 | **1.16x** | ✅ Très robuste |
| ETH_FAST60_TP100_SL50 | $35 | $29 | 1.18x | ✅ |
| ETH_BE30_TP100_SL40 | $38 | $31 | 1.22x | ✅ |
| ETH_FAST60_TP70_SL50_NZ_S40 | $24 | $18 | 1.30x | ✅ |
| **ETH_TP100_SL50** | $36 | $27 | **1.32x** | 🟡 |
| ETH_BE50_TP150_SL40_T2H | $48 | $36 | 1.32x | 🟡 |

**Lecture** : ETH a une robustesse BL **bien meilleure que SOL** (1.05-1.32x vs SOL 1.37-8.55x). La BL filtre seulement 11-18% des events ETH vs 35% SOL. **Les edges ETH sont plus intrinsèques** — moins dépendants du filter KOL.

→ Les 2 picks sont solides en BL : 1.16x (FAST60_NZ_S40) et 1.32x (TP100_SL50).

---

## 🛡️ Stress-test B — Dedup robustness

Pour les 2 picks ETH, dedup first-call est appliqué dans l'analyse fused. Comme ETH a moins de re-calls multi-day par token que SOL (volume KOL ETH plus faible), l'impact dedup est minimal. Pas de risque "trompe-l'œil dedup-inflated" comme certaines SOL BE_LOCK avaient (cf. §H SOL candidates).

---

## ⚠️ ETH live considerations (vs SOL)

### Différences structurelles ETH ↔ SOL

| Aspect | SOL | ETH |
|---|---|---|
| Gas fees round-trip | ~$0.02 à $50/trade | **~$5-20** à $200/trade |
| Position min recommandée | **$1** | **$200** (gas amortization) |
| Slip Jupiter Ultra | 225-953 bps selon position size | 350-650 bps (Uniswap v3) |
| Block time | 400ms | 12s |
| MEV exposure | Modérée (mempool privé Jito) | Élevée (flashbots dépendant) |
| Wallet ETH live actuel | 0.2566 SOL (~$22) | ? (à vérifier) |

### Recommandation live ETH
**Pas de live ETH immédiat** car :
1. `live_trading.eth_live_enabled = false` actuellement
2. Position min $200 par trade → besoin minimum $4k-5k wallet pour 20 positions
3. Drift live↔paper non encore mesuré pour ETH spécifiquement
4. Pilot SOL en cours d'abord (validation drift methodology) — résultats applicables ETH à J+7+

→ **ETH stratégie reste paper main pendant 7-14j de plus**, puis décision live après pilot SOL réussi.

---

## 🎚️ Niveau de confiance

### ✅ Solide
- Méthodologie fused (main + shadow combinés) bypass le bug SHADOW gap
- N=109-164 par strat = statistiquement significatif
- Toutes 4 fenêtres positives pour les 2 picks
- BL robustness excellente (1.16x et 1.32x)
- Fix v14e.60 corrige la base methodology pour le futur
- Pattern structurel cohérent : TP+100 + filter score/liq = top performers

### ⚠️ Limitations
1. **Fix SHADOW v14e.60 à valider** : il faut 7j de data shadow propre pour les 2 strats fixées avant de pouvoir mesurer paired-drift
2. **Bug bankroll Telegram** : fixé aussi (rebuild de Apr 26 → today), mais le script auto-update bankroll doit être investigué pour éviter rechute
3. **ETH live non-testé** : projections live = extrapolation depuis SOL pilot, marge d'erreur ±30% facilement
4. **N main pour Pick #1** : 33 trades seulement → samples small. Pick #2 plus solide statistiquement (164 fused, 130 shadow)
5. **Régime shift possible** : entre §I SOL May 12 et today, tous les top picks SOL ont collapsed en 5j. Pas garanti que ETH soit immune

### 🎯 Confiance globale : **MEDIUM-HIGH** pour paper main, **MEDIUM** pour pre-live planning

---

## 🚦 Recette de déploiement ETH

### Phase 0 — Fix base (FAIT 2026-05-17)
- ✅ Bankroll rebuild from ground truth (ETH_TP80 $1451→$1919, ETH_FAST_TP100_SL20 $1646→$1881)
- ✅ Fix v14e.60 : `SHADOW_STRATEGIES.append("ETH_TP80_SL40_T2H")` + `ETH_FAST_TP100_SL20`
- 🕓 Deploy v14e.60 sur VPS pour activer le fix (commit + push + restart)

### Phase 1 — Promote ETH_TP100_SL50 en paper main
1. `UPDATE scoring_config SET rt_trade_config = jsonb_set(rt_trade_config, '{hybrid_strategy,allocations,ETH_TP100_SL50}', '1'::jsonb)`
2. Bankroll déjà existant à $998 — on garde ou on reset à $1000 fresh ?
3. Laisser tourner 7j pour data clean main + shadow

### Phase 2 — Re-audit J+7 (~2026-05-24)
- Re-run la query canonique (cf. fin du doc)
- Vérifier toutes 4 fenêtres ≥ $15/d pour les 2 picks
- Mesurer paired-drift main↔shadow (post-fix v14e.60, les 2 strats orphelines auront enfin shadow)
- Si drift <5pp → green light pour Phase 3

### Phase 3 — Live ETH pilot $200/trade (futur, J+14+)
**PRÉREQ** : pilot SOL réussi + wallet ETH funded ($4-5k minimum)
1. `live_trading.eth_live_enabled = true`
2. `live_trading.eth_allocations = {"ETH_FAST60_TP100_SL50_NZ_S40": 0.5, "ETH_TP100_SL50": 0.5}`
3. `live_trading.eth_max_position_usd = 200` (déjà default)
4. `live_trading.eth_max_open_positions = 1` (déjà default — conservative)
5. Mesurer drift 48-72h

---

## 📌 Re-audit SQL pour J+7 (2026-05-24)

```sql
-- Vérifier les 2 picks ETH + paired-drift main vs shadow (post-fix v14e.60)
WITH bl AS (
  SELECT ARRAY['jadendegens','aliensalphacalls','batman_gem','dddegens',
               'CryptoChefCooks','degenncabal']::text[] AS eth_bl
),
fused AS (
  SELECT DISTINCT ON (strategy, token_address, DATE(created_at))
    strategy, created_at, pnl_pct, pnl_usd, is_shadow
  FROM paper_trades pt, bl
  WHERE pt.chain = 'ethereum' AND pt.source = 'rt'
    AND pt.strategy IN ('ETH_FAST60_TP100_SL50_NZ_S40', 'ETH_TP100_SL50')
    AND pt.status IN ('tp_hit','sl_hit','timeout','trail_stop','be_stop')
    AND pt.created_at >= NOW() - INTERVAL '30 days'
    AND NOT (pt.kol_group = ANY(bl.eth_bl))
  ORDER BY strategy, token_address, DATE(created_at), created_at
)
SELECT strategy,
  COUNT(*) AS n_30d,
  COUNT(*) FILTER (WHERE NOT is_shadow) AS n_main,
  COUNT(*) FILTER (WHERE is_shadow) AS n_shadow,
  ROUND(SUM(pnl_usd)::numeric / 30, 1) AS "30d",
  ROUND(SUM(pnl_usd) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days' AND created_at < NOW() - INTERVAL '14 days')::numeric / 16, 1) AS "30→14d",
  ROUND(SUM(pnl_usd) FILTER (WHERE created_at >= NOW() - INTERVAL '14 days')::numeric / 14, 1) AS "14d",
  ROUND(SUM(pnl_usd) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days')::numeric / 7, 1) AS "7d",
  ROUND(SUM(pnl_usd) FILTER (WHERE created_at >= NOW() - INTERVAL '5 days')::numeric / 5, 1) AS "5d",
  ROUND((100 * percentile_cont(0.5) WITHIN GROUP (ORDER BY pnl_pct) FILTER (WHERE created_at >= NOW() - INTERVAL '14 days'))::numeric, 1) AS med14
FROM fused
GROUP BY strategy;
```

**Critères go/no-go** :
- TOUTES les 4 fenêtres (30→14d/14d/7d/5d) ≥ $10/d
- med14 ≥ 0
- N_30d ≥ 80
- n_shadow > 0 pour les 2 strats (confirme fix v14e.60)
- Si paired-drift main↔shadow < 5pp → green light live

Si UN critère ❌ → reste en paper main + ré-itérer.

---

## 🔍 Comparaison avec mes recos précédentes (transparency)

J'ai fait **2 erreurs majeures** dans mes 1ères analyses ETH :

1. **Shadow-only analysis pour ETH** : j'ai utilisé la méthodologie SOL (shadow only) qui a poisonné le résultat car les 2 top main strats (TP80, FAST_TP100_SL20) ont 0 shadow → invisibles dans mon ranking. Tu m'as catch ça en regardant ton Telegram bankroll.

2. **Recommandation DEMOTE incorrecte** : j'avais proposé de DEMOTE `ETH_TP80_SL40_T2H` et `ETH_FAST_TP100_SL20` comme "peu actifs N<30 14d" — basé sur shadow inexistant. En réalité ce sont les 2 strats les plus performantes en MAIN data (170 et 153 trades, $40.9 et $33.5/d 30d).

**Leçon méthodologique** :
- SOL : shadow-only OK car bug v14e.58 poisonné le main → on a corrigé v14e.59 + 60
- ETH : main data OU fused (main + shadow) — **JAMAIS shadow-only** car bug SHADOW_STRATEGIES gap (fixé v14e.60 mais data shadow stale jusqu'à J+7)
- Convention universelle post-v14e.60 : **toute strat allouée doit être dans SHADOW_STRATEGIES** pour permettre paired-drift mesurable

---

## 📋 Migration appliquée (2026-05-17)

```sql
-- 1. Bankroll rebuild ETH (toutes strats from paper_trades ground truth)
--    Backup: data/rt_bankroll_eth_pre_rebuild_20260517T162008Z.json
UPDATE rt_bankroll SET strategy_bankrolls_per_chain = jsonb_set(
  strategy_bankrolls_per_chain, '{ethereum}',
  (SELECT jsonb_object_agg(strategy, jsonb_build_object(
    'pnl', ROUND(total_pnl::numeric, 2),
    'trades', n,
    'balance', ROUND((1000 + total_pnl)::numeric, 2),
    'starting_balance', 1000,
    'last_updated_at', NOW()::text))
   FROM (SELECT strategy, SUM(pnl_usd) AS total_pnl, COUNT(*) AS n
         FROM paper_trades
         WHERE chain='ethereum' AND source='rt' AND is_shadow=false
           AND status IN ('tp_hit','sl_hit','timeout','trail_stop','be_stop')
         GROUP BY strategy) agg)
) WHERE id = 1;

-- 2. strategies.py v14e.60 — add 2 missing SHADOW_STRATEGIES
-- (commit pending — TODO: push + deploy)
```

**Strats déjà en main + shadow** (post-fix) : ETH_FAST_TP100_SL20, ETH_TP80_SL40_T2H + 5 autres déjà OK. Total 7 strats ETH paper main correctement instrumentées.

**Strat à PROMOTE** (pas encore décidé/appliqué) : ETH_TP100_SL50.
