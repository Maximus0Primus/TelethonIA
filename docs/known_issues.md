# Known Issues & Traps

Ce fichier recense les pièges structurels du projet qui se répètent. À lire
avant de (re)lancer une optimisation, ajouter un shadow, ou modifier le sim.

---

## 1. HYST/DS kernels — artefact sim (DO NOT promote raw)

**Règle :** aucune variante suffixée `_HYST` / `_DS` ne peut être promue en
main paper sans **pair-test N≥30 vs base** sur intersection tokens. Les sweeps
sim over-estimate ces kernels parce qu'ils lissent les whipsaw que l'exécution
réelle encaisse.

**Historique :** Apr 20 — 4 shadows (FAST_TP80_HYST, FAST_TP100_HYST,
BE25_TP80_HYST, BE25_TP80_DS) avaient l'air top sur raw stats, paired-test a
révélé −$62 / −$54 / +$6 / $0 vs base +$427 / +$137 / +$191 / +$191.
Retirés en v144.1.

**Exceptions qui marchent :** `BE25_TP80_SL30_S30_HYST` et `BE15_TP70_SL50_NZ`
— HYST **+ filtre** (SCORE30 ou NOZEROLIQ). Le filtre force la sélection, le
kernel ne fait que lisser. À isoler, garder l'œil.

**Référence :** `memory/hyst_artifacts_apr20.md`

---

## 2. DTRAIL / TRAIL / DIP / SPLIT shadows — execution artifact

**Règle :** ne jamais promouvoir un trail/dip/split shadow en live. Le sim
modélise 200 bps sell-slip sur ces stratégies ; le live encaisse 9429 bps
(47×) parce que la `position_reconciler` ferme 50-65% des trades avant que le
trail fire.

**Conséquence :** préférer les stratégies **single-exit** (TP/SL/timeout/BE)
pour Solana memecoins. Les cluster TP200 sont OK parce que TP/SL sont
100% statiques — pas de PATCH nécessaire côté Trigger V2.

**Mitigation sim (v144.13) :** `_mega_family_slip_mult` applique ×10 DTRAIL,
×8 TRAIL, ×6 DIP, ×5 SCALP, ×4 SPLIT dans le mega_sweep. Les rankings
post-v144.13 sont plus honnêtes, mais single-exit reste la voie sûre.

**Référence :** `memory/dtrail_shadow_artifact_apr20.md`

---

## 3. Paired-test obligatoire sur base vs variante

**Règle :** si variante et base ont un sample ratio > 2×, ne JAMAIS comparer
leurs PnL aggregate. Toujours **paired-test sur intersection tokens**.
`pair_N < 10` = verdict prématuré, on attend.

**Historique :** Apr 20 — LAZYSLOW/NOLAZY flaggés "à retirer" sur stats raw
(aggregate), paired-test a renversé le verdict (variante +$1.55-3.80pp).

**Script :** `scripts/paired_all_v144_shadows.py`

**Référence :** `memory/paired_test_rule_apr20.md`

---

## 4. Sim mega_sweep over-estimate trail/dtrail/dip

**Règle :** même avec `_mega_family_slip_mult` (v144.13), les top picks
mega_sweep de la famille trail/dtrail/dip sont suspects. Si la sim dit
"DTRAIL10 rank 1" et le paper/live dit "−5%/j", **croire le paper**. Le live
ne ment pas sur la slippage.

**Verif quotidienne :** `verify_sim_live_alignment.py` (gate) + `sim-align-gate.yml`
(CI 04:00 UTC). Target : avg |diff| ≤ 5pp, within_10pp ≥ 80% APRÈS filtre
MEV-pump (v14e.5).

---

## 5. Per-trade ρ ≠ Per-strategy ρ

**Règle :** préciser le niveau d'aggregation quand on rapporte un Spearman.

- Per-trade ρ ≈ +0.9 (sim prédit bien le PnL d'un trade donné)
- Per-strategy ρ ≈ +0.7 (sim prédit moins bien le ranking des strats)

Les shadows v144.x polluent le ranking per-strategy de −0.10. Exclure avant
de rapporter.

**Référence :** `memory/spearman_metrics_apr20.md`

---

## 6. MEV-pump outliers — expected edge, PAS un bug sim

**Règle :** un trade `tp_hit` où `live_pnl > paper_sim_pnl > 0` avec
`|diff| > 50pp` = Jupiter Ultra positive slippage sur un spike extrême. C'est
**documenté comme edge attendu**, PAS un bug logique.

**Exemples :** `$MHGA` +148pp, `$8` +92.91pp (Apr 23) — live a fill à +235% /
+180% sur le pump, sim a book à +87% (TP strict).

**Filtre appliqué v14e.5 :** `verify_sim_live_alignment.py` tag `[MEV]` et
exclut ces rows du gate metric. Idem `nightly_outlier_monitor.py` v144.19.

**Si le filtre vire rouge :** vérifier d'abord que les MEV-pump ont été
tagged. Puis chercher un vrai bug logique (status opposé, paper > live).

---

## 7. Replay drift sur `be_stop` historique

**Règle :** si le gate flag un diff de +10-18pp sur des trades `be_stop`
anciens (pre-v144.19), c'est probablement une divergence entre le code actuel
de `_evaluate_trade_exit` (BE armed logic) et le paper_sim_pnl_pct stocké à
l'époque. Pas un bug sur la ligne live — le live et le sim stocké matchent
toujours <5pp.

**Fix :** re-run `scripts/backfill_paper_sim_pnl_pct.py` pour resynchroniser
les valeurs historiques avec le code courant. Les trades récents n'ont pas
le problème.

---

## 8. Chain gates (v14e) — règles dures

### Filter level
- Une strategy sans clef `"chain"` dans `STRATEGY_FILTERS` est **implicitement
  Solana-only**. Les strats Solana historiques (FAST_TP50, BE25_TP80…) ne
  peuvent PAS s'ouvrir sur un token 0x.
- Une strategy EVM DOIT déclarer `"chain": "ethereum"|"bsc"|"base"` dans son
  filter. Le naming prefix (`ETH_` / `BSC_` / `BASE_`) est convention, pas
  enforcement.

### Live level
- `live_trader.execute_buy/sell/open_live_trade` rejette tout non-Solana avec
  `_is_solana_mint` check (v14e). Ne retirer ce gate sous aucune condition.
- ETH/BSC/Base live nécessitent `live_trader_eth.py` / `live_trader_bsc.py` /
  `live_trader_base.py` — actuellement `NotImplementedError` stubs. Phase 2
  ETH conditionnée à WR≥65% + EV≥+10%/trade sur N≥50.

### Resolver
- `detect_chain()` shape-based retourne `ethereum` par défaut pour 0x —
  ambigu, car ETH/BSC/Base partagent la même shape. Utiliser
  `resolve_evm_chain(addr)` (DS chainId lookup) quand chain matters (price
  fetch, live).
- Cache RT : `_rt_evm_chain_cache` dans `safe_scraper.py`. Une fois résolu,
  ne reinterroge pas DS.

### Bankroll
- `rt_bankroll.strategy_bankrolls_per_chain` nested. Les buckets EVM ne
  contiennent QUE les strats au bon prefix (enforcement: `jsonb_object_agg
  WHERE key LIKE 'PREFIX_%'`). Si tu vois une strat Solana dans le bucket
  ethereum, c'est un residu du rollback post-cleanup — fix en re-filtrant
  via le pattern SQL de v14e.5.
- Legacy flat `strategy_bankrolls` reste en sync (mirror) pour compat. Prévue
  pour removal au prochain reset bankroll.

---

## 9. LAZY throttling — règles v144.6 + v144.20

**Règle :** le LAZY throttle (180s FAST window, 600s SLOW window) doit être
bypassé pour DEUX populations :

1. **paper rows `entry_source="live_sync"`** (v144.6) — shadow-sync mirror,
   doit avoir la cadence live (30s).
2. **live rows `source="rt_live"`** (v144.20) — les live mains passent par
   `_evaluate_trade_exit` → `_should_evaluate_exit` comme les paper, donc
   étaient throttlées contrairement au comment v144.6. Asymétrie résolue.

Paper **mains** (`source="rt"`, pas `live_sync`) gardent LAZY comme baseline
A/B v144.3 — ne pas toucher.

Ne pas revert ces bypass. Si tu vois des outliers `sync=True` avec divergence
LAZY live>paper (Jupiter wick + 177/180s skip), vérifier que **les deux**
bypass sont bien présents dans `_should_evaluate_exit`. L'A/B 14j / 75 LAZY
live trades a montré que le bypass coûte **$0 net** (noise pur) — le throttle
ne paye pas, il casse juste la coherence sim↔paper↔live.

**Root cause du cas** : pump.fun faible liq (<$100k), Jupiter quote wick
transient (<30s), strat LAZY, poll live à 177s/180s du seuil → eval skippée,
wick disparaît au poll suivant. Le fix évalue le SL à chaque poll 30s comme
la config le déclare (`polling_sec=30`).

---

## 10. `_pt_ultra_override` — retiré v144.19, NE PAS remettre

**Règle :** dans `live_trader._paper_sim_ev`, NE JAMAIS réinjecter un
`_pt_ultra_override` qui ferait que la référence sim stockée suit le fill
Jupiter. Ça contamine `paper_sim_pnl_pct` et casse le gate sim-align avec des
faux drifts +148pp sur les pumps.

La ref sim stockée DOIT rester pure (sans le slippage positif live).

---

## Hygiène générale

- `MEMORY.md` ≤ 200 lignes. Archiver dans `memory/<topic>.md` dès que ça
  approche.
- `todo.md` : section "État actuel" en tête, historique v144.* en bas.
  Corriger les chiffres obsolètes (allocations count, registry partition)
  quand tu touches autre chose.
- Tout commit qui ajoute une stratégie paper doit venir avec son filter
  chain explicite, un seed bankroll, et un update `hybrid_strategy.allocations`
  en DB — sinon la strat ne s'ouvre jamais en RT.

## §11. Routing ideas tested Apr 25 — saved from re-testing

Sur 22,475 trades closed paper since v138.3 reset (Apr 17 → Apr 25, exclu
bat_gamble), 4 idées de routing ont été testées via
`scripts/_analyze_routing_ideas.py`. Résultats :

### ❌ Cooccurrence / multi-KOL confirmation gate
**Hypothesis**: trades where ≥2 KOLs called the same token in last 4h
outperform single-KOL. **Result**: 970 main trades on 95 unique tokens —
quasi-aucun chevauchement entre KOLs détecté (cooccurrence=1 partout).
**Verdict**: invalide sur ce dataset — pas assez d'overlap entre KOLs
pour mesurer. **Re-tester quand** : N tokens > 200 ET overlap >5% entre
KOLs (ou utiliser table `kol_mentions` brute pour fenêtre plus large
que les seuls trades fermés).

### ❌ Per-KOL custom timeout
**Hypothesis**: certains KOLs gagnent/perdent significativement plus tôt/tard
que le timeout 30min. **Result**: les patterns timing existent (BatmanSafuCalls
tp_med=0min, Luca_Apes 46min, TheReaperGems 57min) mais sur les 90 trades
slow-KOL qui ont timeout-out, 49 ferment perdants même avec timeout étendu —
les pumps ne se réveillent pas, ils dump. Δ uplift estimé = +0.3pp seulement.
**Verdict**: ROI marginal, ne pas implémenter avant d'avoir stratégies
elles-mêmes profitables sur les KOLs concernés. **Re-tester quand**: nouvelle
batch d'algos ou si les slow-KOLs deviennent positifs net.

### ✅/❓ Score-band routing : +$3.34/d live (+$1,217/yr) — modeste
**Hypothesis**: l'optimal strat dépend du score band.
**Result naïf** (sans correction sample bias): band <30 → TP50_SL15 +29.2%
vs BE25 +15.2% (Δ +14pp brut, mais N=41 vs N=18, opportunités différentes).
**Apples-to-apples** sur N=52 events intersection
(`scripts/_quantify_routing_dollar_day.py`) : avg +11.34% vs BE25 +9.41%
= **+1.93pp réel** sur les MÊMES tokens/KOLs. Live extrapolation à
$1.74/trade × 99 trades/d : **+$3.34/d → +$1,217/yr** annualisé. Mécanisme
plausible : SL serré (-15%) sur calls low-conviction évite drawdown long ;
TP +50% suffit car les pumps score<30 sont rarement >+50%.
**To implement**: nouvelle strat `LOWSCORE_TP50_SL15` (filter
`max_score=29`) ajoutée aux allocs.
**Caveat N**: N=52 trop petit pour IC serré — re-confirmer après 2-3
semaines de data. Le mega-sweep workflow re-runnera
`_quantify_routing_dollar_day.py` chaque 48h pour tracker la dérive.

### ❌ Per-KOL strategy specialization : NEGATIVE en apples-to-apples (corrigé Apr 25)
**Hypothesis**: certains KOLs préfèrent une famille non-default.
**Result naïf**: 23/29 KOLs avec Δ ≥ 3pp sur strat preference, smart-route
moyenne +15.6% vs BE25 baseline +1.8% (Δ +13.7pp sur N=2011 vs N=1742).
**MAIS gros biais d'échantillonnage** — les 2 sets portaient sur
opportunités DIFFÉRENTES (le smart-route excluait les KOLs sans préférence
claire = ceux qui sont les KOLs neutres / fallback BE).
**Apples-to-apples** sur N=52 events où les 2 scénarios sont calculables
sur les MÊMES (KOL, token) : routing +8.53% vs BE25 +9.41% =
**−0.88pp (NEGATIVE), live extrapolation = −$1.51/d soit −$551/yr**.
**Verdict**: la "préférence KOL" était dominée par les events où BE25 est
déjà le best-fit. Effet réel <1pp, probablement structurellement insignifiant.
**Re-tester quand**: N>150 events apples-to-apples (~3 semaines plus de data).
Ne PAS implémenter de table `kol_optimal_family` avant confirmation N≥150.

