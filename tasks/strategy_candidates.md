# Strategy Candidate Tracker

**Goal** : déployer **$50/trade live** avec confiance haute. Construction itérative du shortlist final.

> Living document — mettre à jour après chaque audit, paired-test, ou run mega-sweep.
> Dernière itération : **2026-05-07** (audit complet post-pause-live).

---

## 🎯 Méthodologie — règles d'or

**Avant tout verdict** :

1. **Paper main = shadow** côté calcul PnL (CLAUDE.md confirmé). Pas de privilège méthodologique au paper main. La seule différence : alertes Telegram + éligibilité live.
2. **Apples-to-apples paired-test** sur token intersection — JAMAIS comparer avg% / $/d agrégés entre strats sur des universes ou fenêtres différentes.
3. **N_seuils** : N≥30 par strat = "solide", N≥15 = "probable", N<15 = signal trop faible.
4. **Multi-window stability** : un winner doit performer sur 3j ET 7j ET 14j (ou expliquer la divergence). Une seule fenêtre = potentiel one-off.
5. **Sim ↔ Real cross-validation** : mega-sweep sim peut over-fit moonshots (`TP200_SL40_4H` audited May 7 : sim +$92/d, real -$69/d, drift $161). Toujours valider sim contre paper réel sur la MÊME strat.
6. **Age-band integration** : les filtres d'âge ne sont pas optionnels. Le 0-1h band est un piège massif (cf. §Findings).
7. **KOL blacklist caveat** : la blacklist actuelle (16 SOL + 3 ETH + 6 live) n'est pas garantie optimale. Re-auditer périodiquement (cf. §Open).
8. **Market-day correction** : May 7 today = -$48K total / WR 32% market-wide. Ne pas pénaliser une strat sur la base d'un seul mauvais jour. May 4 (+$39K, WR 64.5%) = inverse upward bias.
9. **KOL-conditioning OBLIGATOIRE** : tout résultat d'audit doit explicitement préciser **quelle blacklist était active**. Une strat évaluée avec la blacklist `_X` peut donner un verdict opposé sous blacklist `_Y`. Les blacklists ne sont pas universelles — un KOL peut être destructeur pour FAST mais profitable pour SLOW (différentes vitesses de pump/dump). Format obligatoire dans les findings :
   > Audit `STRAT_NAME` (window `7j`, blacklist `current=16SOL+3ETH+6live`, N=...).
   - Pour les decisions critiques (promote vers live), faire un **counter-factual blacklist** : la même strat évaluée sans blacklist, ou avec une blacklist alternative — l'edge tient-il ?
   - À l'avenir, considérer des **blacklists par famille** (FAST vs BE vs SLOW) plutôt qu'une seule blacklist globale, si l'analyse counter-factual montre des asymétries fortes.

**Pièges déjà observés (à NE PAS refaire)** :
- ❌ Comparer `AGE3H_X vs X` sur agrégat → 58% WR aggregate, mais paired-test 5/6 strats neutres-négatives. WR était un effet sélection-token, pas valeur du filter.
- ❌ Promote sur 14j window pour une strat promue il y a 3 jours → dilution massive avec pré-promote shadow.
- ❌ Faire confiance au top mega-sweep sim sans cross-check paper réel (`BE15_TP200_SL40_4H` : sim $87/d, real -$93/d, drift $180).

---

## 📊 État data (snapshot May 7 21:00 UTC)

| Métrique | Valeur |
|---|---|
| Total strats registry | 703 |
| Shadow strats actives 14j | ~597 (varie par jour) |
| Last mega-sweep SOL | run_id `25477911459`, 2026-05-07 09:42 UTC, 76 strats, robust 8 |
| Last mega-sweep ETH | run_id `25406889209`, 2026-05-05 22:58 UTC |
| Live status | **PAUSED** depuis 2026-05-02 22:12 UTC |
| Cron crons GH | 7/8 vert, sim-align-gate fix appliqué (commit `d670fba`) |
| Companion-shadow post-promote | ✅ FIX déployé v14e.57 (commit `985a11d`, 2026-05-07 21:02 UTC) |

---

## 🔬 Findings consolidés

### A. Age-band sweet spots SOL (14j, paper+shadow agrégé)

| Band | N | WR | avg pp | Σ $ | Verdict |
|---|---:|---:|---:|---:|---|
| [0-1h] | **48,537** (50% volume) | 38.8% | **-6.88pp** | **-$157,655** | 🔴 **PIÈGE — biggest bleed** |
| **[1-3h]** | 20,248 | **53.1%** | **+4.07pp** | **+$41,873** | 🟢 **SWEET SPOT 1** |
| [3-6h] | 7,319 | 37.7% | -5.11pp | -$17,605 | 🔴 saigne |
| [6-12h] | 6,277 | **31.9%** ⚠️ | **-8.55pp** | -$25,680 | 🔴 **PIRE WR du panel** |
| [12-24h] | 15,355 | 45.2% | -2.43pp | -$17,446 | 🟡 marginal |
| **[24-48h]** | 10,833 | **48.3%** | **+4.13pp** | **+$21,665** | 🟢 **SWEET SPOT 2** |
| [48-72h] | 6,047 | 42.2% | -1.67pp | -$4,537 | 🟡 marginal |

**Implications stratégiques** :
1. Le 0-1h band est responsable de **-$157K** sur 14j à lui seul. Toute strat qui fire massivement sur 0-1h est plombée structurellement.
2. Filtres `min_age=1h` ou `min_age=24h` à explorer prioritairement — pas comme strats nouvelles, mais comme **filtre transversal** sur les top candidats.
3. AGE3H_* prefix testée et **rejetée** (paired-test mai 7) — filter age sans optimisation TP/SL n'apporte rien.

### B. Top 7-day SOL candidates (validated multi-window)

Critères : N≥150, $/d > $40, WR ≥ 48%, med proche 0, shadow_only=true (éligibilité promote).

| # | Strategy | $/d 7j | $/d 3j | WR | med pp | N (7j) | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | **`BE25_LOCK10_TP60_SL30`** | $57 | n/a | **48.5%** | -0.01 | 165 | TOP clean — pas de double, signal stable |
| 2 | **`FAST_TP40_SL30_DS`** | $52 | $20 | **52.1%** | +0.01 | 165 | Best WR du panel (DS source family) |
| 3 | **`FAST_TP50_SL30_BOTH`** | $49 | $42 | 48.5% | -0.02 | 165 | BOTH source family (jp+ds merge) |
| 4 | **`FAST_TP50_SL50`** | $45 | $22 | **49.7%** | -0.01 | 165 | SL=50% lax mais WR maintient |
| 5 | **`FAST_TP50_SL30_JUPITER`** | $44 | $20 | 49.1% | -0.01 | 165 | JUPITER source variant |
| 6 | `TRAIL15_TP50_SL30` | $43 | n/a | 48.5% | -0.01 | 165 | Trailing stop 15% — variant inhabituel |

**Pattern observé** : la famille `FAST_TP50_SL30_{BOTH,JUPITER,DS,NOLAZY}` (A/B sur source quote + polling) montre que **BOTH > JUPITER ≈ DS** par $/d, mais **DS gagne en WR (52.1%)**. Si une seule à promote → **DS** car robustesse > $/d brut.

### C. Source family A/B (paper main candidate selection)

Pattern : ces 4 testent la même baseline (FAST_TP50_SL30) sur même universe avec différentes sources de quote (`strategies.py:2215` : "v144 — Source family A/B: BOTH (merge jp+ds) / JUPITER / DS").

| Source | $/d 7j | WR | med pp | Verdict |
|---|---:|---:|---:|---|
| `_BOTH` (jp+ds merged) | $49 | 48.5% | -0.02 | Highest $/d |
| `_DS` (DexScreener only) | $52 | **52.1%** | **+0.01** | **Best WR + best med** |
| `_JUPITER` (Jupiter only) | $44 | 49.1% | -0.01 | Middle |
| `_NOLAZY` (polling 30s) | $71 | 40.4% | -0.16 | Highest $/d but moonshot |

**Décision pending** : promote `_DS` ou `_BOTH` mais PAS les deux (dedup massif sur même universe).

### D. Score threshold filter `_S35` / `_S40`

Premier finding May 7 — paired-test `_S40` vs baseline a montré edge réel :
- `FAST_TP50_SL30_S40` vs `FAST_TP50_SL30` : N=228, +0.07pp paired, **141/78 v_won (64%)**, +$755/14j
- `BE25_TP80_SL30_S35` vs baseline : N=313, +0.06pp, 147/127 (54%), +$507/14j

**Caveat** : ces strats apparaissent moins haut dans le 7d top — peut-être que leur edge se dilue sur fenêtre courte. À monitorer.

### E. Mega-sweep top sim (May 7) cross-checked vs paper real

| Strat | sim $/d | real $/d | drift | Verdict |
|---|---:|---:|---:|---|
| BE25_LOCK10_TP200_SL40_4H | $92 | **-$69** | $161 | ❌ sim over-fit moonshot |
| BE15_TP200_SL40_4H | $87 | **-$93** | $180 | ❌ idem |
| BE25_TP200_SL40_4H | $81 | **-$91** | $173 | ❌ idem |
| BE50_LOCK25_TP200_SL40_4H | $92 | **-$65** | $157 | ❌ idem |
| **BE25_LOCK15_TP200_SL40_4H_NZ_S40** | $99 | **+$21** | $78 | 🟡 filter NZ_S40 sauve |
| **BE50_LOCK25_TP200_SL40_4H_NZ_S40** | $92 | **+$19** | $73 | 🟡 idem |
| **BE50_LOCK25_TP200_SL40_4H_MCAP** | $92 | **+$15** | $77 | 🟡 filter MCAP sauve |

**Pattern** : la famille `TP200_SL40_4H` est moonshot-driven en sim. Sans filter `NZ_S40` ou `MCAP_S40`, drift 150-180 (catastrophique). **Avec filter** → +$15-21/d real, modeste.

### F. Hypothèses rejetées (May 7)

- ❌ **AGE3H_SOL** (WR 58% aggregate) — paired-test 5/6 strats neutres ou négatives. WR aggregate = effet sélection-token, pas valeur du filter.
- ❌ **BSR55_ETH** — paired-test : avg_diff = 0.01-0.03pp (zéro edge), filter fire au même moment que baseline.
- ❌ **BE15_LOCK5 slip-sensitivity** — drift live↔paper = 0.00pp (N=19), cross-LOCK shadow LOCK5 = même SL_hit rate que LOCK10/15/20.
- ❌ **TP200_SL40_4H sans filter** — sim sur-fit moonshots, real saigne.

---

## 🏆 Classement actuel candidats $50/trade live

> ⚠️ AUCUN candidat n'est encore APPROVED pour $50/trade live. Ce ranking est un **shortlist en construction**.

### Tier S — clean, multi-window stable, prêts pour paper main promote

Profil : WR ≥ 48%, med proche 0, $/d > $30 stable sur 7j, shadow_only=true.

1. **`BE25_LOCK10_TP60_SL30`** — $57/d 7j, WR 48.5%, med -0.01, N=165
   - **Pros** : clean signal, pas de double dans le top, BE25+LOCK10 family validée historiquement
   - **Cons** : pas de TP200 moonshot upside, $/d moyen
   - **À valider** : 3-day window separately (j'ai pas encore le data 3j sur cette strat)

2. **`FAST_TP40_SL30_DS`** — $52/d 7j, WR 52.1%, med +0.01, N=165
   - **Pros** : best WR du panel, DS source = stable quote source, médian POSITIF
   - **Cons** : $/d moins haut que les TP50_SL15_*, source family overlap avec BOTH/JUPITER
   - **À valider** : paired vs `FAST_TP40_SL30_BOTH` et `FAST_TP40_SL30_JUPITER` (head-to-head)

3. **`FAST_TP50_SL30_BOTH`** — $49/d 7j, WR 48.5%, med -0.02, N=165
   - **Pros** : BOTH source = robustness vs single quote source failure, $/d > DS variant
   - **Cons** : WR < DS variant
   - **À valider** : paired vs `FAST_TP50_SL30_DS` à N élargi

### Tier A — high $/d mais moonshot reliance ou small-N

4. `TP50_SL15_NOLAZY` — $71/d 7j MAIS WR 40.4%, med -0.16
5. `TP50_SL15_BOTH` — $70/d 7j, WR 40.4%, med -0.16
6. `BE50_LOCK25_TP200_SL40_4H_NZ_S40` — $55/d 7j, WR 46.9% (sim+real validated)
7. `FAST_TP50_SL30_S40` — $24/d 14j (déjà identifié paired-test win, mais 7j moins clair)

### Tier B — récent, à confirmer (3-day post-promote winners)

8. `FAST60_TP50_SL50_S30` — 3j +$60/d WR 50%, peu de data 7j
9. `BE15_TP70_SL50_NZ` — 3j +$72/d WR 43%, à monitorer
10. `BE25_TP80_SL30_LAZYSLOW` — 3j +$65/d WR 39%, moonshot-flavored

---

## 🧪 Recette de décision promote → live

Le pipeline statistique à appliquer **systématiquement** avant chaque promote (shadow → paper main → live). Une strat qui ne valide pas ces 6 étapes reste en shadow.

1. **N ≥ 30** par strat (verdict intermédiaire) ou **N ≥ 100** (verdict solide). En dessous, c'est du bruit.
2. **Cross-window robustness** : positif sur **14d ET 7d ET 3d**. Si une fenêtre diverge, investiguer (one-off ? regime shift ? promote récent qui dilue ?).
3. **Paired-test apples-to-apples** vs baseline sur **même tokens** (intersection). Jamais comparer aggregate avg/$/d entre strats sur des universes différents.
4. **Bootstrap CI 95%** sur la moyenne du diff paired. Si l'IC inclut 0 → c'est du bruit, on ne promote pas.
5. **Régime stability check** : perf stable jour-par-jour, ou portée par 1-2 outliers ? Test : virer le top 2 trades de la fenêtre, est-ce encore positif ?
6. **Edge minimum vs coûts** : avg pct ≥ **+3%** par trade (couvre coût Solana ~3.5% round-trip : slippage entry + slippage exit + gas). En dessous, l'edge mathématique disparaît dans les frictions.
7. **Blacklist sensitivity** : recompute les étapes 1-6 avec blacklist DÉSACTIVÉE. Si l'edge tient → strat robuste. Si l'edge disparaît → la perf vient de la blacklist, pas de la strat → ne pas promote (ou alors valider que la blacklist actuelle est l'optimale absolue, ce qui est très rarement le cas).

**Ces 7 étapes sont nécessaires mais pas suffisantes** — les 9 critères ci-dessous ajoutent les conditions opérationnelles (companion-shadow drift, KOL filter audit, age-band overlap).

## 🛑 Critères de décision pour $50/trade live

Pour qu'une strat passe en **live $50/trade**, elle doit valider :

1. ✅ **N ≥ 150** sur paper main avec dedup actif
2. ✅ **WR ≥ 48%** stable sur 14j (pas un peak)
3. ✅ **$/d ≥ $30** stable sur 7j ET 14j (pas un seul jour porteur)
4. ✅ **med pp proche 0** (pas moonshot reliance pur)
5. ✅ **Sim ↔ Real drift < $10/d** sur la dernière mega-sweep
6. ✅ **Companion-shadow paired-drift < 5pp** sur 7j post-promote (now possible avec v14e.57)
7. ✅ **Live test : N ≥ 30 trades à $0.50** avant scale to $50, drift contrôlé
8. ✅ **KOL filter audit** : la strat performe sur la blacklist actuelle ET sans (test sensitivity, cf. recette §7). Si fort gap → la strat dépend de la blacklist plutôt que d'avoir un edge intrinsèque. Bonus : tester si une blacklist alternative (par-famille) améliore $/d.
9. ✅ **Age-band overlap** : la strat fire majoritairement sur sweet spots (1-3h ou 24-48h), pas 0-1h massif

**État actuel** : aucune strat ne valide les 9 critères. Plus proche du pack : `BE25_LOCK10_TP60_SL30` valide 1, 2 (à confirmer 14j), 4. Manque 3, 5, 6, 7, 8, 9.

---

## 🚫 KOL blacklists état actuel (snapshot 2026-05-07)

Source : `scoring_config` table, JSONB.

### Live (`rt_trade_config.live_trading.kol_blacklist`) — 6 KOLs all-chain
Bloque le live trading pour ces KOLs sur **toutes les chaînes**, paper main continue à tirer normalement.

```
MaestrosDegen, bagcalls, batman_gem, venom_gambles, jadendegens, aliensalphacalls
```

### Paper chain (`paper_trade_config.kol_chain_blacklist`)

Bloque paper main + shadow par chain. Permet split fin (e.g. ban SOL / allow ETH).

**Solana — 16 KOLs**
```
mad_apes_gambles, papicall, markdegens, ramcalls, leoclub69, ChairmanDN1,
chiggajogambles, bounty_journal, DegenSeals, aliensalphacalls, LevisAlpha,
jadendegens, bagcalls, ryoshikdegen, TheReaperGems, zcallz
```

**Ethereum — 3 KOLs**
```
jadendegens, aliensalphacalls, batman_gem
```

### Paper flat (`paper_trade_config.kol_blacklist`) — non configuré (`null`)

### Splits notables (cross-chain par KOL)

| KOL | SOL | ETH | Live | Note |
|---|---|---|---|---|
| `mad_apes_gambles` | 🔴 ban | 🟢 allow | 🟢 allow | SOL toxique, ETH OK |
| `ryoshikdegen` | 🔴 ban | 🟢 allow | 🟢 allow | idem |
| `bagcalls` | 🔴 ban | 🟢 allow | 🔴 ban | double ban (flat live + chain SOL) |
| `batman_gem` | 🟢 allow | 🔴 ban | 🔴 ban | double ban (flat live + chain ETH) |
| `jadendegens` | 🔴 ban | 🔴 ban | 🔴 ban | triple ban |
| `aliensalphacalls` | 🔴 ban | 🔴 ban | 🔴 ban | triple ban |
| `MaestrosDegen` | 🟢 allow | 🟢 allow | 🔴 ban | live-only ban (no signal en paper) |
| `venom_gambles` | 🟢 allow | 🟢 allow | 🔴 ban | live-only ban (no signal en paper) |
| `batman_gem` | 🟢 allow | 🔴 ban | 🔴 ban | flat live + chain ETH |

### Règle d'usage (extrait MEMORY)

> Flat live ban (`live_trading.kol_blacklist`) ne bloque QUE le live, PAS le paper main. Si stat clearly bad (e.g. WR 1.5% sur N=300 SOL), il faut DOUBLER avec `kol_chain_blacklist.<chain>`.

### À auditer (Q1 ouverte ci-dessous)

- N≥30 fire rate sur chaque banni (réel cost de la ban) ?
- Aucun unban à reconsidérer (signal qui s'améliore) ?
- Aucun KOL allowed qui devrait être banni (paired-test post-7j) ?

Last audit complet : v14e.49g/h/i (2026-04-30) sur jadendegens, aliensalphacalls, ryoshigamble (unban), ryoshikdegen (split), bagcalls, batman_gem.

---

## ❓ Open questions / next iterations

### Q1. KOL blacklist optimale ? (per-strat ?)
État actuel : 16 SOL chain + 3 ETH chain + 6 flat live blacklist (cf. §KOL blacklists état actuel).

**Sous-questions** :
- Re-auditer chaque KOL actuellement banni (paper main fire rate ?, opportunity cost ?)
- Re-auditer chaque KOL actuellement allowed (post-7j, certains saignent ?)
- **Per-strategy blacklist** : la même blacklist est-elle optimale pour FAST et SLOW ? Hypothèse : un KOL "FOMO call" (early pump puis dump) sera destructeur pour FAST_TP50_SL30 (catch le top puis dump, SL fire) mais profitable pour SLOW4H (catch le retracement). Counter-factual : per famille, retirer chaque KOL banni un par un et mesurer le delta $/d.
- **Counter-factual sans blacklist** : pour les top candidats (Tier S), recompute $/d, WR, paired-test avec blacklist DÉSACTIVÉE — l'edge tient-il sans le filter ? Si oui = strat robuste. Si non = la perf vient principalement de la blacklist, pas de la strat.
- **KOL allow-list expérimentation** : au lieu de blacklist (deny by default), tester whitelist (allow by default) sur les top 20 KOLs — gain marginal en signal vs perte en volume ?

**Scripts à créer** :
- `scripts/_kol_blacklist_audit.py` — paired-test KOL-allowed vs KOL-banned trades
- `scripts/_kol_per_strat_breakdown.py` — pour chaque KOL × strat, $/d et WR. Identifie les KOLs qui sont profitables pour certaines strats et destructeurs pour d'autres.
- `scripts/_blacklist_counterfactual.py` — recompute top candidats Tier S avec/sans blacklist active. Output : delta $/d, delta WR, sensitivity score.

### Q2. Filter `_NZ_S40` vs `_MCAP_S40` head-to-head ?
Mega-sweep May 7 montre que les deux sauvent la famille TP200_SL40_4H. Lequel mieux ?
- Paired test sur baseline TP200_SL40_4H (et autres) : N par filter, WR, $/d, med
- Si MCAP_S40 > NZ_S40 → standardiser MCAP_S40

### Q3. Comment intégrer age-band sweet spots dans candidat clean ?
Les strats Tier S ne filtrent pas explicitement sur 1-3h ou 24-48h.
- Test : ajouter `min_age=1, max_age=3` à `BE25_LOCK10_TP60_SL30` → variant `BE25_LOCK10_TP60_SL30_A1to3`
- Test : ajouter `min_age=24, max_age=48` → `BE25_LOCK10_TP60_SL30_A24to48`
- Si l'un des deux variants beat baseline en paired-test → promote ce variant, pas le baseline

### Q4. Resume live deck (4 strats) — quoi en remplacement ?
Live paused depuis May 2. Le précédent deck (4 strats) : BE25_TP80_SL30, FAST_TP50_SL30, TP200_SL40_2H_NZ_S40, BE25_LOCK10_TP100_SL30_NZ_S40.
- Top 3 candidats pour resume : `BE25_LOCK10_TP60_SL30`, `FAST_TP40_SL30_DS`, `BE50_LOCK25_TP200_SL40_4H_NZ_S40`
- Pre-resume : valider companion-shadow post-promote sur 7j (v14e.57 fresh deploy)

### Q5. Source family — qui survit en live ?
`_BOTH/_JUPITER/_DS` paper : BOTH highest $/d, DS highest WR. Mais **en live**, Jupiter Ultra fill prend le dessus (peu importe la source quote choisie pour decision). Le filter source ne change PAS l'exécution live. Donc... la source family est un **artefact paper-only** qui ne survit pas en live. Vérifier cette hypothèse avant promote `_BOTH` ou `_DS` en live.

### Q6. SLOW4H_TP50_SL50 vs SLOW4H_TP50_SL30 ?
SL50 (paper main) : +$42/d post-promote sur 3j
SL30 (shadow) : +$47/d sur 7j, WR 45.7%, med -0.09
- Ces deux sont quasi-identiques côté entries, différent uniquement sur SL%. Lequel survit en moonshot variance ?
- Paired-test à faire.

### Q7. Filtre min_age=1 universel ?
Si 0-1h saigne -$157K/14j (50% volume), pourquoi pas appliquer `min_age=1h` GLOBALEMENT en gate RT ?
- Risk : on rate les early calls sur les vrais moonshots (rare mais existe)
- Trade-off : -$157K loss vs ? upside lost
- À quantifier : sur les wins du 0-1h band, $/winning trade vs same trade fired à 1h (latency penalty)

---

## 📜 Iteration log

### 2026-05-07 (v14e.57 day)
- ✅ Paired-test apples-to-apples 14j top : winner `_S40` filter family ($755/14j paired diff)
- ❌ Hypothèses rejetées : AGE3H_SOL, BSR55_ETH, BE15_LOCK5 slip
- ✅ Companion-shadow post-promote fix déployé (paper_trader.py:1644 v14e.57, commit 985a11d)
- ✅ Sim-align-gate skip propre quand N=0 (commit d670fba)
- ✅ Cleanup 14 backup tables Supabase + 6 search_path + 10 RPC lockdown
- ✅ Mega-sweep cross-check sim↔real : drift catastrophique sur TP200_SL40_4H sans filter
- ✅ Top 7j SOL identifié : BE25_LOCK10_TP60_SL30, FAST_TP40_SL30_DS, FAST_TP50_SL30_BOTH (Tier S)
- ✅ Age-band sweet spots SOL confirmés : [1-3h] et [24-48h]
- ✅ **Combo proposer enrichi (commit `80b7f0c`)** — script `scripts/_propose_combo_extensions.py` étendu avec 8 axes filter (S30/S35/S40/NZ_S40/MCAP/MCAP_S40/A1to3/A24to48). Top 5 mega-sweep `25477911459` → 24 combos générés (vs 0 prior). Bug fix : merge filter parent quand suffix sur base ayant déjà un STRATEGY_FILTERS (e.g. `_NZ_S40_A24to48` garde liq+score). Limitations connues : skip source family `_BOTH/_DS/_JUPITER/_NOLAZY` (need source-routing infra).
- ✅ **5 picks ajoutés à `strategies.py`** (v14e.57 block, lignes ~1060-1110) :
  1. `BE25_LOCK10_TP200_SL40_4H_NZ_S40` — reproduit le pattern NZ_S40 winner ($19/d real attendu)
  2. `BE25_LOCK10_TP200_SL40_4H_MCAP_S40` — alternative MCAP+score
  3. `BE25_LOCK10_TP200_SL40_4H_A1to3` — age band sweet spot 1
  4. `BE25_LOCK15_TP200_SL40_4H_NZ_S40_A24to48` — sweet spot 2 sur top sim already-NZ_S40
  5. `BE50_LOCK25_TP200_SL40_4H_A24to48` — sweet spot 2 sur baseline pure
  Verdict attendu : N≥30 par strat (~5-7j), paired-test post-companion-fix vs leur baseline.
- 🟡 Decision pending : aucun candidat ne valide les 9 critères live $50/trade. Plus proche : `BE25_LOCK10_TP60_SL30`.
- 📌 Next iteration prévue : 2026-05-09 ou 10 (après 2-3j de companion-shadow data accumulée + market normal post-May-7).

---

## 🔗 Liens

- Memory MEMORY.md (condensé) : `~/.claude/projects/.../memory/MEMORY.md`
- TODO opérationnel : `tasks/todo.md`
- Script propose combo extensions : `scripts/_propose_combo_extensions.py`
- Mega-sweep workflow GH : `.github/workflows/mega-sweep-48h.yml` (cron 02:00 UTC tous les 2j)
