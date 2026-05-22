# Strategy Candidate Tracker

**Goal** : déployer **$50/trade live** avec confiance haute. Construction itérative du shortlist final.

> Living document — mettre à jour après chaque audit, paired-test, ou run mega-sweep.
> Dernière itération : **2026-05-12** (collapse Tier S + bug v14e.57 cooldown fix + nouveau ranking filter-rich).

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

## 🚫 KOL blacklists état actuel (snapshot 2026-05-07 post-audit Option B)

Source : `scoring_config` table, JSONB.

### Live flat (`rt_trade_config.live_trading.kol_blacklist`) — 4 KOLs all-chain
Bloque le live sur **toutes les chaînes**, paper main continue normalement.

```
bagcalls, batman_gem, jadendegens, aliensalphacalls
```

### Live chain (`rt_trade_config.live_trading.kol_chain_blacklist`) — NEW v14e.57
Per-chain (mirror du paper). Permet split fin en live.

**Solana — 1 KOL** (`venom_gambles` SOL ban / ETH allow)
```
venom_gambles
```

### Paper chain (`paper_trade_config.kol_chain_blacklist`)

Bloque paper main + shadow par chain. Permet split fin (e.g. ban SOL / allow ETH).

**Solana — 18 KOLs** (audit 2026-05-12 PM : +CarnagecallsGambles)
```
mad_apes_gambles, papicall, markdegens, leoclub69, ChairmanDN1,
DegenSeals, aliensalphacalls, LevisAlpha, jadendegens, bagcalls,
ryoshikdegen, TheReaperGems, zcallz, venom_gambles,
robo_gambles, certifiedprintor, CSCalls, CarnagecallsGambles
```

> **Historique audit blacklist SOL** :
> - 2026-04-30 (v14e.49b) : `CarnagecallsGambles` UNBANNED (was +$596/d 7d at the time)
> - 2026-05-07 (v14e.57) : `venom_gambles` ADDED (paper SOL toxic -$575/d, split allow ETH)
> - 2026-05-12 audit : UNBAN `bounty_journal` (+$2578/d 7d, +$497/d 30d, WR 58.8%, N=4528 — top performer caché), UNBAN `chiggajogambles` (+$1019/d 7d récent flip), UNBAN `ramcalls` (14d +$459/d WR 96.3% watchful). BAN `robo_gambles` (WR 0%, med −57.6%), `certifiedprintor` (WR 1.6%, med −48.3%), `CSCalls` (WR 1.0%, med −22.5%).
> - 2026-05-12 PM : BAN `CarnagecallsGambles` (re-ban après Apr 30 unban). 30d −$1746/d N=10217 = top $ damage panel. Backup `_backup_blacklist_audit_20260512`.

**Ethereum — 6 KOLs** (audit 2026-05-12 : +3 catastrophic bans)
```
jadendegens, aliensalphacalls, batman_gem,
dddegens, CryptoChefCooks, degenncabal
```

> **Historique audit blacklist ETH** :
> - 2026-05-12 audit : BAN `dddegens` (N=368 30d, −$68/d 30d, **−$199/d 7d worsening**, med −37, WR 32%). BAN `CryptoChefCooks` (N=84, **WR 0%**, med −25). BAN `degenncabal` (N=63, **WR 0%**, med **−42**). Les 3 catastrophiques de la list ETH allowed.
> - `aliensalphacalls` ETH : N=14 trop petit (vs WR 78.6% sur 30d) — KEEP ban watchful jusqu'à N≥30 propre.

### Paper flat (`paper_trade_config.kol_blacklist`) — non configuré (`null`)

### Splits notables (cross-chain par KOL, post-audit 2026-05-07)

| KOL | Paper SOL | Paper ETH | Live SOL | Live ETH | Note |
|---|---|---|---|---|---|
| `mad_apes_gambles` | 🔴 ban | 🟢 allow | 🟢 allow | 🟢 allow | SOL toxique, ETH OK |
| `ryoshikdegen` | 🔴 ban | 🟢 allow | 🟢 allow | 🟢 allow | idem |
| `bagcalls` | 🔴 ban | 🟢 allow | 🔴 ban | 🔴 ban | flat live + chain SOL |
| `batman_gem` | 🟢 allow | 🔴 ban | 🔴 ban | 🔴 ban | flat live + chain ETH |
| `jadendegens` | 🔴 ban | 🔴 ban | 🔴 ban | 🔴 ban | banned everywhere |
| `aliensalphacalls` | 🔴 ban | 🔴 ban | 🔴 ban | 🔴 ban | banned everywhere |
| **`venom_gambles`** v14e.57 | 🔴 ban | 🟢 allow | 🔴 ban | 🟢 **ALLOW** | SOL banned 4× / ETH allowed (top performer +$93/d WR 80.8%) |

### Règle d'usage (extrait MEMORY)

> Flat live ban (`live_trading.kol_blacklist`) ne bloque QUE le live, PAS le paper main. Si stat clearly bad (e.g. WR 1.5% sur N=300 SOL), il faut DOUBLER avec `kol_chain_blacklist.<chain>`.

### À auditer (Q1 ouverte ci-dessous)

- N≥30 fire rate sur chaque banni (réel cost de la ban) ?
- Aucun unban à reconsidérer (signal qui s'améliore) ?
- Aucun KOL allowed qui devrait être banni (paired-test post-7j) ?

Last audit complet : v14e.49g/h/i (2026-04-30) sur jadendegens, aliensalphacalls, ryoshigamble (unban), ryoshikdegen (split), bagcalls, batman_gem.

### Audit live blacklist 2026-05-07 (paper data 14j, source kol_group)

> ⚠️ Pour les KOLs aussi en `paper_chain_blacklist`, les trades observés sont **shadow telemetry uniquement** (paper main bloqué, mais shadow continue). Quantifie ce que coûterait un unban.

| KOL | Chain | N | WR | avg pp | sum_usd 14j | $/d | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `aliensalphacalls` | SOL | 1308 | 33.6% | -15.6% | **-$9,710** | -$694 | ✅ ban JUSTIFIÉ massif |
| `aliensalphacalls` | ETH | 24 | 45.8% | +5.4% | -$226 | -$16 | 🟡 ETH marginal (avg+ mais sum-) |
| `bagcalls` | SOL | 339 | **2.9%** | -4.4% | -$722 | -$52 | ✅ ban JUSTIFIÉ (1 token spam'd 339×) |
| `batman_gem` | ETH | 366 | 31.1% | -11.6% | **-$2,116** | -$151 | ✅ ban JUSTIFIÉ |
| `batman_gem` | SOL | 22 | **0.0%** | -38.8% | -$427 | -$30 | ✅ ban JUSTIFIÉ (N petit mais 0% WR) |
| `jadendegens` | SOL | **4,521** | 35.1% | -10.5% | **-$22,570** | **-$1,612** | ✅ ban ULTRA JUSTIFIÉ (top spam-toxic) |
| `jadendegens` | ETH | 158 | 15.2% | -29.4% | -$3,218 | -$230 | ✅ ban JUSTIFIÉ |
| `venom_gambles` | SOL | 344 | **0.0%** | -48.3% | **-$8,045** | -$575 | ✅ ban JUSTIFIÉ (perte massive) |
| **`venom_gambles`** | **ETH** | **73** | **80.8%** | **+35.8%** | **+$1,305** | **+$93** | 🚨 **BAN INJUSTIFIÉ ETH — top performer !** |
| `MaestrosDegen` | both | **0** | n/a | n/a | $0 | $0 | ⚠️ NO SIGNAL 14j (KOL inactif ou pas dans groupes scrapés) |

**Findings critiques** :

1. 🚨 **`venom_gambles` est le seul cas litigieux** : SOL catastrophique (-$575/d) MAIS ETH **excellent** (+$93/d, WR 80.8%, med +62.9%). Le ban live all-chain actuel **bloque +$93/d ETH alors qu'il pourrait fire**. Pour exploiter ce split :
   - **Option A (sans code change)** : KEEP ban live (perte +$93/d ETH acceptée pour éviter le risque -$575/d SOL si SOL re-fire en live).
   - **Option B (avec code change)** : ajouter `live_trading.kol_chain_blacklist` (per-chain, comme `paper_trade_config.kol_chain_blacklist`). Ensuite unban venom du `kol_blacklist` global + ajouter à `live_chain.solana`. Permet allow ETH live.
   - **Action interim** : ajouter venom à `paper_chain_blacklist.solana` (déjà -$8K en paper shadow → confirme toxicité SOL, devrait être paper-banni aussi).

2. ✅ **`jadendegens` mérite son triple ban** : -$1,842/d cumul cross-chain en paper shadow, 4,679 fires en 14j sur 22 tokens (= spam pumper professionnel). Unban = catastrophe.

3. ⚠️ **`MaestrosDegen`** : 0 fires en 14j. Soit le KOL est inactif, soit RT listener ne capture pas. Ban "préventif" sans data. **Action** : retirer du blacklist pour collecter data, ou laisser (cost zéro).

4. 🟡 **`aliensalphacalls` ETH** : N=24 avg +5.4% mais médian -11.6% et sum -$226 = moonshot reliance. Ban global justifié quand SOL est -$694/d. ETH-only signal trop faible.

**Action items** :

| # | Action | Status |
|---|---|---|
| 1 | Ajouter `venom_gambles` à `paper_chain_blacklist.solana` | ✅ DONE (migration `v14e57_blacklist_audit_may7`) |
| 2 | Implémenter Option B : `live_trading.kol_chain_blacklist` per-chain | ✅ DONE (commit `25ff5de`, safe_scraper.py:1742, +3 tests) |
| 3 | Move venom du flat live → live chain SOL only (allow ETH live) | ✅ DONE (migration `v14e57_split_venom_live_eth_allow`) |
| 4 | Remove `MaestrosDegen` du live blacklist (no signal 14j) | ✅ DONE |
| 5 | Re-audit complet dans 7j (Mai 14) — verdict shadow post-companion-fix | 📅 scheduled |

**Net post-audit** :
- Live blacklist all-chain : 6 → **4 KOLs** (-Maestros, -venom)
- Live chain SOL : **1 KOL** (venom_gambles, structure neuve)
- Paper chain SOL : 16 → **17 KOLs** (+venom)
- Quand live resume : `venom_gambles` ETH calls fire en live (capture +$93/d signal), SOL bloqué (évite -$575/d). MaestrosDegen libéré pour collecter de la data si actif.

---

## ❓ Open questions / next iterations

### Q1bis. Audit méta blacklist 2026-05-07 — verdict + 4 cas litigieux

**Verdict global** : la blacklist SOL/ETH est **bonne à ~80%** sur 14j de paper shadow (volume amplifié × ~600 strats actives).

- **13/17 SOL bannis** confirmés bad : papicall (-$4K/d), leoclub69, zcallz, chiggajogambles, DegenSeals, TheReaperGems, markdegens, aliensalphacalls, venom_gambles, LevisAlpha, ryoshikdegen, bagcalls, jadendegens (mais variance — voir bas).
- **3/3 ETH bannis** confirmés bad : jadendegens, batman_gem, aliensalphacalls.
- **Allowed side ETH** : tous profitables (DegenSeals +$316/d, TheReaperGems +$112, mad_apes_ETH +$586, etc.) → ETH blacklist propre.

**4 cas litigieux à investiguer (next iteration)** :

| KOL | 14j $/d | 7j $/d | Hypothèse |
|---|---:|---:|---|
| **`bounty_journal`** | **+$1,046** | **+$3,341** WR 60% | 🚨 faux positif probable — flip majeur 14j ET 7j positif |
| `mad_apes_gambles` SOL | -$589 | **+$1,800** WR 56% | 🟡 7j flip positif après bad 14j |
| `jadendegens` SOL | -$1,612 | +$2,293 WR 93% | 🟡 variance forte (1-2 jours moonshot ?) |
| `ChairmanDN1` SOL | -$793 | +$615 WR 95% | 🟡 idem |
| `ramcalls` SOL | -$361 | +$918 WR 96% | 🟡 idem |

**Caveat critique de lecture** : les `$/d` shadow sont amplifiés par le volume (~600 strats × N tokens). L'impact paper main réel = **~2-5% de ces chiffres** (rapport active strats / total shadow). Donc bounty_journal "+$14K/14j shadow" ≈ ~$300-700/14j paper main si unban.

**Actions deferred pour next iteration (Mai 14 ou +)** :
- [ ] Daily breakdown bounty_journal + 4 autres litigieux 7j → outlier ou nouveau signal ?
- [ ] Per-strat-family audit (bounty_journal × FAST/BE/SLOW/SCALP) — la blacklist optimale est-elle per-famille ?
- [ ] Si bounty_journal confirmé : unban paper SOL + test shadow main 7j → si tient, paper main avec alloc petite ($500 seed).
- [ ] Si mad_apes_gambles 14j stabilise positif : unban paper SOL aussi (mais already allowed ETH where +$586/d).
- [ ] Re-audit des 4 jours +24h post-promote v14e.57 picks (`BE25_LOCK10_TP200_SL40_4H_*`) pour voir si les bannis qu'on consider unban changent la perf.

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

## 🔨 Mechanism reference — Dedup 24h + companion-shadow drift

> Section de référence sur les 2 mécanismes critiques qui gouvernent comment les trades s'ouvrent et comment on mesure le live drift.

### A. Dedup 24h cooldown (paper main + shadow)

**Ce que ça fait** : empêche d'ouvrir un trade `(token, strategy)` si un trade identique a déjà été ouvert ou fermé dans les dernières 24h. Le but : éviter que les KOLs qui re-spam un dead token génèrent une cascade de trades perdants sur la même paire.

**Configuration** :
- Source : `scoring_config.paper_trade_config.dedup_cooldown_hours` (JSONB)
- Valeur actuelle : **24h**
- Source unique de vérité dans `paper_trader.py:_load_paper_trade_config()` ligne 1066

**Algorithme** (`paper_trader.py:open_paper_trades()`) :

1. **Open-combos check** (ligne 1242-1251) : query `paper_trades` où `status='open'` pour les tokens du batch → set `(token, strategy)` à skip.
2. **Cooldown-combos check** (ligne 1253-1283) : query `paper_trades` où `status != 'open'` ET `exit_at > now - 24h` → set `(token, strategy)` à skip.
3. Pour chaque `(token, strategy)` candidat :
   - Si dans `open_combos` → skip (existe déjà ouvert)
   - Si dans `cooldown_combos_main` → skip MAIN (cf. v14e.58 fix ci-dessous)
   - Pour shadow loop : si dans `cooldown_combos` (= all) AND `not is_promoted` → skip shadow aussi

**Historique des versions** :
- **v105** : a élargi le cooldown à toutes les trades (main + shadow), pour éviter le pollution des données shadow par re-entries spam.
- **v144.2** : exclude `source='rt_live'` du cooldown (live et paper sont des univers distincts).
- **v14e.58** (mai 12) : split en deux sets — `cooldown_combos_main` (is_shadow=False only, gates main) + `cooldown_combos` (all, gates shadow non-promoted). Fix régression v14e.57.

**Impact sur perf live** : avec dedup 24h actif, certaines strats **manquent les re-pumps** (KOL re-call un token 2-3h après le 1er → blocked). Selon l'analyse mai 12 (cf. §E iteration log) :
- Strats BE_LOCK family **bénéficieraient** d'un dedup réduit (4h ?) — re-entries +$266 à +$761 sur 5j sur certaines
- Strats SCALP / SLOW4H / TP_NZ_S40 family **gagnent rien à perdre** dedup — re-entries flat ou negatives
- **Décision parquée** : variance trop forte sur 7j shadow ($RKC moonshot = 66% du re-entry profit sur BE25_LOCK10_NZ_S40). Re-évaluer post-v14e.58 sur 14j.

### B. Companion shadow paired-drift (mesure du coût d'être en live)

**Ce que ça mesure** : la différence en pp de PnL entre un trade **live réel** (`source='rt_live'`) et son **shadow twin** (`is_shadow=true` ouvert sur le même token, même cycle, même strat).

```
paired_drift_pp = pnl_pct(live) - pnl_pct(shadow_companion)
```

**Pourquoi ça existe** : la note memory "shadow = main cosmétique" garantit que main paper et shadow calculent le PnL **identiquement** (même entry_price, même logique TP/SL, même position $50). Mais la version **live** subit :

- Slippage Jupiter réel (entry + exit)
- Frais Solana (~$0.02-0.06/round-trip = 2-6% à $1, 0.04-0.12% à $50)
- MEV (sandwich attacks)
- Latency price-vs-fill

Donc `live_pnl < shadow_pnl` toujours d'un certain écart. Le drift médian par strat = **coût d'être en live**.

**Le seuil "< 5pp sur 7j post-promote"** : si la drift médiane dépasse 5pp en absolu, les conditions live divergent trop du modèle paper → strat pas viable à scale.

**Référence empirique** (memory note "v14e.49 drift live↔paper ACTÉ non-divergent", 1 mai) :
- N=175 pairs apples-to-apples sur 5 strats sur 47.9h depuis live deploy
- Drift médian par strat : **−1.20 à −2.36pp** (homogène, sain)
- 31% des paires : live > paper (positive slip — chance Jupiter)
- Tail des rugs (5.7%) : slip 5000-85000 bps = catastrophique mais rare
- Conclusion : pas de divergence systémique au seuil 5pp ✓

### C. Le problème associé — bug v14e.57 (résolu par v14e.58)

**Background** : avant v14e.54 (2 mai), shadow loop dans `paper_trader.py:1644` créait des shadows pour TOUS les `SHADOW_STRATEGIES`, incluant les strats promoted (= déployées en paper main). v14e.54 a ajouté un skip `if strat_name in real_strats: continue` pour éviter les doubles inserts. **Effet secondaire** : les strats promoted ont perdu leur shadow twin → paired-drift mesure devenait impossible (= "pre vs post" temporel, pas paired).

**v14e.57 (commit `985a11d`, 7 mai 21:00 UTC)** : a réintroduit la création des shadows pour strats promoted, MAIS avec un bypass de `open_combos` + `cooldown_combos` au shadow loop pour permettre la mesure paired-drift propre.

**Bug** : la création du shadow companion fonctionne, MAIS la query `cooldown_combos` (ligne 1262, populate depuis `paper_trades` sans filtre `is_shadow`) inclut TOUTES les fermetures — main + shadow. Quand un shadow companion ferme (souvent <1h via TP/SL/timeout), il pollue `cooldown_combos`. Le main loop (ligne 1459 pré-fix) consulte ce même set → main re-entry bloqué.

**Cascade** :
1. Cycle T : KOL call token X → main `SCALP_TP15_SL30` ouvre + shadow companion ouvre
2. Cycle T+30min : shadow companion ferme `sl_hit` à 30% loss → `exit_at` < cooldown_window
3. Cycle T+1h : autre KOL call token X → `cooldown_combos` contient `(X, SCALP_TP15_SL30)` (vient de la fermeture shadow)
4. Main loop ligne 1459 : skip → MAIN bloqué
5. Shadow loop ligne 1654 : passe (`is_promoted` bypass) → shadow companion s'ouvre à nouveau
6. Shadow ferme dans <24h → cooldown bloque pour 24h supplémentaires
7. **Boucle infinie : main jamais re-fire sur ce strat tant que shadow re-pollue le cooldown**

**Symptôme observé** : 14 paper main SOL strats (SCALP, SLOW4H, BE15_LOCK, BE25_LOCK_S40, TP200_S40, FAST_TP100_S35, FAST45_S30, BE50_LOCK25_MCAP) ont **freeze depuis 7 mai 17:22-19:43 UTC** (dernier fire pre-bug). Seules AGE24/48/72_FAST_TP50_SL30 ont continué à firer (leur filtre age-band rate l'effet de cascade — chaque token ne match qu'une AGE band particulière).

**Fix v14e.58 (commit `8b5e4d1`, 12 mai)** : split `cooldown_combos` en deux :
- `cooldown_combos_main` (uniquement is_shadow=False) → gate main re-entry (ligne 1467)
- `cooldown_combos` (all = main + shadow) → gate shadow re-entry (ligne 1657)

Restore le comportement pre-v14e.57 pour main (dedup sur ses propres fermetures uniquement) tout en préservant l'anti-spam v105 pour shadow (re-entry sur dead token).

**Impact sur les bankrolls** : 5 jours frozen → 14 strats SOL avec 0 main fires. Backfill appliqué le 12 mai (first-call dedup-aware) — net −$771 sur 945 trades = signal réel de regime shift, pas artefact (cf. §B iteration log).

### D. First-call only vs no-dedup — méthodologie d'analyse

**Le problème central** : quand on regarde le PnL shadow brut sur 7d, certaines strats apparaissent en haut du ranking parce que **shadow companion (v14e.57)** fire à chaque appel KOL sans dedup. Mais paper main avec **dedup 24h** ne fire qu'une fois par token par jour. Donc le shadow brut **n'est pas un proxy fidèle de ce que main ferait en live**.

**4 angles d'analyse selon dedup window** :

| Mode | Définition SQL | Sens |
|---|---|---|
| `no_dedup` | `SUM(pnl_usd)` (tous trades) | $/d théorique sans dedup = limite supérieure |
| `dedup_1h` | `rank=1 per (strat, token, FLOOR(epoch/3600))` | dedup window 1h |
| `dedup_6h` | `rank=1 per (strat, token, FLOOR(epoch/21600))` | dedup window 6h |
| `dedup_24h` | `rank=1 per (strat, token, DATE(cycle_ts))` | dedup 24h (= configuration actuelle) |
| `dedup_inf` | `rank=1 per (strat, token)` sur tout le window | 1 trade par token total = très conservateur |

**Interprétation des 3 profils typiques** observés (cf. iteration log §H pour data) :

1. **Dedup-INDÉPENDANT** : `no_dedup` ≈ `dedup_24h` ≈ `dedup_inf`. Les events de cette strat ne se chevauchent pas naturellement parce que les filtres (chain, NZ, MCAP, score≥35/40 sélectifs) écartent la plupart des re-calls. Exemple : `TD2_BE5_TP120_SL44_T25` (38.3 / 38.3 / 38.5). **Verdict** : dedup setting non-significatif, shadow brut = main avec dedup OK.

2. **Dedup-DÉPENDANT POSITIF** (re-entries gonflent) : `no_dedup` >> `dedup_24h`. La perf brute est dominée par les re-entries du companion shadow. Exemple : `BE25_LOCK10_TP100_SL30_NZ_S40` (59.1 → 5.3, drop 91%). **Verdict** : ne PAS promote en live tel quel — shadow brut est trompeur, main avec dedup 24h ferait $5/d pas $59/d.

3. **Dedup-DÉPENDANT NÉGATIF** (re-entries hurt) : `no_dedup` < `dedup_short`. Les re-entries perdent net. Exemple : `BE50_LOCK25_TP200_SL40_4H_MCAP` (24.1 no_dedup → 32.3 dedup_1h). **Verdict** : dedup HELP ici, garder dedup 24h actuel ou tester dedup 6h pour capturer le mid-pump.

**Quand utiliser quelle métrique** :
- Décision promote → live : utiliser `dedup_24h` (= ce que main fera en vrai)
- Brainstorm strat tweaks : `no_dedup` et `dedup_inf` bornent l'enveloppe possible
- Audit régresseur post-fix : comparer `dedup_24h` pre-fix vs post-fix pour valider le débuggage

### E. Verify queries

```sql
-- Vérifier que main fire post-fix v14e.58
SELECT strategy, COUNT(*), MAX(cycle_ts) AS last
FROM paper_trades
WHERE chain='solana' AND is_shadow=false AND source='rt'
  AND cycle_ts >= '2026-05-12 13:36:26+00'
GROUP BY strategy ORDER BY last DESC;

-- Mesurer paired-drift live ↔ shadow companion (post-promote)
WITH paired AS (
  SELECT
    l.strategy, l.token_address,
    l.pnl_pct AS live_pp, s.pnl_pct AS shadow_pp,
    l.pnl_pct - s.pnl_pct AS drift_pp
  FROM paper_trades l
  JOIN paper_trades s
    ON l.token_address = s.token_address
    AND l.strategy = s.strategy
    AND abs(extract(epoch from (l.cycle_ts - s.cycle_ts))) < 30
    AND s.is_shadow = true
  WHERE l.source = 'rt_live'
    AND l.status != 'open' AND s.status != 'open'
    AND l.cycle_ts > NOW() - INTERVAL '7 days'
)
SELECT strategy, COUNT(*) AS n_paired,
  ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY drift_pp)::numeric, 2) AS median_drift_pp,
  ROUND(AVG(drift_pp)::numeric, 2) AS mean_drift_pp
FROM paired GROUP BY strategy ORDER BY n_paired DESC;
```

---

## 📜 Iteration log

### 2026-05-12 (post-bug v14e.57 cooldown poisoning, fix v14e.58, Tier S collapse)

#### A. 🐛 Bug v14e.57 → fix v14e.58 (commit `8b5e4d1`)

**Root cause** : la cooldown query dans `open_paper_trades` (paper_trader.py:1262) ne filtrait PAS `is_shadow`. Quand v14e.57 (`985a11d`) a ré-introduit le **companion shadow pour strats promoted** (avec bypass cooldown au shadow loop ligne 1654), les fermetures shadow ont pollué `cooldown_combos`. Conséquence : 14 paper main SOL strats GELÉES depuis May 7 21:57 UTC.

**Symptôme observé** : entre May 7 21:00 et May 12 13:36, seuls AGE24/48/72_FAST_TP50_SL30 firaient en main (leur age-band filtre les excluait du collapse de cooldown). Les 14 autres (SCALP, SLOW4H, BE15_LOCK, BE25_LOCK_*S40, TP200_*S40, FAST_TP100_S35, FAST45_S30, BE50_LOCK25_*MCAP) avec 0 main fires.

**Fix** : split `cooldown_combos` en deux sets — `cooldown_combos_main` (is_shadow=False only) pour main loop, `cooldown_combos` (all) pour shadow loop. Preserve v105 anti-spam shadow, débloque main.

**Backfill** (first-call dedup-aware) : 14 strats SOL `strategy_bankrolls_per_chain` mis à jour avec ce que main aurait fait pendant les 5j frozen. **Net : −$771 sur 945 trades** (signal réel — la plupart des strats ont break en regime shift, voir B). Backup table `_backup_bankroll_v14e58_backfill_20260512`.

**Caveat méthodo** : per memory note "shadow = main" est vrai pour le calcul PnL par-trade, mais shadow companion (post v14e.57) fire sur CHAQUE call KOL (bypass dedup), alors que main avec dedup 24h ne fire que sur le 1er call par token. Donc shadow_pnl total ≠ main_pnl. Pour backfill réaliste : utiliser uniquement le 1er shadow par (token, strat).

#### B. 💀 Tier S/A May 7 — collapse général en 5 jours

8/10 candidats Tier S/A May 7 ont effondrés. Le 7d signal du May 7 n'a PAS prédit May 8-12 :

| Strat (Tier May 7) | $/d May 7 | $/d 12 mai | Verdict |
|---|---|---|---|
| BE25_LOCK10_TP60_SL30 (S #1) | +$57 | **−$13** | 💀 collapse |
| FAST_TP40_SL30_DS (S #2, WR 52%) | +$52 | **−$12** | 💀 collapse |
| FAST_TP50_SL30_BOTH (S #3) | +$49 | **−$13** | 💀 collapse |
| TP50_SL15_NOLAZY (A) | +$71 | **−$16** | 💀 collapse |
| TP50_SL15_BOTH (A) | +$70 | **−$15** | 💀 collapse |
| FAST_TP50_SL30_S40 (A) | +$24 | **−$16** | 💀 collapse |
| FAST60_TP50_SL50_S30 (B 3j) | +$60 (3j) | **−$20** | 💀 collapse |
| BE25_TP80_SL30_LAZYSLOW (B 3j) | +$65 (3j) | **−$28** | 💀 collapse |
| BE50_LOCK25_TP200_SL40_4H_NZ_S40 (A) | +$55 | **+$40** | ✅ tient |
| BE15_TP70_SL50_NZ (B) | +$72 (3j) | **+$26** | ✅ tient |

**Leçon** : un winner 7j est instable. **Plus jamais promote sur 7j seul** — exiger 14j stable + 3j positif + raisonnement mécanique.

#### C. 🆕 Le nouveau top SOL shadow 7d (filter-rich domine)

| # | Strat | $/d 7j | N | WR | Présent May 7 ? |
|---|---|---:|---:|---:|---|
| 1 | **BE25_LOCK10_TP100_SL30_NZ_S40** | $118 | 100 | 48% | ❌ |
| 2 | **TP300_SL50_4H_NZ_S40** | $81 | 84 | 31% | ❌ |
| 3 | **TP200_SL40_2H_NZ_S40** | $78 | 85 | 39% | ex-deck live |
| 4 | **FAST_TP40_SL30_S40** | $67 | 96 | 39% | ❌ |
| 5 | TP200_SL40_4H_MCAP_S40 | $57 | 59 | 34% | ✅ tier A |
| 6 | BE25_LOCK10_TP100_SL30_S40 | $42 | 85 | 46% | ❌ |
| 7 | FAST_TP200_SL40_60M_MCAP_S40 | $42 | 71 | 35% | ❌ |
| 8 | BE50_LOCK25_TP200_SL40_4H_NZ_S40 | $40 | 84 | 37% | ✅ tier A |
| 9 | BE15_LOCK5_TP50_SL30 | $38 | 118 | 46% | partiellement |
| 10 | FAST_TP40_SL30_NOLAZY | $38 | 161 | 38% | ❌ |
| 11 | FAST_TP100_SL20_S35 | $35 | 109 | 43% | tier A |
| 12 | FAST60_TP100_SL50_NZ_S40 | $34 | 85 | 37% | ❌ |

**Pattern critique** : le nouveau top est **dominé par `_NZ_S40` et `_MCAP_S40`** (filter combos). Ça confirme le finding May 7 §E (filter NZ_S40/MCAP_S40 sauve la family TP200_SL40_4H). **À envisager** : standardiser les variants `_NZ_S40` ou `_MCAP_S40` sur les bases performantes.

#### D. 🧪 Les 5 picks v14e.57 — N modéré, verdict mitigé

| Strat (pick May 7) | N (5j) | $/d 5j | Verdict |
|---|---|---|---|
| BE25_LOCK10_TP200_SL40_4H_MCAP_S40 | 42 | **+$15** | 🟡 sim said +$19/d ✓ |
| BE25_LOCK10_TP200_SL40_4H_NZ_S40 | 51 | +$10 | 🟡 mild winner |
| BE25_LOCK10_TP200_SL40_4H_A1to3 | 24 | −$25 | ❌ losing |
| BE25_LOCK15_TP200_SL40_4H_NZ_S40_A24to48 | 9 | −$16 | ❌ N too small |
| BE50_LOCK25_TP200_SL40_4H_A24to48 | 13 | −$18 | ❌ N too small |

**Leçon** : 2/5 picks sim-aligned, 3/5 perdent. L'hypothèse "age-band sweet spot 1-3h / 24-48h appliqué transversalement" ne tient pas pour ces variants. Continuer N gathering puis ré-évaluer à N≥30.

#### E. 🔬 Dedup-différencié par strat — analyse + verdict

**Investigation** : sur 12 top SOL shadow 7d, split first_call (= ce que main avec dedup fait) vs re-entry (= shadow only, bypass cooldown) :

| Profil | Strats | Verdict dedup |
|---|---|---|
| **FIRST-ONLY** (re-entries perdent) | TP300_SL50_4H_NZ_S40, TP200_SL40_2H_NZ_S40, BE50_LOCK25_TP200_SL40_4H_NZ_S40, FAST60_TP100_SL50_NZ_S40 | 🔒 garder dedup 24h |
| **BOTH+** (re-entries marginales) | TP200_SL40_4H_MCAP_S40, FAST_TP40_SL30_S40, FAST_TP200_SL40_60M_MCAP_S40, FAST_TP40_SL30_NOLAZY | 🟡 dedup ne hurt pas mais helps pas non plus |
| **RE-ENTRY DRIVEN** (re-entries dominent le profit) | **BE25_LOCK10_TP100_SL30_NZ_S40** (+$761 re), **FAST_TP100_SL20_S35** (+$394 re), BE25_LOCK10_TP100_SL30_S40 (+$303 re), BE15_LOCK5_TP50_SL30 (+$266 re) | 🔓 candidate dedup-off, MAIS variance massive |

**Variance critique** : sur BE25_LOCK10_TP100_SL30_NZ_S40 +$827 total dont +$761 re-entry, **~$500 vient d'un seul token $RKC** (May 11). Donc 1 moonshot = 66% du résultat. N=64 re-entries mais effective N (distinct tokens) bien plus petit.

**Decision** : ❌ **PAS implémenter dedup-diff maintenant**. Le 7d signal "BE_LOCK loves re-entries" est probablement regime-dépendant. Attendre 14j post-v14e.58 (donnée propre) pour ré-évaluer.

#### F. 🆕 Action items déduits

- [ ] **DEMOTE** les 8 Tier S/A May 7 collapsed du registry actif (BE25_LOCK10_TP60_SL30, FAST_TP40_SL30_DS, FAST_TP50_SL30_BOTH, TP50_SL15_NOLAZY, TP50_SL15_BOTH, FAST_TP50_SL30_S40, FAST60_TP50_SL50_S30, BE25_TP80_SL30_LAZYSLOW)
- [ ] **PROMOTE candidates Tier A (12 mai)** : BE25_LOCK10_TP100_SL30_NZ_S40 (top), TP300_SL50_4H_NZ_S40, TP200_SL40_2H_NZ_S40, FAST_TP40_SL30_S40, BE25_LOCK10_TP100_SL30_S40 — N≥85 each, $/d > $40 stable filter-rich
- [ ] **Re-audit dans 7-14j** : valider que les nouveaux Tier A tiennent post-bug-fix avec données propres (companion shadow correct).
- [ ] **Dedup-diff** parqué — re-examiner après 14j de data post-v14e.58.
- [ ] **Bug RT KOL matching** (post-restart 13:36) — `0/98 KOL groups matched`. Bloque actuellement la vérif du fix. À investiguer.

#### G. ⚠️ Revision du top 7d (May 12 PM) — first-call only (dedup-aware) renverse le ranking

**Contexte** : la liste §C above ("Le nouveau top SOL shadow 7d filter-rich") est basée sur le **total pnl 7d shadow** — qui pour les strats `_NZ_S40` filter-rich est dominé par les **re-entries du companion shadow bypass (v14e.57)**. Donc le ranking §C **NE reflète PAS** ce que paper main avec dedup 24h aurait fait. Re-analyse forcée après une question utilisateur sur la stabilité 20j de FAST_TP40_SL30_S40 (mon premier live pick).

**Multi-window 30d/14d/7d sur 30 strats shadow filtre dpd_30 > $0** :

Vrais stables (multi-window cohérent, med ≥ −3, WR ≥ 45) :

| # | Strat | $/d 30d | $/d 14d | $/d 7d | med 14d | WR 14d | N 14d |
|---|---|---|---|---|---|---|---|
| 🥇 | **BE25_LOCK15_TP200_SL40_4H_NZ_S40** | $18 | $35 | $27 | **+4.0** | **54%** | 137 |
| 🥈 | **BE25_LOCK10_TP80_SL30_S40** | $15 | $33 | $28 | **+2.0** | **53%** | 154 |
| 🥉 | **SCALP_TP10_SL15** | $10 | $28 | $13 | **+9.5** | **61%** | 350 |
| 4 | SCALP_TP15_SL15 | $13 | $34 | $14 | **+10.2** | 55% | 350 |
| 5 | FAST_TP50_SL30_MCAP_S40 | $12 | $32 | $11 | +2.0 | 54% | 105 |
| 6 | BE15_LOCK5_TP50_SL30 | $15 | $39 | $38 | −1.4 | 46% | 193 |

Top 7d "fancy" qui étaient mes premiers picks — moonshot pur (med catastrophique) :

| Strat | $/d 30d | $/d 14d | $/d 7d | med 14d | Verdict |
|---|---|---|---|---|---|
| TP300_SL50_4H_NZ_S40 | $38 | $58 | $81 | **−26.9** | 💀 pure moonshot |
| TP200_SL40_2H_NZ_S40 | $26 | $52 | $78 | −14.3 | 💀 |
| TP200_SL40_4H_MCAP_S40 | $23 | $51 | $57 | **−27.4** | 💀 |
| TP300_SL40_6H_MCAP_S40 | $23 | $49 | $18 | **−40.6** | 💀 + crash 7d |
| **FAST_TP40_SL30_S40** (mon 1er live pick) | $18 | $47 | $67 | −5.5 | 🟡 moonshot-day dep. (May 11 +$231 + May 8 +$191 = 80% du 14d profit) |
| BE25_LOCK10_TP100_SL30_NZ_S40 | $30 | $59 | $118 | −5.9 | 🟡 $RKC carries 66% du 7d |

**Verification dedup-aware (first-call only)** sur les top stables, 14d et 30d :

| Strat | n_first 14d | WR | med | $/d 14d first | $/d 30d first | re_pnl 14d |
|---|---|---|---|---|---|---|
| 🥇 **SCALP_TP15_SL15** | 324 | **55.6%** | **+11.9** | **$36** | **$14** | −$21 |
| 🥈 **SCALP_TP10_SL15** | 324 | **61.4%** | +9.5 | $28 | $10 | −$5 |
| 🥉 FAST_TP50_SL30_MCAP_S40 | 84 | 54.8% | +2.2 | $26 | $9 | +$83 |
| 4 **BE25_LOCK15_TP200_SL40_4H_NZ_S40** | 103 | 51.5% | +2.9 | $24 | **$13** | +$148 |
| 5 BE15_LOCK5_TP50_SL30 | 157 | 45.9% | −1.4 | $21 | $6 | +$258 |
| 6 BE25_LOCK10_TP80_SL30_S40 | 120 | 50.8% | +1.1 | $14 | $6 | +$274 |
| 💀 BE25_LOCK10_TP100_SL30_NZ_S40 | 36 | 41.7% | **−24.8** | **$5** | $6 | **+$753** |

**Findings critiques** :

1. **BE25_LOCK10_TP100_SL30_NZ_S40** était classé #1 sur 7d total ($118/d) — **first-call only révèle $5/d avec med −24.8%**. Le +$753 re-entry est dominé par le moonshot $RKC du 11 mai (~$500). **NE PAS PROMOTE EN LIVE — c'est un piège.**

2. **SCALP_TP15_SL15** et **SCALP_TP10_SL15** sont les **vrais champions** : med 14d **POSITIF** (+11.9 et +9.5), WR 55-61%, dedup-safe (re-entries ~0), $/d stable sur 30d + 14d + 7d. **NON PROMOTED** = pas affectés par le bug v14e.57, données propres.

3. **BE25_LOCK15_TP200_SL40_4H_NZ_S40** : meilleur dans la liste filter-rich §C corrigée — med +2.9 first-call, $13/d 30d stable, design BE+LOCK. **Top choix si on veut TP-wide + filter NZ.**

4. **FAST_TP40_SL30_S40** (mon 1er live pick) — 30d $18/d / 14d $47/d / 7d $67/d. La trajectoire $18→$47→$67 reverse-chrono est un **flag de regime-luck**, pas de skill. Sans les 2 jours top May 8+11 ($231+$191), il fait **$6/d sur 11j**. **NE PAS PROMOTE EN LIVE.**

**Revision du #🥇 recommandation live $50/trade** :

| Pick | Justification |
|---|---|
| ~~FAST_TP40_SL30_S40~~ | ❌ rejeté — moonshot-day dep. |
| ~~BE25_LOCK10_TP100_SL30_NZ_S40~~ | ❌ rejeté — first-call $5/d, $RKC effect |
| **`SCALP_TP15_SL15`** | ✅ **NOUVEAU TOP** — med +11.9, WR 55.6%, dedup-safe, NON PROMOTED (data propre), N=575 sur 30d |
| **`BE25_LOCK15_TP200_SL40_4H_NZ_S40`** | 🥈 alternative BE+LOCK design, med +2.9, plus conservateur |

**Caveat** : SCALP_TP15_SL15 est NON PROMOTED → pas de companion-shadow paired-drift mesuré. Reste à valider en live test $0.50/trade pendant 2-3j avant scale.

**Action items révisés** :
- [ ] Promote SCALP_TP15_SL15 + BE25_LOCK15_TP200_SL40_4H_NZ_S40 en paper main (allocations) → générer companion shadow data sur 7j
- [ ] À J+7 (~Mai 19) : mesurer companion-shadow paired-drift sur ces 2 strats. Si <5pp → green light live test $0.50.
- [ ] Garder l'ancienne reco FAST_TP40_SL30_S40 en historique (sert d'exemple de "piège 7d signal").

#### H. 🔬 Simulation multi-dedup window — ranking ultra-critique

**Contexte** : §G's first-call analysis traitait dedup=24h comme baseline mais user a poussé pour comprendre l'impact dedup variable. Re-analyse avec **5 dedup windows simulés** (no_dedup, 1h, 6h, 24h, inf) sur 14d. Cf. §🔨 Mechanism reference §D pour méthodologie.

**Findings critiques sur 26 top strats SOL shadow N≥80** :

##### 🟢 Tier S — Stable money makers (dedup-indépendant, med ≥ −3, multi-window cohérent)

| # | Strat | 30d ded24 | 14d ded24 | 7d ded24 | med 14d | N 30d | Note |
|---|---|---|---|---|---|---|---|
| 🥇 | **TD2_BE5_TP120_SL44_T25** | **$18** | **$38** | $15 | −2.8 | **496** | Plus solide tous windows + gros sample |
| 🥈 | **BE25_LOCK15_TP200_SL40_4H_NZ_S40** | $14 | $26 | $9 | **+3.7** | 159 | Med positif, BE+LOCK design |
| 🥉 | **SCALP_TP15_SL15** | $13 | $34 | $14 | **+10.2** | **605** | Best median panel, NON PROMOTED |

##### 🟡 Tier A — Plus de $/d mais variance moonshot (med 14d négatif fort)

| Strat | 30d | 14d | 7d | med 14d | Risque |
|---|---|---|---|---|---|
| TP300_SL50_4H_NZ_S40 | **$39** | $58 | **$81** | **−27** | ❌ drawdown brutal |
| TP200_SL40_2H_NZ_S40 | $26 | $52 | $78 | −14 | ❌ moonshot pur |
| TP200_SL40_4H_MCAP_S40 | $19 | $43 | $40 | −28 | ❌ |
| TP300_SL40_6H_MCAP_S40 | $23 | $49 | $18 | −41 | ❌ + cooling 7d |
| **FAST_TP40_SL30_S40** (1er pick rejeté §G) | $18 | $47 | $67 | −5.5 | 🟡 day-dep |
| FAST_TP200_SL40_60M_MCAP_S40 | $18 | $41 | $42 | −9.4 | 🟡 |
| BE50_LOCK25_TP200_SL40_4H_NZ_S40 | $16 | $35 | $40 | −9.5 | 🟡 |

##### 💀 Tier D — Crash 7d ou trompe-l'œil dedup-inflated

| Strat | 30d | 14d | **7d** | Verdict |
|---|---|---|---|---|
| FAST_TP50_SL30_BOTH | $13 | $33 | **−$13** | collapsed |
| FAST_TP40_SL30_DS | $13 | $30 | **−$13** | collapsed |
| BE25_LOCK10_TP60_SL30 | $12 | $27 | **−$13** | collapsed (= Tier S May 7) |
| BE25_LOCK10_TP80_SL30_S40 | $7 | $15 | **−$8** | crashing |
| FAST_TP50_SL30_JUPITER | $9 | $28 | **−$18** | collapsed |
| **BE25_LOCK10_TP100_SL30_NZ_S40** | **$6** | **$5** | $10 | 💀 dedup24h reveal — first-call only $5/d (vs $118/d shadow brut 7d) |

##### Comportement dedup-sensitivity sur 7 strats clés (14d)

| Strat | no_dedup | dedup_1h | dedup_24h | dedup_inf | Profil |
|---|---|---|---|---|---|
| TP300_SL50_4H_NZ_S40 | $58.1 | $58.1 | $58.1 | $60.1 | DEDUP-INDÉPENDANT (events naturels distincts) |
| TD2_BE5_TP120 | $38.3 | $38.3 | $38.3 | $38.5 | DEDUP-INDÉPENDANT |
| SCALP_TP15_SL15 | $34.4 | $34.4 | $34.4 | $35.9 | DEDUP-INDÉPENDANT |
| BE25_LOCK15_TP200_4H_NZ_S40 | $35.0 | $22.3 | $26.0 | $24.4 | re-entries +$11 sur dedup_24h |
| BE25_LOCK10_TP100_SL30_NZ_S40 | **$59.1** | $7.8 | $5.3 | $5.3 | 💀 91% drop = trompe-l'œil |
| BE15_LOCK5_TP50_SL30 | $39.4 | $18.2 | $20.6 | $21.0 | re-entries +$19 |
| **BE50_LOCK25_TP200_SL40_4H_MCAP** | $24.1 | **$32.3** | $28.0 | $32.6 | **dedup HELP** (re-entries net negative) |

**Insight critique** : la majorité des "winners" filter-rich (`_NZ_S40`, `_MCAP_S40`) ont des events naturellement distincts (chain + filter score=40 = peu de re-calls éligibles) → ils sont **dedup-indépendants**. Le bug v14e.57 cooldown a affecté **seulement** les strats BE_LOCK générales (sans filter score serré) qui captent les re-calls naturellement. C'est pour ces strats que mon ranking §C était biaisé.

##### Recommandations finales live $50/trade (ordre par risque/$/d) :

| Profil objectif | Strat | $/mois ($50/trade) | Caveat |
|---|---|---|---|
| **MAX stable** | TD2_BE5_TP120_SL44_T25 | ~$540 | N=496 solide, med −2.8 acceptable, dedup-indep |
| **Conservateur** | BE25_LOCK15_TP200_SL40_4H_NZ_S40 | ~$420 | Med +3.7, BE+LOCK safety, mais 7d cooling ($9) |
| **Best WR psy comfort** | SCALP_TP15_SL15 | ~$390 | WR 55%, med +10.2, N=605, NON PROMOTED data propre |
| **High risk variance** | TP300_SL50_4H_NZ_S40 | ~$1170 | Med −27 = drawdowns brutaux. Buffer 50x bankroll obligatoire. Non recommandé pour $50 direct |
| **Diversifié multi-strat** | 3× $50 sur tier S | ~$1350/mo | Sharpe optimal, mais 3× max_open_positions |

**Recommandation user-facing** : **TD2_BE5_TP120_SL44_T25** comme premier live $50/trade.
- 30d / 14d / 7d cohérent
- N=496 = statistiquement solide
- Dedup-indépendant → perf prévisible
- Median proche 0 → pas portage moonshot
- Aucun signe de regime shift

**Action items §H** :
- [ ] Promote TD2_BE5_TP120_SL44_T25 en paper main (vérifier qu'il est dans `hybrid_strategy.allocations`)
- [ ] Lancer live test $0.50/trade sur cette strat (modifier `live_trading.allocations` + enable=true)
- [ ] Mesurer paired-drift sur 7j (companion shadow vs live)
- [ ] Si drift <5pp et N≥30 → scale à $50/trade
- [ ] En parallèle : monitor SCALP_TP15_SL15 + BE25_LOCK15_TP200_4H_NZ_S40 pour diversification phase 2
#### I. 🚨 Critical revision — blacklist filter applied (= what main avec dedup + blacklist fait vraiment)

**Le point manqué** : §H rank était basé sur shadow data brut. Mais `paper_trade_config.kol_chain_blacklist.solana` (17 KOLs : mad_apes_gambles, papicall, markdegens, ramcalls, leoclub69, ChairmanDN1, chiggajogambles, bounty_journal, DegenSeals, aliensalphacalls, LevisAlpha, jadendegens, bagcalls, ryoshikdegen, TheReaperGems, zcallz, venom_gambles) bloque MAIN mais shadow continue à recevoir leurs trades. Donc le ranking sur shadow surcompte les KOLs toxic qui ne firent jamais main.

Re-analyse avec `kol_group NOT IN (blacklist_17)` :

##### Comparaison shadow brut vs blacklist filter (30d, dedup_24h-approx) :

| Strat | shadow brut | **avec BL** | Δ | bl_pnl_30d | % events from BL |
|---|---|---|---|---|---|
| TP200_SL40_2H_NZ_S40 | $26 | **$36** | +$10 | −$285 | 35% |
| **FAST_TP40_SL30_S40** | $18 | **$27** | +$9 | −$267 | 38% |
| **BE50_LOCK25_TP200_SL40_4H_NZ_S40** | $16 | **$26** | +$10 | −$296 | 35% |
| 🔥 **DECAY_TP50_SL50_E15** | **−$14** | **+$26** | **+$40** | **−$1186** | 38% |
| 🔥 **TP50_SL50** | **−$16** | **+$25** | **+$41** | **−$1222** | 38% |
| TD2_BE5_TP120_SL44_T25 | $18 | $17 | −$1 | +$38 | 39% |
| SCALP_TP15_SL15 | $13 | $16 | +$3 | −$72 | 38% |
| BE25_LOCK15_TP200_SL40_4H_NZ_S40 | $14 | $14 | 0 | −$22 | 32% |

##### 🔥 Winners cachés révélés par le blacklist :

- **DECAY_TP50_SL50_E15** : shadow brut **−$14/d** 30d → looked like net loser. Avec blacklist : **+$26/d**. Les 17 KOLs bannis lui faisaient perdre $1186 sur 30j. **N=375 = sample massif.** Code : généré dynamiquement (`strategies.py:275-280` via `_DECAY_STRATEGIES`), TP starts +50% decays to +15% at t=120min, SL −50%.
- **TP50_SL50** : shadow brut **−$16/d** → looked terrible. Avec blacklist : **+$25/d**. Same magnitude flip. Code : `strategies.py:171-173`, simple TP +50% / SL −50% / horizon 120min.

##### Final top tier post-blacklist (= reality with dedup + blacklist actifs)

| # | Strat | $/d 30d BL | $/d 14d BL | $/d 7d BL | med 14d | N 30d | Notes |
|---|---|---|---|---|---|---|---|
| 🥇 | **BE50_LOCK25_TP200_SL40_4H_NZ_S40** | $26 | $51 | **$47** | **−1.3** | 124 | STABLE cross-window + BE+LOCK safety + med proche 0 |
| 🥈 | **FAST_TP40_SL30_S40** | $27 | $59 | $76 | −5.2 | 162 | Trajectory IMPROVING, mais variance moonshot-day |
| 🥉 | **DECAY_TP50_SL50_E15** | $26 | $58 | $2 | **+3.4** | 375 | Med positif, gros N, MAIS 7d cooling à $2 |
| 4 | **TP50_SL50** | $25 | $57 | $10 | **+0.8** | 375 | Med positif, gros N, 7d $10 cooling moindre |
| 5 | TP200_SL40_2H_NZ_S40 | $36 | $73 | $87 | −11.7 | 126 | Highest $/d mais moonshot reliance |
| 6 | TD2_BE5_TP120_SL44_T25 | $17 | (pas vérif) | (pas vérif) | (pas vérif) | 496 | Blacklist HURT légèrement, mais robust dedup-indep |

##### Recommandation reco $50/trade FINALE post-blacklist :

| Profil objectif | Strat | $/mois ($50) | Confiance |
|---|---|---|---|
| **MAX stable + safety BE/LOCK** | `BE50_LOCK25_TP200_SL40_4H_NZ_S40` | ~$780/mo | 🟢🟢🟢 |
| Plus aggressive uptrend | `FAST_TP40_SL30_S40` | ~$810/mo | 🟢🟢 (variance) |
| MAX big-sample reveal | `DECAY_TP50_SL50_E15` | ~$780/mo | 🟢🟢 (cooling 7d) |
| Conservative | `TP50_SL50` | ~$750/mo | 🟢🟢 (cooling 7d) |
| Stable dedup-indep | `TD2_BE5_TP120_SL44_T25` | ~$510/mo | 🟢 |

**Verdict critique** : ma reco précédente "TD2_BE5_TP120" était techniquement valide mais le blacklist effect réordonne. Le **vrai top avec blacklist actif** est **BE50_LOCK25_TP200_SL40_4H_NZ_S40** :
- 30d / 14d / 7d cohérents ($26 / $51 / $47, **pas de cooling**)
- Med14 −1.3 = très proche 0 (pas moonshot)
- BE+LOCK design = safety built-in
- 35% events from blacklisted = blacklist le PROTÈGE de l'attrition toxic
- N=124 OK

**Action items §I** :
- [ ] Décision live $50 : commencer par BE50_LOCK25_TP200_SL40_4H_NZ_S40 (premier choix) avec test $0.50 d'abord
- [ ] **Méthodo critique apprise** : TOUJOURS filtrer la blacklist actuelle (`paper_trade_config.kol_chain_blacklist.solana`) lors du ranking shadow. Sans ce filtre, les strats qui captent beaucoup de toxic KOLs (38% events des SCALP/DECAY/TP50_SL50) sont **systématiquement sous-évaluées**.
- [ ] Re-faire un audit blacklist counter-factual sur les Tier S (BE50_LOCK25_NZ_S40, DECAY_TP50_SL50_E15, TP50_SL50) — l'edge tient-il **sans blacklist** ? Si oui = strat intrinsèquement bonne. Si non = dépend de la blacklist (cf. recette §7 dans Méthodologie).

#### J. 🚫 Audit blacklist 12 mai — 3 unban + 3 ban catastrophiques

**Contexte** : suite à §I (impact massif blacklist sur ranking), re-audit complet des 17 SOL bannis + scan KOLs allowed loosing.

##### Findings audit 14d-30d shadow

**Faux positifs détectés (3 UNBAN)** — paper shadow montre que ces 3 KOLs sont profitables, ban actuelle leur fait perdre du revenu :

| KOL | n30 | $/d 30d | $/d 14d | $/d 7d | WR 14d | Décision |
|---|---|---|---|---|---|---|
| **bounty_journal** | 4528 | **+$497** | **+$1732** | **+$2578** | **58.8%** | 🔥 UNBAN URGENT (déjà flag May 7) |
| **chiggajogambles** | 5150 | +$70 | +$296 | **+$1019** | 44.7% | UNBAN (flip massif récent) |
| ramcalls | 959 | −$157 | **+$459** | n/a inactive | 96.3% | UNBAN watchful (small recent N) |

**Catastrophiques détectés (3 BAN)** — KOLs allowed avec WR < 2% (cash-zero) :

| KOL | n30 | $/d 30d | WR 30d | med pp | Décision |
|---|---|---|---|---|---|
| **certifiedprintor** | 554 | −$288 | **1.6%** | −48.3 | 🚫 BAN |
| **robo_gambles** | 285 | −$250 | **0.0%** | −57.6 | 🚫 BAN |
| **CSCalls** | 386 | −$137 | **1.0%** | −22.5 | 🚫 BAN |

##### À investiguer (next iteration) — défense rééditer pour ces 4 :

| KOL | n30 | $/d 30d | WR 30d | Note |
|---|---|---|---|---|
| **CarnagecallsGambles** | **10217** | **−$1746** | 29.4% | **Top $ damage panel**. Unbanned Apr 30 (+$596/d at the time) puis a explosé en toxic. Re-ban probable mais N immense — investiguer si comeback signal. |
| sadcatgamble | 3796 | −$704 | 32.8% | High volume, med −27.5 |
| Luca_Apes | 2667 | −$698 | 30.4% | |
| MaybachGambleCalls | 7256 | −$465 | 39.0% | |

##### Confirmés keep ban — les 14 restants

| KOL | $/d 30d | $/d 14d | Verdict |
|---|---|---|---|
| leoclub69 | −$2371 | −$4657 | 🔴 WORST |
| papicall | −$2326 | −$3260 | 🔴 |
| zcallz | −$1182 | −$1930 | 🔴 |
| ChairmanDN1 | −$963 | −$1304 | 🔴 |
| jadendegens | −$918 | −$616 | 🔴 (déjà ban triple) |
| markdegens | −$683 | −$327 | 🔴 |
| DegenSeals | −$460 | −$742 | 🔴 |
| aliensalphacalls | −$334 | −$386 | 🔴 (déjà ban triple) |
| venom_gambles | −$263 | −$563 | 🔴 SOL ban only |
| LevisAlpha | −$252 | −$334 | 🔴 |
| mad_apes_gambles | −$202 | +$479 | 🟡 oscille, keep ban |
| TheReaperGems | −$169 | −$1060 | 🔴 |
| bagcalls | −$88 | −$188 | 🔴 |
| ryoshikdegen | −$106 | n/a | 🟡 inactive, keep ban |

##### Migration SQL appliquée

```sql
UPDATE scoring_config SET paper_trade_config = jsonb_set(
  paper_trade_config, '{kol_chain_blacklist,solana}',
  '[ ... 17 KOLs ... ]'::jsonb
) WHERE id = 1;
-- Backup: _backup_blacklist_audit_20260512
```

##### Impact attendu

- **bounty_journal** unbanned : ajoute potentiellement +$1700/d shadow → +X% paper main (selon le ratio active strats / total shadow ≈ 5-10%, soit ~$85-170/d paper main attendu).
- **Strats du top tier** (BE50_LOCK25_TP200_SL40_4H_NZ_S40, FAST_TP40_SL30_S40, DECAY_TP50_SL50_E15, TP50_SL50) vont voir leur perf paper main **augmenter** car ils captent bounty_journal/chiggajogambles qui sont maintenant allowed.
- **Re-run ranking** après 7j de nouvelles données pour confirmer nouveau top.

##### Action items §J

- [x] Apply blacklist update Supabase SOL (3 unban + 3 ban)
- [x] Backup `_backup_blacklist_audit_20260512` créée
- [x] **CarnagecallsGambles BANNED** (12 mai PM) — top $ damage 30d −$1746/d N=10217. Sa flip Apr 30 → May 12 (+$596 → −$1746) montre que les unban-sur-7d-signal sont risqués. SOL blacklist 17 → 18.
- [x] **ETH blacklist audit** — BAN dddegens (−$199/d 7d worsening + med −37), CryptoChefCooks (WR 0%), degenncabal (WR 0%, med −42). ETH blacklist 3 → 6. aliensalphacalls ETH kept ban watchful (N=14 30d trop petit pour validation unban malgré WR 78.6%).
- [x] **Re-audit (Mai 20)** : sur SOL paper depuis le 12 mai — `bounty_journal` TIENT massivement (N=737 WR 82.8% med +34.86% **+$4898/d**, la star du panel), `ramcalls` ~flat (N=22 WR 63.6% med +11.56% total −$17, garder watchful). **`chiggajogambles` a FLIPPÉ négatif** (N=869 WR 19.9% med −8.62% **−$1221/d**) → RE-BANNI SOL (`kol_chain_blacklist.solana` 20 → 21, Mai 20). Bans confirmés toujours négatifs (CSCalls −$1003/d, Carnage −$899/d). Backup `data/scoring_config_pre_chigga_reban_20260520T074606Z.json`. aliensalphacalls ETH : pas re-évalué (hors scope ce run).
- [x] **DROP `_backup_blacklist_audit_20260512`** — fait Mai 20 (rollback non-nécessaire et nuisible : perdrait bounty_journal, ré-admettrait CSCalls/Carnage). Aussi droppé `_backup_blacklist_audit_20260507` (lifecycle). 0 table `_backup_*` restante, advisor RLS clean (que des INFO ZERO-POLICY intentionnels).

#### 📚 Note méthodologique consolidée (à appliquer désormais)

Pour évaluer **toute strat candidate** pour live, la query doit appliquer **3 filtres simultanément** :

1. **`is_shadow = TRUE` + `source = 'rt'`** : utiliser shadow data RT (paper main brut a v14e.57 bug)
2. **`rank=1 per (token, day)`** : simuler dedup 24h
3. **`kol_group NOT IN paper_chain_blacklist.solana`** : exclure les KOLs blacklist

Manquer #3 sur-estime tout ce qui capte des KOLs toxic. Manquer #2 surestime les BE_LOCK avec re-entries. Manquer #1 est miner par le bug v14e.57.

---

- [x] Vérifier le nom complet de TD2_BE5_TP120_SL44_T25 dans `strategies.py` — confirmé `scraper/strategies.py:1797-1807` :
  - **TD2** = TIME_DECAY_V2 fine winner
  - **BE5** = SL move to entry à t=5min, peu importe le peak
  - **TP schedule adaptatif** : `[(0, 2.20), (5, 1.40), (15, 1.00), (25, 1.00)]` = +120% TP à t=0 → +40% à t=5min → take any profit à t=15-25min
  - **SL** = −44% (sl_mult 0.55), horizon 25min
  - **Status** : SHADOW ONLY (in SHADOW_STRATEGIES, NOT in active_strategies / hybrid_strategy.allocations)
  - **Bénéfice du status** : 30d data **propre** = non-affectée par bug v14e.57 (pas de companion shadow créé), signal $18/d 30d genuine
  - **Caveat** : son design "take profit early at TP120% but accept TP40% after 5min" capture les fast moonshots ; à valider en live pour vérifier le timing du sell vs slippage Jupiter

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
- ✅ **Audit live blacklist appliqué** (commits `17ed2f0` + `25ff5de` + `b19181f`, migrations `v14e57_blacklist_audit_may7` + `v14e57_split_venom_*`) : MaestrosDegen unban (no signal), venom_gambles split SOL ban / ETH allow via nouvelle infra `live_trading.kol_chain_blacklist`. Live flat 6→4, paper SOL 16→17.
- ✅ **Audit méta blacklist** (Q1bis) : 13/17 SOL + 3/3 ETH confirmés bad. **4 cas litigieux** marqués pour next iteration : `bounty_journal` (top suspect, +$3,341/d 7j shadow), `mad_apes_gambles` SOL, `jadendegens` 7j-flip, `ChairmanDN1`, `ramcalls`.
- 📌 Next iteration prévue : 2026-05-09 ou 10 (après 2-3j de companion-shadow data accumulée + market normal post-May-7). Tasks différées : daily breakdown des 4 litigieux + per-strat-family audit + re-test 7j.

---

## 🔗 Liens

- Memory MEMORY.md (condensé) : `~/.claude/projects/.../memory/MEMORY.md`
- TODO opérationnel : `tasks/todo.md`
- Script propose combo extensions : `scripts/_propose_combo_extensions.py`
- Mega-sweep workflow GH : `.github/workflows/mega-sweep-48h.yml` (cron 02:00 UTC tous les 2j)
