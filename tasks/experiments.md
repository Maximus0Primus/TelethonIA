# Registre d'expériences — TelethonIA

> **But** : à force de tester, affiner jusqu'à avoir vraiment tout testé. Chaque
> hypothèse laisse une trace, qu'elle marche ou non, pour qu'on ne la retente pas
> à l'aveugle dans trois semaines et qu'on voie ce qui reste inexploré.
>
> **Règle d'or** : un résultat sans **contrôle** ne rentre pas ici comme validé.
> Le 5 août, trois résultats spectaculaires ont été tués par leur contrôle
> (dip-buy +12.6 %, balayage de sorties $457, SL conditionnel). Un quatrième
> serait passé sans lui.

## Schéma

| champ | sens |
|---|---|
| **Hypothèse** | ce qu'on croyait, formulé de façon réfutable |
| **Méthode** | données, fenêtre, split, métrique |
| **Contrôle** | permutation / apparié / hors-sélection — sinon `AUCUN` = non concluant |
| **Résultat** | le chiffre |
| **Verdict** | ✅ validé · ❌ mort · ⚠️ non concluant · 🔭 non testé |

---

## ✅ VALIDÉ — ce sur quoi on peut construire

### E01 · Identité du KOL
- **Hypothèse** : certains KOLs sont durablement meilleurs, indépendamment de leurs métriques agrégées.
- **Méthode** : 120 j, dédup 24 h, split train/test, 2 715 cellules stratégie×KOL.
- **Contrôle** : permutation (pnl rebattu dans chaque stratégie) ×2 tirages + découpage mensuel.
- **Résultat** : 161 survivants réels vs 91-99 au hasard. `slingoorioyaps` **90 vs 12** (7.5× le bruit), +5.9 % train / +5.8 % test, **4 mois positifs sur 4**, pire mois +13.4 %. Re-test au seuil 15/15 : **104 vs 2**.
- **Verdict** : ✅ — mémoire `kol_axis_is_the_only_edge_aug5`

### E02 · Bande de sentiment du message (U inversé)
- **Hypothèse** : le sentiment du 1er message prédit le résultat — mais pas de façon monotone.
- **Méthode** : 120 j, 6 stratégies, `kol_mentions.sentiment` (rempli 100 %, 51 k messages).
- **Contrôle** : permutation du **sentiment** dans chaque mois×stratégie ×3 + retrait de slingoorioyaps + mensuel.
- **Résultat** : bande 0.5-0.6 = **+7.97 %** (train +6.84 / test +11.04) vs hasard −3.39 / +1.71 / +0.14. ≥0.7 = −11.71 %. Sans slingoorioyaps : +7.81 % (il ne pèse que 4.4 %). Bat le reste du flux **4 mois sur 4**.
- **Verdict** : ✅ — mémoire `sentiment_band_edge_aug5`

### E03 · Cadence des calls du KOL (silence vs rafale)
- **Hypothèse** : un KOL qui enchaîne les calls mitraille ; celui qui a attendu sélectionne.
- **Méthode** : 531 stratégies, ~600 k lignes, gap causal depuis le call précédent du **même** KOL.
- **Contrôle** : dose-response monotone + retrait d'olympeqg (57 % du bucket rafale).
- **Résultat** : rafale <1 h **−5.43 %** → 1-6 h −3.12 % → 6-24 h −1.31 % → **24-72 h +2.22 %**. Améliore les 8 stratégies testées, **n'en rend aucune fiable seule**.
- **Verdict** : ✅ comme réducteur de risque, ❌ comme stratégie — mémoire `kol_silence_gap_edge_aug5`

### E04 · Ban d'olympeqg
- **Hypothèse** : le plus gros pourvoyeur de flux est un perdant systématique.
- **Méthode** : 120 j, 358 stratégies, split train/test + mensuel + validation à chaud.
- **Contrôle** : permutation (0 survivant réel **et** 0 au hasard) + 9 trades main frais post-ban.
- **Résultat** : 0/358, −4.8 % train / −10.1 % test, 12-25 % du flux. Le bannir = **+1.3 à +2.3 pp/trade sur TOUTES les stratégies**. `FAST_TP50_SL30_MCAP_S40` passe −0.76 % → +1.39 %. Validation à chaud : −13.86 % sur 9 trades cette semaine-là.
- **Verdict** : ✅ **APPLIQUÉ EN PROD** le 5 août (v14e.70)

### E05 · Prédire la survie (pas le gain)
- **Hypothèse** : le dump est structurel donc apprenable ; le pump non.
- **Méthode** : classification au niveau token, split temporel, précision@top20 %.
- **Contrôle** : **12 tirages** de permutation, lecture de la fourchette p10-p90.
- **Résultat** : « ne dumpe pas sous −50 % » = **AUC 0.72**, prec **65.8 %** vs plafond de bruit 44.4 %. Trading : EV −2.42 % → **+5.60 %** (top 20 %). Bas 20 % = −10.29 % **et le plus gros potentiel** (best 154.7 %).
- **Verdict** : ✅ méthodologiquement, ⚠️ opérationnellement — **redondant avec E02** (+1.8 pp seulement dans la bande, n=52). Mémoire `downside_predictable_upside_not`

---

## ❌ MORT — testé, ne pas y revenir

| # | Hypothèse | Contrôle | Résultat | Mémoire |
|---|---|---|---|---|
| E06 | Features token (score, liq, âge, mcap, BSR, pump.fun, n_kol) | permutation sur 4 377 candidats | **3 survivants réels vs 6 au HASARD** | `kol_axis_is_the_only_edge_aug5` |
| E07 | Entry-timing : retard, dip-buy, dip+reclaim | `fill_lag=1` + mono-source | +12.6 %/trade **était un artefact multi-sources**. Mono-source : meilleur = −0.0 % | `price_ticks_multisource_backtest_trap` |
| E08 | Forme récente du KOL (10 calls précédents) | split train/test | **ANTI-prédictive** : forme>0 → −4.35 % en test, forme<0 → −2.91 % | `kol_silence_gap_edge_aug5` |
| E09 | Heure UTC optimale | split train/test | Les 4 tranches **inversent leur signe** entre train et test | idem |
| E10 | Ré-entrées (ce que le dédup 24 h jette) | population hors bande, n=2 098 | 2ᵉ entrée **−1.91 %** vs 1ʳᵉ −0.80 %. Le +4.4 % dans la bande = n=130 dont 67 % sur un seul mois | todo 5 août |
| E11 | Relâcher les gardes de la stratégie pour du volume | mensuel + géométrique | Sans garde $450 → +score40 $1 230 → +mcap $1 402. **Les gardes achètent de l'edge** | todo 5 août |
| E12 | ML régression sur `pnl_pct` | 12 permutations | Plancher de bruit **~10 pp**. Meilleur modèle +3.64 % vs p90 hasard +3.63 % | `ml_is_not_the_path_filter_not_ranking` |
| E13 | ML classification sur l'**upside** (2x, 2x propre) | 12 permutations | AUC 0.48-0.54. Bruit | `downside_predictable_upside_not` |
| E14 | **SL conditionnel** au risque de dump | permutation de `p_survie` | Même SL optimal (70 %) aux deux extrêmes. Gain +0.08 pp (borne haute). **Le contrôle mélangé sortait un gradient PLUS PROPRE que le réel** | idem |
| E15 | Sélection batch (top-3 par score) | comparaison aux rangs 4-10 et au reste | Score 7× plus haut (55.4 vs 7.6) et **survie légèrement PIRE** (49.5 % vs 52.8 %) | ce fichier |
| E16 | `whale_new_entries`, `ml_pred`, `n_kol_confirmations` | fill rate | **0 % remplis** côté RT (`unique_kols` toujours = 1) | `kol_silence_gap_edge_aug5` |

---

## 🔭 NON TESTÉ — le champ restant

| # | Piste | Pourquoi ça n'a pas été fait | Coût |
|---|---|---|---|
| E17 | Contenu textuel au-delà du sentiment (embeddings du message) | jamais tenté ; `message_text` rempli à 100 %, 51 k messages | moyen |
| E18 | Co-occurrence de KOLs (mêmes 2 KOLs qui callent ensemble) | `n_kol_confirmations` mort, mais la co-occurrence se reconstruit depuis `kol_mentions` | faible |
| E19 | Construction de portefeuille (quels N tokens tenir simultanément) | tout a été mesuré trade par trade, jamais en portefeuille | moyen |
| E20 | Sortie dynamique sur les ticks (pas TP/SL fixe) | `sim_engines` a les moteurs ; jamais croisé avec les axes validés | élevé |
| ~~E21~~ | ~~Sizing conditionnel selon la proba de survie~~ | **→ devenu E22, voir ci-dessous** | — |

---

## ✅ E22 · SIZING KELLY — le levier qu'on cherchait n'était pas un signal

- **Hypothèse** : je reportais la moyenne géométrique à **f=1** (tout le capital sur chaque
  trade) et j'en concluais « ça ne compose pas ». C'est un chiffre opérationnellement absurde.
  À une fraction f plus faible, le taux de croissance peut être positif alors qu'il est
  négatif à f=1.
- **Méthode** : `FAST_TP50_SL30_MCAP_S40` + bande 0.30-0.70, 195 trades sur 4 mois,
  séquence chronologique réelle, coût live **−0.4 pp/trade inclus**, courbe de capital
  complète pour mesurer le drawdown (pas seulement le point final).
- **Contrôle** : aucun nécessaire — ce n'est pas une découverte de signal mais une
  ré-expression du même signal. Le seul biais est le Kelly **in-sample** (voir réserve).
- **Résultat** :

| fraction | capital final (4 mois) | drawdown max | plus bas |
|---|---|---|---|
| 0.05 | +185 % | 10.8 % | 96 % |
| **0.10** (quart-Kelly) | **+313 %** | **20.8 %** | 92 % |
| 0.15 | +486 % | 30.3 % | 88 % |
| **0.20** (demi-Kelly) | **+694 %** | 39.0 % | 83 % |
| 0.30 | +1 108 % | 54.5 % | 74 % |
| 0.40 (optimum brut) | +1 285 % | 67.3 % | 64 % |
| **1.00** | **−99 %** | — | — |

- **Verdict** : ✅ — **c'est le meilleur résultat de la session, et l'edge n'a pas bougé.**
  Ce qui change, c'est la taille. À f=1 on perd 99 % ; à f=0.10 on fait ×4.1 en 4 mois.
- **Réserve** : le f optimal est estimé **sur les mêmes données** que l'évaluation. Remède
  standard : prendre la moitié ou le quart de l'optimum estimé. Optimum brut 0.40
  ⇒ **retenir f=0.10 (quart)**, qui donne +313 % avec un DD de 20.8 %, **sous le seuil de
  30 %** que le projet s'est fixé (skill `simulate-live`). f=0.20 dépasse ce seuil.
- **Suppose** des trades séquentiels non chevauchants — vrai à 11 trades/semaine sur un
  horizon de 30 min.

---

## Journal des sessions

### 2026-08-05 — session fondatrice du registre
16 hypothèses testées, 5 validées, 11 tuées. **3 résultats spectaculaires tués par leur contrôle.**
2 bugs silencieux trouvés dans le sweep (33 % des arms morts depuis avril ; ETH KO depuis 2 semaines).
Appliqué en prod : ban olympeqg, filtres `kol_whitelist` / `min_kol_gap_hours` / bande de sentiment (shadow).
