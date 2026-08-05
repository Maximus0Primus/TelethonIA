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
> **REGISTRE ÉPUISÉ.** 25 hypothèses testées, plus une seule piste non explorée
> (ci-dessous). 2 axes tiennent (E01 KOL, E02 sentiment), le levier principal est
> le sizing (E22), et **7 résultats spectaculaires sont morts à leur contrôle**.

| # | Piste | Pourquoi pas fait | Coût |
|---|---|---|---|
| E28 | Refaire E20b quand `price_ticks` aura 90 j au lieu de 30 | n=60 tokens aujourd'hui, insuffisant pour départager 10 règles | attendre |
> 2 axes tiennent (E01 KOL, E02 sentiment), le levier principal est le sizing (E22),
> et **6 résultats spectaculaires sont morts à leur contrôle**.
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

## ❌ E23 · Taille variable selon la proba de survie

- **Hypothèse** : E22 a montré que le sizing est le levier, E05 que la survie est prédictible
  (AUC 0.72). Faire varier f **par trade** selon cette proba devrait battre un f constant.
  (E14 a tué le SL conditionnel, mais SL ≠ taille : l'un change la forme du trade, l'autre
  l'allocation de capital.)
- **Méthode** : hors échantillon (modèle entraîné sur les 60 % anciens), 86 trades de la bande.
  Chaque règle **renormalisée pour que `mean(f_i) == 0.10`** — sinon on comparerait "plus de
  capital" à "mieux réparti".
- **Contrôle** : même règle sur des `p_survie` **mélangées**, 5 tirages.
- **Résultat** :

| règle | capital | DD max | fourchette du hasard |
|---|---|---|---|
| **constant (E22)** | **+112 %** | **14.3 %** | invariant |
| proportionnel à p | +50 % | 22.2 % | +8 % à +203 % |
| proportionnel à p² | −18 % | 46.3 % | −47 % à +173 % |
| binaire 1.5×/0.5× | +113 % | 16.0 % | +60 % à +194 % |
| binaire 2×/0× (skip) | +104 % | 26.1 % | +30 % à +341 % |
| top 30 % seulement | +39 % | 41.6 % | −15 % à +154 % |

- **Verdict** : ❌ — le constant gagne ou égale, avec **toujours moins de drawdown**. Chaque
  règle variable tombe dans sa propre fourchette de contrôle ⇒ elle n'exploite rien, elle
  ajoute de la variance. Les `p_survie` sont très asymétriques (méd 0.18) donc "proportionnel
  à p" concentre le capital sur peu de trades, d'où le DD.
- **Note** : n=86 hors échantillon, petit. Mais le sens du résultat (DD systématiquement pire)
  est cohérent sur les 5 règles.

---

## ❌ E18 · Co-occurrence de KOLs — look-ahead pur

- **Hypothèse** : un token callé par plusieurs KOLs est un meilleur signal.
- **1er résultat (FAUX)** : dose-response magnifique — 1 KOL −7.5 %, 2 KOLs +1.2 %,
  3-4 KOLs +5.6 %, **5+ KOLs +11.3 %**. Et ça semblait s'empiler avec la bande (+12.8 %).
- **Le piège** : le compte de KOLs portait sur **toutes** les mentions du token, y compris
  celles arrivées **après** l'ouverture du trade. Un token callé par 5 KOLs l'est *parce
  qu'il a déjà pompé*. Circulaire.
- **Contrôle** : recompter en ne gardant que les KOLs ayant mentionné le token **avant**
  `created_at`.
- **Résultat causal** :

| co-occurrence causale | n | EV tous | EV dans bande |
|---|---|---|---|
| **1 seul KOL avant** | 292 | +2.5 % | **+10.7 %** |
| 2 KOLs avant | 169 | +3.0 % | +6.0 % |
| 3-4 KOLs avant | 102 | +1.0 % | −2.8 % |
| 5+ KOLs avant | 20 | −3.7 % | +6.7 % |

- **Verdict** : ❌ — le gradient disparaît, et dans la bande le **meilleur cas est UN SEUL
  KOL**, l'inverse de la lecture naïve. 5ᵉ résultat de la journée tué par son contrôle.
- **Règle générale qui en sort** : toute feature agrégée sur un token doit être recalculée
  **à la date d'entrée**. `n_kol_confirmations` étant mort (0 partout), la tentation de le
  reconstruire depuis `kol_mentions` est forte — c'est exactement là qu'est le piège.

---

## ❌ E19 · Construction de portefeuille — sans objet

- **Mesure** : sur les 195 trades de la config validée, **0.12 position simultanée en
  moyenne**, **max 2**. Durée moyenne 23 min, p90 32 min.
- **Verdict** : ❌ sans objet — la stratégie est quasi toujours à plat, il n'y a pas de
  portefeuille à construire. **Valide au passage l'hypothèse séquentielle d'E22** : les
  chiffres Kelly n'ont pas besoin de correction de concurrence.

## ❌ E24 · Élargir l'univers (tokens mentionnés mais jamais tradés)

- **Hypothèse** : 8 272 tokens mentionnés, 2 817 tradés. Le gate d'admission RT jette
  peut-être de l'argent — ce serait la source de volume qui manque.
- **Méthode** : 2 896 tokens exclus ayant ≥3 snapshots prix, comparés aux 2 141 tradés
  avec **la même mesure** (upside depuis le 1er snapshot, pas le PnL des trades — sinon
  pas comparable).
- **Résultat** :

| groupe | tokens | % +50 % | % 2x | % 4x | upside moyen |
|---|---|---|---|---|---|
| **tradé** | 2 141 | **30.8 %** | **21.1 %** | **9.6 %** | **+102.1 %** |
| exclu | 2 896 | 19.8 % | 12.3 % | 4.3 % | +53.1 % |

- **Verdict** : ❌ pour le volume, ✅ **pour le système** — le gate admet des tokens avec
  ~2× l'upside du pool rejeté, sur les 4 métriques simultanément. Il mérite sa place.
  Élargir l'univers dégraderait.
- **Réserve** : les exclus ont moins de snapshots (31 vs 43), donc leur max est peut-être
  sous-estimé. Mais l'écart est trop large et trop cohérent pour s'expliquer par ça.

---

## ❌ E17 · Le texte au-delà du sentiment

- **Hypothèse** : `message_text` (100 % rempli) porte du sens que le `sentiment` scalaire perd.
- **Méthode** : TF-IDF mots+bigrammes (4 000 features, min_df=5), régression logistique,
  cible = survie (la seule cible avec du signal, E05). Split **temporel** — le vocabulaire
  memecoin tourne vite, les tickers de mai n'existent plus en juillet, d'où le risque.
  Nettoyage des URLs, adresses base58 et `$TICKERS` pour empêcher la mémorisation de tokens
  précis (17.5 % des messages sont une adresse nue et deviennent vides — c'est correct,
  ils ne portent aucun texte).
- **Contrôle** : 8 tirages à labels mélangés.
- **Résultat** :

| modèle | AUC | prec@top20 % | fourchette hasard |
|---|---|---|---|
| sentiment seul | 0.5716 | 44.6 % | 0.428 à 0.572 |
| texte seul | 0.5720 | 42.0 % | 0.469 à 0.515 |
| **les deux** | **0.5716** | 42.9 % | 0.470 à 0.531 |

- **Verdict** : ❌ — **les trois atteignent le même AUC ~0.572**, et les combiner n'améliore
  rien. Si le texte portait une information indépendante du sentiment, « les deux »
  dépasserait. C'est le même signal, et 0.572 est faible (le modèle de survie fait 0.72).
- **Note** : l'AUC faible du sentiment seul vient de la régression logistique, qui ne peut
  ajuster que du monotone alors que la relation est un **U inversé**. Ce n'est pas une limite
  du signal mais du modèle — le filtre en bande capture le U correctement.
- **Fausse alerte au passage** : les 3 AUC affichaient « 0.572 » identiques, ce qui m'a fait
  soupçonner un bug. Vérification à 6 décimales : 0.571594 / 0.572034 / 0.571573. Pas de bug.

---

## ❌ E25 · Détecter le régime à l'avance

- **Hypothèse** : le régime est bimodal (88 % de stratégies positives certaines semaines,
  8 % d'autres). S'il est prévisible, on allume/éteint la stratégie — levier plus gros que
  n'importe quel filtre.
- **Méthode** : 15 semaines. Métriques de la semaine **précédente** (volume, % 2x, % dump,
  upside) vs EV de la stratégie la semaine courante.
- **Contrôle** : permutation des semaines, 5 000 tirages, en prenant le **max sur les
  métriques testées** (correction implicite pour tests multiples).
- **Résultat** : corr. volume précédent **0.524** vs p95 du hasard **0.576** → bruit.
  corr. dump précédent 0.472 vs 0.572 → bruit. Split opérationnel +13.0 % (après forte
  semaine) vs +3.2 % : c'est exactement ce que le test qualifie de bruit à n=7/8.
- **Verdict** : ❌ à n=15 semaines. À revoir vers 40+ semaines de données.
- **Acquis annexe** : corrélation de l'EV avec **sa propre EV précédente = 0.073** ⇒ aucune
  persistance semaine à semaine, donc pas de momentum exploitable non plus.
- **Chiffre à garder** : sur ces 15 semaines, **EV moyenne +7.8 %, 12 semaines positives**.

## ⚠️ E26 · Balayage du TP sur l'univers filtré

- **Hypothèse** : j'utilise TP50 depuis le début sans l'avoir remis en question.
- **Méthode** : TP40→TP100 à SL30 constant (7 candidats seulement ⇒ risque de sélection
  faible), univers bande de sentiment, croissance à f=0.10 sur la séquence réelle.
- **Résultat** : TP50 **+123 %**, TP70 +105 %, TP60 +79 %, TP80 +43 %, TP40 +31 %,
  TP90 −2 %, TP100 −26 %. Courbe lisse, sommet large entre TP50 et TP70.
- **Verdict** : ⚠️ **confirme le choix existant sans l'améliorer**. Utile comme robustesse :
  ce n'est pas un optimum en pointe d'aiguille, les voisins tiennent.

## ❌ E27 · Sorties partielles (SCALE_OUT / MOONBAG)

- **Hypothèse** : sortir en plusieurs fois lisse les résultats et améliore la composition.
- **Résultat** à f=0.10 sur l'univers filtré : **SCALE_OUT −81 %**, **MOONBAG −95 %**,
  contre +213 % pour la config validée. SCALE_OUT a pourtant +2 791 en somme brute
  juin-juillet — la composition le tue quand même (distribution des pertes).
- **Verdict** : ❌ — et c'est un rappel : **la somme brute et le rendement composé peuvent
  pointer en sens opposés.**
- **Annexe** : `BE15_TP50_SL30` + bande donne **2.3× le volume** (447 trades, 25/sem) pour
  +179 % contre +213 %. Mais il perd en mai (3/4 mois au lieu de 4/4) et fait 12/17 semaines
  positives contre 12/15. **Le volume seul ne compense pas la qualité.**

---

## ❌ E20b · Sorties dynamiques rejouées tick par tick

- **Hypothèse** : toutes les stratégies ont un TP/SL **fixé à l'entrée**. Laisser la sortie
  dépendre de ce que le prix a fait devrait faire mieux.
- **Méthode** : rejeu tick par tick, **mono-source** (leçon E07), slippage de **production**
  via `sim_engines._exit`, univers = bande de sentiment, jugement en **composition à f=0.10**.
  10 règles : TP/SL fixes, 3 trailings, 2 TP décroissants, 2 sorties sur ticks baissiers.
- **Contrôle** : rejouer **la même chose sur la seconde source**. Si le classement s'inverse,
  c'est de la sélection (10 règles sur n=60).
- **Résultat** :

| règle | Jupiter | DexScreener |
|---|---|---|
| trailing arm+20 **give15** | −6 pp | **+58 pp** ← meilleur sur DS, négatif sur Jupiter |
| **trailing arm+20 give25** | **+11 pp** | **+21 pp** ← seule positive sur les deux |
| TP décroissant 200→120 | +15 pp | +7 pp |
| sortie 3 ticks baissiers | −35 pp | −42 pp |
| sortie 5 ticks baissiers | −31 pp | −71 pp |

- **Verdict** : ❌ **non actionnable.** Le meilleur sur une source est négatif sur l'autre.
  La seule règle qui tient sur les deux appartient à la famille **TRAIL, indépendamment
  confirmée mauvaise EN LIVE** (DTRAIL10_ACT15_SL70 = −$45 réel). Et les niveaux absolus
  diffèrent d'un **facteur 5** entre sources (référence +15 % vs +83 %), ce qui disqualifie
  la précision de tout classement à n=60.
- **Acquis robuste (négatif)** : les sorties sur **N ticks baissiers consécutifs** sont
  mauvaises sur les DEUX sources (−31 à −71 pp). Ça, c'est solide.
- **Limite dure** : `price_ticks` ne retient que 30 j, et la bande ne garde que ~9 % du flux
  ⇒ 60 tokens. Il faut ~90 j de rétention pour trancher. → **E28**

---

## ⚠️ E29 · UNION de filtres au lieu d'intersection

- **Erreur de raisonnement corrigée** : toute la session j'ai empilé les filtres
  (intersection), ce qui réduit le volume. Les deux axes validés sélectionnent des tokens
  **différents** (slingoorioyaps ne pèse que 4.4 % de la bande) ⇒ l'**union** augmente le
  volume sans diluer la qualité. Jamais calculée avant.
- **Résultat sur la période complète** :

| variante | n | /sem | EV | capital f=0.10 |
|---|---|---|---|---|
| bande seule | 195 | 12.6 | 7.2 % | +213 % |
| **UNION (bande OU slingoorioyaps)** | **217** | **14.0** | 7.2 % | **+261 %** |
| intersection *(ce que je testais)* | 10 | 0.6 | 10.0 % | +9 % |

  +48 pp de rendement composé pour 22 trades de même qualité (+7.6 %).

- **MAIS l'extension est morte.** Ajouter d'autres KOLs classés par EV de la moitié train,
  évalué sur la moitié test :

| variante | n | EV test | capital |
|---|---|---|---|
| bande seule | 83 | **+7.4 %** | **+67 %** |
| + top 3 KOLs | 89 | 5.4 % | +45 % |
| + top 5 KOLs | 118 | 4.3 % | +41 % |
| + top 8 KOLs | 126 | 3.5 % | +32 % |
| **top 5 KOLs SEULS** | 48 | **−5.0 %** | **−27 %** |

- **Verdict** : ⚠️ le **principe** de l'union est bon et corrige une erreur de méthode, mais
  il n'y a **qu'un seul KOL validé** avec qui faire l'union. Sélectionner d'autres KOLs par
  classement d'EV dégrade monotonement et les 5 meilleurs sont **négatifs en test**.
  `slingoorioyaps` tenait parce qu'il battait son null de **7.5×**, pas parce qu'il était
  bien classé. Le gain de +48 pp est lui-même partiellement in-sample.
- **Règle** : pour élargir l'union il faut un KOL qui passe le **null de permutation**, pas
  un KOL bien classé. Aujourd'hui il n'y en a qu'un.

---

## Journal des sessions

### 2026-08-05 — session fondatrice du registre

**25 hypothèses testées : 2 axes validés, 21 morts, 1 confirmation, 1 en attente de données.**

**7 résultats spectaculaires tués par leur contrôle** — sans contrôle systématique, 7 fausses
stratégies auraient été livrées ce jour-là :

| ce qui brillait | ce que le contrôle a montré |
|---|---|
| dip-buy −50 % à **+12.6 %/trade**, 5/5 semaines | artefact multi-sources ; mono-source = −0.0 % |
| balayage de sorties → **$100 → $457** | le hasard atteint 5.1 % de géom. contre 5.5 % réel |
| SL conditionnel au risque de dump | même SL optimal partout ; **le contrôle mélangé était PLUS PROPRE** |
| sizing variable selon la survie | chaque règle dans sa propre fourchette de hasard, DD pire |
| co-occurrence KOL **+11.3 %** | look-ahead : les KOLs comptés arrivaient **après** l'entrée |
| régime prévisible (corr. 0.524) | p95 du hasard = 0.576 |
| sorties dynamiques **+58 pp** | négatif sur l'autre source |

**2 bugs silencieux trouvés dans le sweep** : 33 % des arms (BSR*/KW*) lisaient des colonnes
absentes du `select` depuis avril et ne matchaient rien ; le sweep ETH échouait à chaque run
depuis 2+ semaines faute d'index `(chain, created_at)`.

**Appliqué en prod** : ban olympeqg (v14e.70), filtres `kol_whitelist` / `min_kol_gap_hours`
(v14e.70), bande de sentiment en shadow 3 bras (v14e.71), nouveaux axes + plancher de bruit
dans le sweep (v14e.72/73), garde de fraîcheur ETH (v14e.72b).

**Le résultat le plus important n'est pas un signal** : c'est E22, le sizing. Reporter la
moyenne géométrique à f=1 faisait conclure « ça ne compose pas » ; à f=0.10 la même stratégie
fait ×4.1 en 4 mois.
