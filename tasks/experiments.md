# Registre d'expériences — TelethonIA

> **But** : à force de tester, affiner jusqu'à avoir vraiment tout testé. Chaque
> hypothèse laisse une trace, qu'elle marche ou non, pour qu'on ne la retente pas
> à l'aveugle dans trois semaines et qu'on voie ce qui reste inexploré.
>
> **Règle d'or** : un résultat sans **contrôle** ne rentre pas ici comme validé.
> Le 5 août, trois résultats spectaculaires ont été tués par leur contrôle
> (dip-buy +12.6 %, balayage de sorties $457, SL conditionnel). Un quatrième
> serait passé sans lui.

## 🔄 MÉTHODE — ce qui a réellement produit les résultats

> **Constat du 5 août, à ne pas perdre.** 31 hypothèses testées. **21 mortes.** Les
> trouvailles qui ont compté ne sont PAS venues de la recherche à l'intérieur d'un
> cadre — elles sont venues du fait de **remettre le cadre en question**. Et à chaque
> fois, c'est l'user qui a poussé, pas la recherche qui a abouti.

| ce qui a payé | le cadre qui était faux | gain |
|---|---|---|
| **E30 portefeuille** | « une stratégie à la fois » + « classer par EV » — or la mise est plafonnée, donc l'argent vaut **n × EV** | **×3.8** |
| **E22 sizing Kelly** | « la géométrique à f=1 dit si ça compose » — absurde, personne ne mise 100 % | **×4.1** |
| **E29 union** | « les filtres s'empilent » — l'intersection réduit le volume, l'union l'augmente | +48 pp |

Chercher une variante de plus a produit **21 échecs**. Questionner une hypothèse implicite
a produit les 3 seuls multiplicateurs. **Le rendement de la remise en cause du cadre est
d'un ordre de grandeur supérieur à celui de la recherche.**

### Les cadres ENCORE non questionnés (par ordre de coût croissant)

| # | Hypothèse implicite jamais testée | Pourquoi ça pourrait tomber |
|---|---|---|
| C1 | **Le sentiment vient du 1er message** | et celui du message qui DÉCLENCHE l'entrée ? Ou l'évolution entre les deux ? Données déjà là. |
| C2 | **On trade tout ce qui passe le filtre** | et si la RARETÉ était un signal ? Jour à 3 opportunités vs jour à 15. Jamais mesuré. |
| C3 | **Un trade par token** | scaling-in sur re-test du prix d'entrée. E27 a tué les sorties partielles de la grille, PAS les entrées échelonnées. |
| C4 | **Horizon intraday (30 min – 2 h)** | toute la grille est intraday. Le multi-jour n'a jamais été le sujet. |
| C5 | **L'objectif est le \$/jour** | avec un plafond à +23 \$/jour (E31), l'efficience du capital ou le risk-adjusted comptent peut-être plus. |
| C6 | **L'univers = tokens callés sur Telegram** | le seul vrai axe neuf serait externe : activité on-chain, flux de liquidité, suivi de wallets. |

### Règle de travail

1. **Avant de tester** : lire ce fichier. Ne pas rechercher ce qui est déjà mort.
2. **Avant de chercher une variante** : demander *quelle hypothèse implicite je n'ai pas
   remise en cause ?* Prendre C1→C6 dans l'ordre.
3. **Après chaque test** : consigner ici, réussite ou échec, avec le **contrôle** utilisé.
4. **Un résultat sans contrôle n'est pas un résultat.** 7 faux positifs le 5 août.

---

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

## ✅ E30 · PORTEFEUILLE 3 STRATÉGIES — la bonne fonction objectif

- **Erreur corrigée (user)** : j'ai classé par **EV** toute la session. Avec une mise
  plafonnée (~$100, contrainte de liquidité memecoin), ce qui compte est **N × EV**.
  Classer par EV privilégie mécaniquement les filtres serrés ⇒ j'optimisais la mauvaise chose.
- **La config** — 3 stratégies en parallèle, chacune avec sa bande :

| stratégie | bande sentiment |
|---|---|
| `BE25_TP80_SL30` | 0.25 – 0.75 |
| `FAST_TP50_SL30_MCAP_S40` | 0.30 – 0.70 |
| `TP50_SL40_S35` | 0.35 – 0.70 |

  **796 trades, 51/semaine, EV +6.7 %, 14/16 semaines positives, 4 mois positifs sur 4**
  (+871, +1106, +1950, +1111). Net à $100/trade : **$5 038** sur 3,5 mois (~$1 439/mois),
  contre $1 324 **au total** pour la config précédente.
- **Pourquoi ça marche** : profils mensuels **complémentaires**. `BE25` seul fait −434 en mai,
  `FAST` fait +788 le même mois ; en juin c'est l'inverse (+1416 vs +56). Séparément chacune
  a un mois faible, ensemble aucun.
- `TD2_BE5_TP120_SL44_T25` sortait premier ($2 449) mais **écarté** : préfixe `TD2_` dans les
  familles d'artefacts auto-dépréciées.

### ⚠️ La contrainte de capital change tout (`scripts/_portfolio_sim.py`)

**2.82 positions simultanées en moyenne, p95 = 6, pic = 20.** Simulation avec refus de trade
quand le capital libre est insuffisant, coût fixe en $ (pas en %) :

**Capital $100, mise $20 (f=0.20) :**

| | final | DD | trades ratés |
|---|---|---|---|
| paper | **$4 791** | 49.8 % | 1 % |
| **avec dérive −3.5 pp** | **$24** | **92.5 %** | 6 % |

**Effet du capital de départ (avec dérive) :**

| capital | final | × | DD | ratés |
|---|---|---|---|---|
| **$100** | **$24** | **0.2×** | 92.5 % | 6 % |
| **$200** | **$1 630** | **8.1×** | 85.4 % | 2 % |
| $500 | $2 967 | 5.9× | 50.7 % | 0 % |
| $1 000 | $3 467 | 3.5× | 39.5 % | 0 % |
| $5 000 | $7 467 | 1.5× | 14.2 % | 0 % |

**À $100, AUCUN f ne fonctionne** avec la dérive : f=0.10 → $86, f=0.15 → $31,
f=0.20 → $24, f=0.30 → $9. Et f≥0.15 fait rater 54 à 83 % des trades faute de capital libre.

- **Verdict** : ✅ la config est **la meilleure connue**, ⚠️ mais **$100 est sous le seuil de
  viabilité**. Entre la ruine et ×8, il n'y a que **$100 de capital de départ** de différence.
  Seuil praticable : **$500** (0 % de trades ratés, DD 50.7 %).
- **⚠️ JAMAIS TESTÉ EN LIVE.** Le live est coupé depuis le 5 juin et tournait sur
  `FAST_TP50_SL30_MCAP_S40` **sans** filtre sentiment. La dérive de cette config est
  **inconnue** — et c'est le facteur qui domine tout ($4 791 contre $24).

---

## ⚠️ E31 · LIMITE STRUCTURELLE — la méthode a un plafond de capacité

> Observation user (5 août) : *« ce n'est vraiment pas une méthode optimale, elle
> n'est bien que si on a beaucoup d'argent au départ »*. Vérifié, et c'est pire :
> **il y a aussi un plafond par le haut.**

La capacité est fixée par **7.4 trades/jour × 100 $ max par token** (liquidité
memecoin). Au-delà, le capital n'achète plus rien.

| capital | gain/jour | gain 3,5 mois | rendement | annualisé | DD |
|---|---|---|---|---|---|
| 100 $ | +1.4 $ | +154 $ | 154 % | 2 290 % | 47.4 % |
| 300 $ | +6.2 $ | +668 $ | 223 % | 5 331 % | 41.3 % |
| **500 $** | +11.0 $ | +1 182 $ | 236 % | 6 166 % | 40.2 % |
| **1 000 $** | **+23.1 $** | **+2 467 $** | **247 %** | **6 848 %** | 39.5 % |
| 2 000 $ | +23.1 $ | +2 467 $ | 123 % | 1 450 % | 27.3 % |
| 5 000 $ | +23.1 $ | +2 467 $ | 49 % | 293 % | 14.2 % |
| 50 000 $ | +23.1 $ | +2 467 $ | **5 %** | 18 % | 1.8 % |

**Le gain absolu SATURE à +23 $/jour dès 1 000 $.** Ajouter du capital ne fait que
diluer le rendement et réduire le drawdown — ça n'augmente jamais le gain.

### Les deux murs

- **En bas** : sous ~500 $, les frais fixes (0.13 $/trade, soit 1.3 % sur une mise
  de 10 $) et un drawdown de 40-47 % rendent la chose fragile. À 100 $ le gain
  n'est que de +1.4 $/jour.
- **En haut** : au-delà de ~1 000-2 000 $, le capital est inutile. La liquidité
  memecoin plafonne la mise, et le nombre de tokens plafonne la fréquence.

**Fenêtre utilisable : 500 $ – 2 000 $.** C'est une contrainte de la classe d'actifs,
pas de la stratégie — aucune optimisation de signal ne la déplacera. Seuls
compteraient plus de tokens/jour (E11 : relâcher les filtres coûte plus d'edge
qu'il n'apporte de volume) ou une mise plus grosse (impossible, c'est la liquidité).

### Ce que ça implique

La question n'est pas « comment gagner plus » mais « ce plafond justifie-t-il
l'effort ». À 1 000 $ : ~700 $/mois en paper, ~350 $/mois après dérive attendue.
C'est un rendement remarquable en %, un montant modeste en absolu, et **ça ne
grandira pas** avec le capital. À enregistrer avant toute future optimisation :
gagner 20 % d'EV en plus ne déplace pas le plafond, ça le déplace de 20 %.

---

## ⚠️ E32 · MEGA SWEEP 28 ARMS — le classement est du bruit, l'apparié donne SCORE45

- **Hypothèse** : le premier sweep où les 28 arms tournent vraiment (v14e.72, après le bug
  des arms morts d'avril) va faire émerger une meilleure config que le deck E30.
- **Méthode** : run `31040338036`, SOL, 28/07→05/08 (**9 jours**), 6 553 260 configs,
  5 023 200 avec N≥30. Ré-analysé **en local** avec le script courant : le job
  `merge_and_analyze` du run avait fait son checkout sur le SHA de 19:38 le 05/08, donc
  **avant** v14e.73/74 — ni plancher de bruit ni portefeuille n'ont tourné sur ces données.
- **Contrôle** : (a) plancher de bruit de sélection v14e.73 ; (b) test APPARIÉ par cellule
  (strategy, source, smoothing, polling, age_band) contre `NONE` ; (c) le même apparié
  restreint à **une** cellule de référence (`jupiter/raw/lazy_fast`, 1 196 comparaisons)
  parce que les 251k cellules rejouent les mêmes trades et ne sont pas indépendantes ;
  (d) chaque bras jugé en EV **et** en argent (n × EV), la correction v14e.74.
- **Résultat** :
  - plancher **23.87 pts** ; top du CSV 49.61 ; **2 754/5 023 200** au-dessus, tous des
    `TP500_*` à n=46-55 (loterie). Le n°1 du classement argent est **sous le plancher**.
    `cross_regime_robust` = 0 ⇒ top_robust **vide**. Portefeuille = 2 configs, ×1.95,
    corr +0.54. ⇒ **le classement du sweep ne désigne rien.**
  - apparié indépendant : **SCORE45 +6.71 pp d'EV, +341 d'argent, 70 % des cellules
    gagnées, 36 % du volume conservé** ; SCORE40 +5.31/+316/74 % ; NOZEROLIQ_BSR52
    +4.37/+213/59 % ; NOZEROLIQ +2.46/+190/63 %.
  - **GAP24 −3.42 pp et −327 d'argent** (37 % de cellules gagnées) — contredit E?/silence
    du KOL validé sur 120 j. **SENT30_70 +0.57 pp mais −97 d'argent** (42 %) — le filtre
    du deck E30 n'apporte rien sur cette fenêtre.
  - `BSR_MCAP` : n°1 sur 251k cellules (+941), n°11 sur les cellules indépendantes (+15).
    **Artefact de réplication** — la lecture « toutes cellules » est trompeuse par
    construction, ne jamais la citer seule.
- **Verdict** : ⚠️ — un seul candidat, **SCORE45**, et seulement en shadow : 9 jours, un
  seul régime, et le sweep rejoue des règles de sortie **sans** la bande de sentiment que
  le deck applique réellement. Les 28 arms sont confirmés vivants (le bug d'avril est bien
  mort).
- **⚠️ REQUALIFIÉ le même jour (voir E33)** : ces chiffres portent sur **8.9 % de
  l'univers**. La contradiction GAP24 n'était pas une contradiction, c'était un bug.

---

## 🐛 E33 · POURQUOI LE SWEEP NE POUVAIT RIEN TROUVER — il voyait 9 jours sur 4 mois

- **Hypothèse** (posée par l'user, pas par la recherche) : si le sweep ne sort jamais rien
  d'exploitable, le problème n'est peut-être pas la sévérité des contrôles mais l'instrument.
- **Méthode** : relecture du log du run à la ligne près. `Universe: 2717 unique tokens since
  2026-04-13` immédiatement suivi de `240 with ticks`. Ce rapprochement n'avait jamais été
  fait — les deux lignes sont à 130 lignes d'écart dans le log, séparées par la barre de
  progression `20/2717 … 2700/2717` qui donne l'illusion que tout a été traité.
- **Cause** : `_mega_sweep_run_extended` allait chercher les ticks dans une fenêtre ancrée
  sur l'horloge (`now − 8 jours`), identique pour tous les tokens, quel que soit
  `--mega-since`. Tout token plus vieux que 8 jours ressortait sans ticks et le replay le
  sautait **en silence**.
- **Contrôle** : requête SQL comparant, mois par mois, la couverture avec la fenêtre 8 j
  fixe et avec une fenêtre propre à chaque token. Avril/mai/juin : **0 token couvert**.
  Total **242 / 2 734 (8.9 %)** contre **2 734 / 2 734 (100 %)** après correction. Les 242
  reproduisent le « 240 with ticks » observé ⇒ la cause est établie, pas supposée.
- **Résultat** : profondeur **9 jours → 4 mois**, univers **×11.3**. Financé par
  `--mega-lean-grid` (lissage × cadence 70 → 6 : des hypothèses de lecture du prix, pas des
  choix de stratégie, qui consommaient 98.6 % du budget).
- **Verdict** : 🐛 corrigé v14e.77. **Conséquence méthodologique la plus chère** : sur 3 runs
  et plusieurs sessions, « le sweep ne trouve rien » a été lu comme un résultat sur les
  stratégies alors que c'était un défaut de l'instrument. Le garde-fou ajouté (alerte quand
  <50 % de l'univers a des ticks) existe pour que ça ne puisse plus passer inaperçu.
- **À faire** : relancer, puis rejuger SCORE45, GAP24 et SENT30_70 sur les 4 mois.

---

## ✅ E34 · LE SWEEP SUR 4 MOIS RÉELS — un seul bras survit, et ce n'est pas celui du deck

> Run `31089886117` (06/08, 09:37→14:33 UTC, 18 shards, SHA `29c0100`). **Le premier sweep
> qui voit réellement 4 mois** — il clôt E33 et requalifie E32.

- **Contrôle d'instrument passé en premier** (protocole E33) :
  `Universe: 2734 unique tokens since 2026-04-13` → **`2734 with ticks`**, soit **100 %**
  contre 8.9 % au run précédent. Aucun `[WARN]`. Les résultats sont opposables.

- **Le classement reste du bruit — et ce n'est plus un problème de volume de données.**
  Plancher de bruit de sélection **24.88 pts**, meilleure config du CSV **23.90 pts** :
  **0 / 962 802** configs le dépassent. Avec 11× plus d'univers et 4 mois au lieu de 9 jours,
  le top-30 est **toujours** sous le plancher. ⇒ Ce n'était pas la faim de données : le
  classement par max sur ~1 M de configs est structurellement du dredging. **Ne jamais
  promouvoir depuis le top-30**, la conclusion est maintenant définitive.

- **Ce que les 4 mois débloquent vraiment** : `cross_regime_robust = True` sur **59 138**
  configs, contre **0** au run précédent. La machinerie de régime ne fonctionnait pas faute
  de régimes visibles sur 9 jours. Les colonnes `mois+` / `top_mois` sont opposables pour la
  première fois.

- **`[verdict par bras]`** — test apparié, cellule `jupiter/raw/lazy_fast`, 2 404 cellules :

| bras | Δ$/j | Δargent | ΔEV | gagne | volume | mois+ | top_mois | |
|---|---|---|---|---|---|---|---|---|
| **SENT_NOHYPE** | **+1.8** | **+183** | **+0.51** | **74 %** | **96 %** | **4/5** | 44 % | ✅ **retenu** |
| NOBURST | −0.6 | −64 | +0.38 | 39 % | 83 % | 4/5 | 49 % | |
| GAP24 | −2.9 | −290 | **+1.06** | 29 % | 42 % | 3/5 | 49 % | ❌ |
| SCORE45 | −4.0 | −403 | −0.69 | 24 % | 34 % | 4/5 | **66 %** | ❌ un seul mois fait le résultat |
| SENT30_70 | −4.2 | −422 | +0.22 | 25 % | 27 % | 3/5 | 52 % | ❌ |
| SENT50_60 | −49.1 | −5699 | **+7.50** | 32 % | **5 %** | 5/5 | 42 % | ❌ |
| TOPKOL | −59.8 | −6933 | −2.49 | 23 % | 13 % | 4/5 | 54 % | ❌ |

  **1 bras retenu sur 24 : `SENT_NOHYPE`.**

- **Le résultat qui compte : c'est la borne HAUTE du sentiment qui paie, pas la bande.**
  `SENT_NOHYPE` = `s < 0.70` (couper la seule queue de hype).
  `SENT30_70` = `0.30 ≤ s < 0.70` (couper les deux queues) — **c'est le filtre en production
  sur les trois `PF_*` du deck E30**.
  `SENT_NOHYPE` **domine `SENT30_70` sur les deux axes** : EV supérieure (+0.51 vs +0.22)
  **et** 96 % du volume conservé contre 27 %. La borne basse ne paie pas sa perte de volume
  — elle ne gagne même pas en EV. ⚠️ Ça ne contredit pas le U inversé d'E28 : l'EV monte bien
  quand la bande se resserre (`SENT50_60` : **EV +7.50**, la plus haute du tableau, 5/5 mois
  positifs). Mais à mise plafonnée **argent = n × EV**, et couper 95 % du volume coûte
  −$49/j. C'est **exactement la leçon d'E30**, reconfirmée sur un axe indépendant.

- **Les 3 verdicts requalifiés d'E32, tranchés :**
  - **SCORE45** — le candidat n°1 du run aveugle est **mort** : −$4.0/j, EV −0.69, et
    `top_mois 66 %` (un seul mois porte tout). Son +6.71 pp mesuré sur 8.9 % de l'univers
    était un artefact de fenêtre.
  - **GAP24** — la contradiction est résolue, et **les deux camps avaient raison** :
    EV **+1.06** (la validation 120 j ne mentait pas) mais **−$290** d'argent, parce qu'il
    jette 58 % du volume. Bon filtre au sens de la qualité par trade, perdant à mise
    plafonnée. **Ne pas promouvoir.**
  - **SENT30_70** — « n'apporte rien » devient **« coûte de l'argent »** : −$4.2/j,
    3/5 mois positifs, plus d'un mois négatif.

- **Portefeuille : rien.** `1 configs decorrelees (|r| ≤ 0.55)`, TOTAL = meilleure seule
  = 26 676 → **1.00×**. L'espoir « la complémentarité inter-mois était invisible sur 9 jours »
  ne s'est pas matérialisé. E30 reste une découverte manuelle non reproduite par le sweep.

- **⚠️ Vérification au niveau du DECK** (posée par l'user : « on avait vu que la bande était le
  meilleur »). Le `[verdict par bras]` moyenne sur **toutes** les stratégies ; le résultat du
  05/08 portait sur **une** stratégie. Rejeu sur les 3 exits du deck, cellule
  `jupiter/raw/lazy_fast`, `age_band=ALL`, 4 mois :

| filtre | n | argent | EV | vs NONE |
|---|---|---|---|---|
| `NONE` | 8 063 | 24 672 | 3.06 | — |
| **`SENT_NOHYPE`** | 7 712 | **27 327** | 3.54 | **+2 656** |
| `SENT30_70` | 1 978 | 7 047 | 3.56 | **−17 625** |
| `SENT45_65` | 834 | 4 663 | 5.59 | −20 009 |
| `SENT50_60` | 435 | 4 552 | 10.46 | −20 120 |
| `GAP24` | 2 876 | 13 803 | 4.80 | −10 869 |

  **Le classement est `SENT_NOHYPE` > `NONE` > `GAP24` > `SENT30_70` > bandes serrées.**
  ⇒ **Retirer purement la bande bat déjà `SENT30_70`** (24 672 vs 7 047), et `SENT_NOHYPE` bat
  les deux. **Le résultat du 05/08 n'est PAS démenti** : sur `FAST_TP50_SL30_MCAP_S40` la bande
  monte bien l'EV de **3.09 → 3.51**, exactement ce qui avait été mesuré. Mais `SENT_NOHYPE`
  donne **la même EV (3.51)** en gardant **96 %** du volume au lieu de 25 %. ⇒ **tout le gain
  d'EV de la bande vient de sa borne haute ; la borne basse n'apporte aucune EV et détruit
  75 % du volume.** L'option « couper seulement le haut » n'avait jamais été testée le 05/08.
  ⚠️ Contre-signal à ne pas cacher : sur le split walk-forward, `SENT30_70` fait **mieux**
  (`wf_test` +4.08 vs −0.63 sur FAST). Échantillon de queue très petit, mais c'est la seule
  mesure hors-échantillon disponible — raison de plus pour passer par le shadow.

- **Verdict** : ✅ l'instrument fonctionne enfin ; ❌ il ne désigne toujours pas de gagnant par
  classement. Le seul acquis est `SENT_NOHYPE`, obtenu par **test apparié**, pas par le top-30.
- **À faire** : cloner les 3 `PF_*` en variantes sans borne basse (`max_sentiment` seul)
  **en shadow**, et les comparer en apparié au deck actuel. ⚠️ **Pas de promotion directe** —
  un seul run, et le passage à `SENT_NOHYPE` multiplierait le volume de trades du deck par
  ~3.5, ce qui déplace le point de fonctionnement vers le plafond de capacité d'E31.

---

## ✅ E35 · POURQUOI LE SWEEP NE TROUVAIT QUE DU BRUIT — il n'appariait qu'un axe sur deux

- **Hypothèse** (posée par l'user) : « faut trouver une solution pour que ce run trouve
  vraiment de bonnes stratégies et pas du bruit. »
- **Constat** : le sweep possédait **déjà** l'instrument qui marche — test apparié + plancher
  de permutation — et ne l'appliquait qu'aux **filtres** (`verdict_par_bras`). Les stratégies
  n'avaient qu'un **classement**. Et c'est le classement qui est du bruit.
- **Cause, mécanique** : le classement compare des configs mesurées sur des **tokens
  différents** (filtre, bande d'âge, hypothèse de lecture du prix distincts). Cette variance
  inter-cellules est énorme devant l'écart réel entre deux sorties, et c'est exactement elle
  que le maximum de ~1 M de tests va chercher. Ajouter des données n'y change rien — E34 l'a
  prouvé : 11× plus d'univers, sommet toujours sous le plancher.
- **Méthode** : `verdict_par_exit` (v14e.80). **Dans** une cellule (même filtre, même bande
  d'âge, même source/lissage/cadence), toutes les sorties voient les **mêmes tokens**. On
  mesure l'écart de chaque sortie à la **médiane de sa cellule** — pas à une référence
  arbitraire — puis la médiane de ces écarts sur toutes les cellules.
- **Contrôle** (règle L2) : permutation des étiquettes de stratégie **à l'intérieur** de
  chaque cellule, ce qui détruit le lien sortie → résultat en conservant la structure et les
  effectifs. p95 du **maximum** sous H0 ⇒ absorbe la multiplicité des ~600 sorties.
- **Résultat**, run `31089886117`, cellule `jupiter/raw/lazy_fast`, 601 sorties :

| | plancher de bruit | meilleur observé | survivants |
|---|---|---|---|
| classement de configs | 24.88 | 23.90 ❌ **dessous** | **0** / 962 802 |
| **apparié par sortie** | **150** | **+1 490** ✅ **10×** | **226** (67 après regroupement des clones) |

- **CONTRÔLE MULTI-SOURCES** — celui qui avait tué E20b (+58 pp sur une source, négatif sur
  l'autre). Rejoué sur `dexscreener`, source de prix indépendante :
  **Spearman ρ = 0.987** sur l'argent, **0.991** sur l'EV, **14/20** du top-20 commun,
  **2.2 %** de changements de signe. Même tête de classement sur les deux sources.
  ⇒ **le résultat n'est pas un artefact de source.**
- **Ce que ça désigne** : le haut du classement est **entièrement la famille BE+LOCK**
  (break-even + verrouillage), sur les deux sources :
  `BE25_LOCK15_TP150_SL40_T2H` (+1 490, EV +4.76, 75 % des cellules, **5/5 mois**),
  `BE25_LOCK15_TP200_SL40_4H_NZ_S40`, `BE35_LOCK20_TP150_SL40_T2H`,
  `TD2_BE5_TP120_SL44_T25` (**99 %** des cellules, `top_mois` 31 % = le plus régulier).
  ⚠️ **Aucun des 3 exits du deck E30 n'y figure** — le deck utilise `BE25_TP80_SL30` (sans
  LOCK), `FAST_TP50_SL30_MCAP_S40`, `TP50_SL40_S35`.
- **Verdict** : ✅ l'axe stratégie devient opposable. **La leçon n'est pas « il fallait plus de
  données » mais « il fallait apparier ».**
- **À faire** : ⚠️ ne rien promouvoir sur ce seul résultat. BE+LOCK est **path-dépendant**,
  donc exposé au piège des sorties dynamiques (E20b, `dtrail_shadow_artifact`) même s'il passe
  le contrôle multi-sources. Prochain pas : les faire tourner **en shadow** et comparer en
  apparié au deck, comme pour `SENT_NOHYPE`.

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
