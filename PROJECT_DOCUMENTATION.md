# TelethonIA - Documentation Technique Complète du Projet

## 1. Vue d'ensemble

### 1.1 Qu'est-ce que TelethonIA ?

TelethonIA est une plateforme d'analyse de memecoins crypto qui :
1. **Collecte** automatiquement les messages de 60+ groupes Telegram de KOLs (Key Opinion Leaders) crypto
2. **Analyse** le sentiment de chaque message avec un pipeline NLP hybride (3 méthodes combinées)
3. **Extrait** les tokens mentionnés ($PEPE, $DOGE, etc.)
4. **Score** chaque token selon plusieurs dimensions (conviction, consensus, momentum, réseau)
5. **Affiche** les résultats dans un dashboard Streamlit interactif multi-pages

### 1.2 Problème résolu

Les traders crypto qui veulent identifier des opportunités "early" sur les memecoins doivent :
- Suivre 50+ groupes Telegram en parallèle
- Lire des centaines de messages par jour
- Détecter manuellement les consensus entre KOLs
- Évaluer le sentiment général sur chaque token

**TelethonIA automatise tout ce processus** et fournit un classement actionnable.

### 1.3 Cible utilisateur

| Segment | Description | Besoin |
|---------|-------------|--------|
| **Crypto-curieux** | Connaît BTC/ETH, veut découvrir les memecoins | Filtrage simple, explications pédagogiques |
| **Trader intermédiaire** | Suit quelques groupes, rate les opportunités | Détection de consensus, alertes |
| **Degen light** | Actif mais pas 6h/jour sur TG | Signaux filtrés, scoring transparent |

---

## 2. Architecture technique

### 2.1 Structure des fichiers

```
TelethonIA/
├── exportfinaljson.py              # Scraper Telegram (collecte)
├── ConvictionApp/                  # Application Streamlit
│   ├── app.py                      # Page d'accueil + configuration
│   ├── utils.py                    # Coeur du pipeline (1144 lignes)
│   ├── sentiment_local.py          # Wrapper CryptoBERT
│   ├── summarizer_deepseek.py      # Résumés via API DeepSeek
│   ├── backtest_weights_core.py    # Backtesting (1233 lignes)
│   ├── requirements.txt
│   ├── cache/
│   │   ├── api_keys.json
│   │   ├── pair_cache.json
│   │   └── token_hints.json
│   ├── data/telegram/              # JSONs exportés
│   └── pages/                      # Pages Streamlit
│       ├── Dashboard_global.py     # Classement principal
│       ├── Exploration_visuelle.py # Graphes et heatmaps
│       ├── Vue_par_groupe.py       # Analyse par groupe
│       ├── Investissement.py       # Super classement
│       ├── Backtest_Weights.py     # Optimisation Optuna
│       ├── Stats_historiques_&_Graph.py
│       └── recup_tokens_values.py
├── TelethonClient.py               # Setup auth Telegram
├── getID.py                        # Récupération IDs groupes
├── memecoin_dashboard.py           # Dashboard alternatif (heuristique)
└── group_cache.json                # Cache IDs groupes
```

### 2.2 Flux de données

```
┌─────────────────────────────────────────────────────────────────┐
│                         COLLECTE                                 │
├─────────────────────────────────────────────────────────────────┤
│  exportfinaljson.py                                              │
│  └─> Telethon API (GetHistoryRequest)                           │
│      └─> 60+ groupes Telegram                                   │
│          └─> 50 messages/groupe                                 │
│              └─> messages_export_YYYYMMDD_HHMMSS.json           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PARSING & DEDUP                           │
├─────────────────────────────────────────────────────────────────┤
│  utils.py                                                        │
│  ├─> parse_messages_json()     # Supporte 2 formats JSON        │
│  ├─> deduplicate_messages()    # Fusionne exports, élimine dup  │
│  └─> load_many_jsons()         # Charge plusieurs fichiers      │
│                                                                  │
│  Colonnes produites :                                            │
│  [id, date, group, text, conviction, remark, tokens]            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTRACTION TOKENS                            │
├─────────────────────────────────────────────────────────────────┤
│  utils.py                                                        │
│  ├─> Regex: \$[A-Z][A-Z0-9]{1,14}                               │
│  ├─> Alias sans $ si contexte crypto détecté                    │
│  ├─> Blacklist: TOKEN, COIN, MEME, USD, BTC...                  │
│  └─> explode_tokens() : 1 ligne par (message, token)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ANALYSE DE SENTIMENT                          │
├─────────────────────────────────────────────────────────────────┤
│  3 canaux parallèles :                                           │
│                                                                  │
│  1. VADER (vaderSentiment)                                      │
│     └─> Score [-1, +1] basé sur lexique anglais                 │
│                                                                  │
│  2. Lexique Crypto (custom, 50+ termes)                         │
│     ├─> Positifs: "ath" +0.60, "listing" +0.55, "pump" +0.30    │
│     ├─> Négatifs: "rug" -0.80, "scam" -0.75, "exploit" -0.70    │
│     └─> Gestion des négateurs ("not bullish" -> flip)           │
│                                                                  │
│  3. CryptoBERT (ElKulako/cryptobert via HuggingFace)            │
│     ├─> RoBERTa fine-tuné sur 3.2M messages crypto              │
│     ├─> Classification: Bullish / Bearish / Neutral             │
│     └─> Stretch: tanh(1.8 * arctanh(x)) pour amplifier          │
│                                                                  │
│  Fusion :                                                        │
│  sentiment = (w_hf*HF + w_vader*VADER + w_crypto*LEX) / sum     │
│            + rule_adjustments * gain                             │
│                                                                  │
│  Poids par défaut: HF=0.50, VADER=0.35, LEX=0.15, gain=1.20     │
│                                                                  │
│  Ajustement par conviction groupe :                              │
│  w_sentiment = sentiment * (1 + alpha * (conviction - 5) / 10)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SCORING TOKENS                              │
├─────────────────────────────────────────────────────────────────┤
│  Plusieurs scores calculés :                                     │
│                                                                  │
│  1. score_conviction (0-10)                                     │
│     = alpha * norm(mentions) + (1-alpha) * norm(sentiment)      │
│     alpha = 0.6 par défaut                                      │
│                                                                  │
│  2. score_quick_win (0-10)                                      │
│     = 0.30*sentiment + 0.25*wilson + 0.20*breadth               │
│       + 0.15*momentum + 0.10*accel                              │
│     * (0.7 + 0.3 * polarisation_inverse)                        │
│                                                                  │
│  3. score_invest (0-10) - page Investissement                   │
│     = weighted(avg_score, avg_sent, pagerank, groups_count)     │
│                                                                  │
│  4. super_score (0-10) - mode expert                            │
│     = quality + consensus + network + dynamic - polarisation    │
│     + bonus persistance optionnel                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSE GRAPHE                               │
├─────────────────────────────────────────────────────────────────┤
│  graph_edges_advanced() construit 3 types d'arêtes :            │
│                                                                  │
│  1. group-token : quels groupes parlent de quels tokens         │
│  2. token-token : co-mentions dans mêmes messages               │
│     └─> Poids: NPMI (Normalized PMI) + Jaccard                  │
│  3. group-group : similarité entre groupes                      │
│                                                                  │
│  Decay temporel : exp(-age_h / tau)  avec tau=12h par défaut    │
│                                                                  │
│  Métriques extraites :                                           │
│  ├─> PageRank par token                                         │
│  ├─> Clusters Louvain                                           │
│  ├─> Autorité groupes                                           │
│  └─> Convergence clusters                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALISATION                                │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit multi-pages avec Plotly :                            │
│  ├─> Tableaux classement                                        │
│  ├─> Heatmaps (groupes x tokens, temps x tokens)                │
│  ├─> Bubble charts (sentiment x mentions)                       │
│  ├─> Bump charts (évolution des rangs)                          │
│  ├─> Streamgraphs par cluster                                   │
│  ├─> Volcano plots (sentiment vs anomalie)                      │
│  ├─> Sankey diagrams (rank-flow)                                │
│  └─> Graphes de réseau (NetworkX + PyVis)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Description détaillée des pages

### 3.1 Page d'accueil (`app.py`)

**Fonction** : Configuration globale et aperçu des données

**Fonctionnalités** :
- Upload de fichiers JSON (multi-fichiers avec déduplication)
- Configuration de la période d'analyse (2h, 6h, 12h, 24h, 48h, Tout, ou plage personnalisée)
- Réglage des poids de sentiment :
  - Poids modèle HuggingFace (CryptoBERT)
  - Poids VADER
  - Poids lexique crypto
  - Gain dynamique (multiplicateur)
- Réglage des ajustements :
  - Poids des règles/boosts lexicaux
  - Poids conviction de groupe
  - Détection alias sans $
- Activation/désactivation du résumeur (DistilBART ou DeepSeek)
- Aperçu des messages bruts avec colonnes : date, group, tokens, sentiment, text, remark

**Paramètres clés** :
```python
use_hf = False          # Activer CryptoBERT
w_hf = 0.50             # Poids HF
w_vader = 0.35          # Poids VADER
w_crypto = 0.15         # Poids lexique
gain_sent = 1.30        # Multiplicateur
rule_weight = 1.0       # Poids règles
group_alpha = 1.0       # Poids conviction groupe
```

---

### 3.2 Dashboard global (`Dashboard_global.py`)

**Fonction** : Classement principal des tokens avec tous les indicateurs

**Fonctionnalités** :

#### KPIs en haut de page
- Nombre de messages
- Nombre de groupes
- Tokens uniques
- Sentiment moyen

#### Classement par token (tableau principal)
Colonnes affichées :
| Colonne | Description |
|---------|-------------|
| token | Symbole du token |
| score_conviction | Score 0-10 (mentions + sentiment) |
| score_conviction_graph | Score 0-10 avec renfort graphe |
| score_quick_win | Score 0-10 orienté trading court terme |
| mentions | Nombre total de mentions |
| breadth | Nombre de groupes uniques |
| sentiment | Sentiment moyen [-1, +1] |
| wilson_low | Borne basse Wilson (confiance) |
| polarisation | Désaccord entre groupes [0, 1] |
| momentum | Pente récente des mentions |
| accel | Accélération des mentions |
| AutoritéGroupes | Score d'autorité (PageRank groupes) |
| ConvergenceClusters | Multi-cluster coverage |
| CentralitéPR | PageRank token |
| résumé | Résumé DeepSeek (si activé) |

#### Paramètres avancés (sidebar)
- Poids mentions vs sentiment (alpha)
- Demi-vie temporelle (tau)
- Seuils NPMI et co-mentions pour le graphe
- Poids Autorité/Convergence/Centralité

#### Graphiques
- Evolution temporelle du score de conviction (par token)
- Détection des flips de sentiment
- Détection des newcomers (tokens émergents liés aux leaders)

---

### 3.3 Exploration visuelle (`Exploration_visuelle.py`)

**Fonction** : Visualisations avancées pour explorer les données

**Fonctionnalités** :

#### Heatmaps
- **Mentions** : matrice tokens x temps, avec annotations :
  - `•` = z-score > seuil (spike significatif)
  - `◀▶` = flip de sentiment détecté
- **Sentiment** : même matrice, colorée RdYlGn

#### Options heatmap
- Top N tokens (5-50)
- Critère de sélection : Mentions, Breadth, Spike récent, Score conviction
- Seuil z-score pour cellules significatives
- Tri par clusters (Louvain) optionnel

#### Graphiques optionnels (toggles)
1. **Bubble chart** : Sentiment x Mentions (taille = breadth, couleur = cluster)
2. **Bump chart** : Evolution des rangs dans le temps (top 20)
3. **Streamgraph** : Mentions par cluster empilées
4. **Ridgeline** : Distribution des sentiments par token (violon)
5. **Volcano plot** : Sentiment vs z-score anomalie
6. **Rank-flow** : Sankey diagram du top 10 entre fenêtres

---

### 3.4 Vue par groupe (`Vue_par_groupe.py`)

**Fonction** : Analyse détaillée groupe par groupe

**Fonctionnalités** :

#### Filtres
- Sélection du groupe
- Sélection optionnelle d'un token

#### Affichages
1. **Messages récents** : tableau avec date, tokens, sentiment, text, remark
2. **Détail par token** :
   - mentions, sentiment, ci95 (intervalle confiance)
   - score_conviction, mots-clés, résumé, Sentiment_HF
3. **Top conviction du groupe** : classement intra-groupe

#### Heatmap Groupes x Tokens
- Matrice de score_conviction
- Tri optionnel par clusters (algorithme Louvain sur graphe biparti)

#### Consensus picks
- Tokens présents dans le Top-K de plusieurs groupes
- Paramètres : Top-K par groupe, Min groupes
- Colonnes : groups_count, groups_list, avg_score, avg_sent, mentions_total

---

### 3.5 Investissement (`Investissement.py`)

**Fonction** : Classement avancé pour décision d'investissement

**Fonctionnalités** :

#### Classement "Investissables"
Tokens remplissant les critères de consensus :
- Présents dans le Top-K de au moins N groupes
- Sentiment moyen > seuil

Score composé :
```
score_invest = weighted(
    avg_score,      # Score intra-groupe moyen
    avg_sent,       # Sentiment moyen inter-groupes
    pagerank,       # Centralité réseau
    groups_count    # Convergence
)
```

#### Bump chart investissables
Evolution des rangs dans le temps (optionnel)

#### Super Classement (mode expert)
Score composite avancé avec 9 composantes :

| Composante | Poids défaut | Description |
|------------|--------------|-------------|
| quality_sent | 0.22 | Sentiment moyen normalisé |
| quality_wilson | 0.12 | Wilson lower bound (confiance) |
| cons_breadth | 0.14 | Couverture inter-groupes |
| cons_groups | 0.10 | Nombre de groupes |
| network_pr | 0.15 | PageRank |
| dyn_mom | 0.17 | Momentum |
| dyn_acc | 0.05 | Accélération |
| stability | 0.10 | 1 - CI95 |
| polar_penalty | 0.15 | Pénalité polarisation |

#### Bonus persistance (optionnel)
Récompense les tokens restés longtemps dans le haut du classement :
- Fenêtre configurable (6h - 168h)
- Top-R seuil (3-10)
- Mode additif ou multiplicatif

---

### 3.6 Backtest & Weights (`Backtest_Weights.py`)

**Fonction** : Optimisation des poids avec backtesting

**Fonctionnalités** :

#### Sources de données
1. **Construction automatique** depuis l'app
2. **Upload CSV** (signals_features.csv + prices.csv)

#### Association tokens -> contrats
Éditeur pour renseigner :
- chainId (solana, ethereum, base, bsc...)
- Contract address (CA)
- Pair/pool address (prioritaire)

Cache des hints persistant

#### Sources de prix
1. **Saisie manuelle** (éditeur avec template)
2. **Contrats/pools saisis**
3. **Automatique** (Dexscreener -> GeckoTerminal)
4. **Upload prices.csv**
5. **API Birdeye / CoinGecko Pro**

#### Features sélectionnables
- score_conviction_graph
- score_conviction
- score_quick_win
- pagerank, breadth, polarisation
- wilson_low, momentum, accel
- mentions, sentiment

#### Optimisation Optuna
- Mode sélection : threshold ou top-N
- Horizons configurables (mid=30j, long=90j)
- Métrique : winrate ou median_return
- Walk-forward avec N folds

#### Outputs
- Meilleurs poids trouvés
- Trades sélectionnés
- Diagnostics par fold
- Export CSV + JSON config

---

## 4. Pipeline de sentiment (détail)

### 4.1 VADER

Librairie `vaderSentiment` qui calcule un score [-1, +1] basé sur :
- Lexique de 7500+ mots annotés
- Règles pour ponctuation, majuscules, emojis basiques
- Bon pour l'anglais général, moins pour le slang crypto

### 4.2 Lexique crypto custom

50+ termes avec scores manuels :

```python
CRYPTO_LEXICON = {
    # Très positifs
    "ath": 0.60, "all time high": 0.60, "mooning": 0.55,
    "listing": 0.55, "listed": 0.55, "cex listing": 0.55,

    # Positifs
    "bullish": 0.50, "pump": 0.30, "moon": 0.40,
    "audit passed": 0.45, "renounced": 0.40,

    # Négatifs
    "rug": -0.80, "rugged": -0.80, "rugpull": -0.80,
    "scam": -0.75, "scammer": -0.75,
    "exploit": -0.70, "hacked": -0.70,
    "dump": -0.50, "dumping": -0.50,
    "high tax": -0.35, "honeypot": -0.70,

    # Neutres avec contexte
    "dyor": 0.0, "nfa": 0.0,
}
```

Gestion des négateurs :
```python
NEGATORS = ["not", "no", "isn't", "aren't", "wasn't", "weren't",
            "don't", "doesn't", "didn't", "won't", "wouldn't",
            "can't", "couldn't", "shouldn't", "never"]
# Si négateur dans les 3 mots précédents -> flip le signe
```

### 4.3 CryptoBERT

Modèle `ElKulako/cryptobert` sur HuggingFace :
- Base : RoBERTa (125M paramètres)
- Fine-tuné sur 3.2M messages crypto (Twitter, Reddit, StockTwits, Telegram)
- Classification : Bullish / Bearish / Neutral

Traitement dans `sentiment_local.py` :
1. Tokenization avec troncation à 512 tokens
2. Inference en batch
3. Conversion score [0,1] -> [-1, +1]
4. Stretch : `tanh(1.8 * arctanh(x))` pour amplifier les signaux faibles

Boosters d'intensité :
- CAPS ratio >= 45% : +0.08
- `!` ou `!!!` : +0.03 par `!` (max 0.12)
- Mots positifs/négatifs du lexique : +/-0.05
- Sarcasme ("lol" + négatif) : atténue x0.85

### 4.4 Fusion finale

```python
# Scores individuels [-1, +1]
s_vader = vader_analyzer.polarity_scores(text)["compound"]
s_crypto = calculate_crypto_lexicon_score(text)
s_hf = cryptobert_score(text)

# Blend pondéré
w_sum = w_vader + w_crypto + w_hf
sentiment = (w_vader * s_vader + w_crypto * s_crypto + w_hf * s_hf) / w_sum

# Ajustements par règles
for word in positive_words:
    if word in text.lower():
        sentiment += 0.05 * rule_weight

for word in negative_words:
    if word in text.lower():
        sentiment -= 0.07 * rule_weight

# Gain dynamique
sentiment = sentiment * gain  # gain = 1.20 par défaut

# Ajustement par conviction du groupe (6-10)
conviction = message["conviction"]  # Score KOL
w_sentiment = sentiment * (1 + group_alpha * (conviction - 5) / 10)
```

---

## 5. Système de scoring

### 5.1 Score de conviction (base)

```python
# Normalisation
mentions_norm = mentions / max_mentions  # [0, 1]
sentiment_norm = (sentiment + 1) / 2      # [-1,1] -> [0,1]

# Score [0, 10]
alpha = 0.6  # Poids mentions vs sentiment
score_conviction = 10 * (alpha * mentions_norm + (1-alpha) * sentiment_norm)
```

### 5.2 Score Quick Win

Orienté trading court terme, pénalise la polarisation :

```python
# Composantes normalisées [0, 1]
sent01 = (sentiment + 1) / 2
wil01 = wilson_low.clip(0, 1)
br01 = (breadth / max_breadth).clip(0, 1)
mom01 = (momentum / max_momentum / 2 + 0.5).clip(0, 1)
acc01 = (accel / max_accel / 2 + 0.5).clip(0, 1)
pol01_inv = 1 - polarisation.clip(0, 1)

# Combinaison
quick = (0.30 * sent01 +
         0.25 * wil01 +
         0.20 * br01 +
         0.15 * mom01 +
         0.10 * acc01)

# Pénalité polarisation
quick = quick * (0.7 + 0.3 * pol01_inv)

score_quick_win = 10 * quick
```

### 5.3 Score Investissement

Basé sur le consensus inter-groupes :

```python
# Après consensus_table() : tokens dans Top-K de >= N groupes
score_norm = avg_score / 10           # [0, 1]
sent_norm = (avg_sent + 1) / 2        # [0, 1]
groups_norm = groups_count / max_groups  # [0, 1]
pr_norm = pagerank_normalized         # [0, 1]

# Poids configurables
w_score, w_sent, w_pr, w_groups = 0.35, 0.25, 0.20, 0.20

score_invest = 10 * (
    w_score * score_norm +
    w_sent * sent_norm +
    w_pr * pr_norm +
    w_groups * groups_norm
) / (w_score + w_sent + w_pr + w_groups)
```

### 5.4 Super Score (expert)

9 composantes + bonus optionnel :

```python
# Composantes positives (toutes normalisées 0-1)
quality = w_sent * nz_sent + w_wilson * nz_wilson
consensus = w_breadth * nz_breadth + w_groups * nz_groups
network = w_pr * nz_pagerank
dynamic = w_mom * nz_momentum + w_acc * nz_accel
stability = w_stab * (1 - ci95)

# Pénalité
penalty = w_polar * nz_polarisation

# Score de base
super_score = 10 * (
    (quality + consensus + network + dynamic + stability) / sum_weights
    - penalty / (penalty_weight + sum_weights)
)

# Bonus persistance (optionnel)
if use_persist:
    persist_frac = fraction_of_bins_in_top_R
    if multiplicative:
        super_score *= (1 + w_persist * persist_frac * (1 + gain))
    else:
        super_score += 10 * w_persist * (1 + gain) * persist_frac
```

---

## 6. Données des KOLs

### 6.1 Liste des 60+ groupes Telegram

Chaque groupe a un score de conviction (6-10) et des remarques :

| Score | Groupes | Caractéristiques |
|-------|---------|------------------|
| **10/10** | overdose_gems_calls, cryptorugmuncher, thetonymoontana | Winrate extrême, conviction maximale |
| **9/10** | marcellcooks, PoseidonTAA, Carnagecalls, MarkGems | Très peu de calls mais très bons |
| **8/10** | ghastlygems, slingdeez, archercallz, LevisAlpha, darkocalls... | Bonne conviction, différents styles |
| **7/10** | shahlito, sadcatgamble, veigarcalls, Luca_Apes... | Plus de calls, moins long terme |
| **6/10** | houseofdegeneracy | Intéressant mais moins fort |

### 6.2 Synergies entre groupes

Certains groupes sont liés et doivent être considérés ensemble :
- **LevisAlpha + dylansdegens + jsdao** : si en lien, signal fort
- **shahlito + marcellcooks** : si en lien, très bien
- **BossmanCallsOfficial** : à croiser avec Levis, Dylans, Shas, Marcell

### 6.3 Groupes spéciaux

- **cryptorugmuncher** (10/10) : Explique les rugs et scams - signaux **négatifs** à intégrer
- **thetonymoontana** (10/10) : Projets communautaires, souvent bullish
- **PoseidonTAA** (9/10) : Orienté analyses techniques

---

## 7. Fonctionnalités de détection

### 7.1 Flips de sentiment

Détection quand un token passe de négatif à positif (ou inverse) :

```python
def flip_detector(df, win_h=12, thr=0.1):
    mid = now - timedelta(hours=win_h)
    s_before = sentiment[date < mid].mean()
    s_after = sentiment[date >= mid].mean()

    # Flip si franchissement du seuil
    if (s_before < -thr and s_after > +thr) or \
       (s_before > +thr and s_after < -thr):
        return True
```

### 7.2 Newcomers

Tokens récemment apparus et liés aux leaders :

```python
def newcomers(df, hours_recent=24, top_k_leaders=5, npmi_min=0.1):
    # Leaders = top tokens par mentions
    leaders = top_tokens(df, k=top_k_leaders)

    # Tokens vus pour la première fois dans les N dernières heures
    first_seen = df.groupby("token")["date"].min()
    candidates = first_seen[first_seen >= cutoff]

    # Garder ceux avec forte co-mention (NPMI) avec un leader
    for token in candidates:
        npmi_with_leaders = max(npmi(token, leader) for leader in leaders)
        if npmi_with_leaders >= npmi_min:
            yield token
```

### 7.3 Consensus inter-groupes

Token mentionné dans le Top-K de plusieurs groupes :

```python
def consensus_table(scores, top_k=5, min_groups=2):
    # Garder Top-K par groupe
    topk = scores[scores["rank_in_group"] <= top_k]

    # Agréger par token
    consensus = topk.groupby("token").agg(
        groups_count = nunique("group"),
        groups_list = list("group"),
        avg_sent = mean("sentiment"),
        avg_score = mean("score_conviction")
    )

    # Filtrer par minimum de groupes
    return consensus[consensus["groups_count"] >= min_groups]
```

---

## 8. APIs et intégrations externes

### 8.1 Telegram (Telethon)

- **Authentification** : api_id + api_hash + phone
- **Méthode** : `GetHistoryRequest(limit=50)` par groupe
- **Rate limiting** : 1 seconde entre requêtes (basique)
- **Session** : Fichier `.session` pour persistance

### 8.2 DeepSeek (résumés)

- **Endpoint** : `https://api.deepseek.com/v1/chat/completions`
- **Modèle** : deepseek-chat
- **Format de sortie** structuré :
  1. Description (type + fonction)
  2. Catalyseurs (2-4 bullets)
  3. Risques (2-4 bullets)
  4. Sentiment global (Optimiste/Prudent/Négatif)

### 8.3 Prix des tokens

Plusieurs sources supportées :

| Source | Usage | Limite |
|--------|-------|--------|
| **Dexscreener** | Recherche paires | Rate limit modéré |
| **GeckoTerminal** | OHLCV historique | Pagination |
| **Birdeye** | Solana principalement | API key requise |
| **CoinGecko Pro** | Coins listés | API key requise |

Caches implémentés :
- `pair_cache.json` : Association token -> paire (TTL 7j)
- `ohlcv/` : Données OHLCV (TTL 3j)
- `token_hints.json` : Mappings manuels

---

## 9. Configuration et paramètres

### 9.1 Session state Streamlit

Tous les paramètres sont persistés entre les pages via `st.session_state` :

```python
# Période
period = "24h"
use_custom_period = False
custom_start_date, custom_start_time = None, None
custom_end_date, custom_end_time = None, None

# Sentiment
use_hf = False
w_hf, w_vader, w_crypto = 0.50, 0.35, 0.15
gain_sent = 1.30
rule_weight = 1.0
group_alpha = 1.0
alias_no_dollar = True

# Scoring
mentions_alpha = 0.6
tau_hours = 12.0

# Graphe
score_graph_on = True
gamma_struct = 0.30
wA, wC, wPRT = 0.60, 0.40, 0.20
npmi_min_sc = 0.10
min_cooc_sc = 3

# Données
RAW_ALL = pd.DataFrame()  # Dataset brut fusionné
RAW_DF = pd.DataFrame()   # Dataset avec sentiment calculé
```

### 9.2 Fichiers de cache

```
cache/
├── api_keys.json       # Clés API (à sécuriser!)
├── pair_cache.json     # {token: {chain, address, pair}}
├── token_hints.json    # Mappings manuels user
└── ohlcv/              # Données OHLCV par token
    ├── PEPE_solana.parquet
    └── ...
```

---

## 10. Problèmes connus et améliorations prévues

### 10.1 Problèmes critiques

1. **Sécurité** : API keys hardcodées dans le code
2. **Pas de git** : Aucun versioning
3. **Pas de tests** : Aucun test unitaire

### 10.2 Problèmes techniques

1. **Performance** : CryptoBERT recalcule à chaque changement de filtre
2. **Scalabilité** : O(n²) pour le graphe groupe-groupe
3. **Rate limiting** : Trop basique pour Telegram

### 10.3 Améliorations planifiées

1. **Nouvelle app clean** avec uniquement les meilleures fonctionnalités
2. **Intégration API prix** (vs ATH, volume, holders)
3. **Backtesting avancé** avec XGBoost + Optuna
4. **Si succès** : Migration vers Next.js + Supabase + Stripe

---

## 11. Annexes

### 11.1 Dépendances principales

```
streamlit>=1.36
pandas>=2.0
numpy>=1.25
plotly>=5.22
vaderSentiment>=3.3.2
transformers>=4.40
torch>=2.2
networkx>=3.2
telethon>=1.28
```

### 11.2 Commandes utiles

```bash
# Lancer l'app
cd ConvictionApp
streamlit run app.py

# Exporter les messages Telegram
python exportfinaljson.py

# Voir les IDs des groupes
python getID.py
```

### 11.3 Format JSON d'export

```json
{
  "GroupName": [
    {
      "id": 12345,
      "date": "2024-01-15T14:30:00",
      "text": "🚀 $PEPE looking bullish, might moon soon",
      "conviction": 8,
      "remark": "très bonne conviction"
    }
  ]
}
```

ou format plat :

```json
[
  {
    "id": 12345,
    "date": "2024-01-15T14:30:00",
    "group": "GroupName",
    "text": "🚀 $PEPE looking bullish",
    "conviction": 8,
    "remark": "..."
  }
]
```
