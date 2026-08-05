# Operational Backlog

> Reconcilié le 2026-05-22 contre l'état réel (l'ancienne version datait du 17-18 mai,
> ~moitié déjà fait/périmé). Source de vérité pour les données volatiles (allocations,
> blacklist, slippage) = `scoring_config` en DB, PAS ce fichier.
> 🎯 Décisions stratégie : `tasks/strategy_candidates.md`.

## 🛑 Live status (réel — 2026-06-05)
- **LIVE OFF COMPLET** : `live_trading.enabled = false` (killé le 05/06, deck SOL FAST_MCAP saignait : 7d WR 28.6%, brut −$3.19, net pire). 0 position ouverte au kill. Backup `data/rt_trade_config_pre_live_kill_all_20260605T202008Z.json`. Resume = flip `enabled=true` (allocs SOL `{FAST_TP50_SL30_MCAP_S40:1}` toujours présentes).
- **ETH jamais live** (`eth_live_enabled=false`) — l'ETH positif est paper-only.
- Config inchangée au cas où resume : `slippage_buy_bps=1000`, `slippage_sell_bps=500`, `max_position_sol=0.012`.
- 🔴 **Sweep ATA CASSÉ** : Helius à court de quota (`-32429 max usage reached`) → `_solana_rpc_url()` sans fallback → sweep + close immédiat no-op silencieux. **44 ATA vides = 0.091 SOL (~$13.65) rent bloqué** (corrige l'ancienne note "wallet 0 ATA vide"). Wallet live `9t3yNhW…` = 0.666 SOL. Compter via Jupiter holdings, PAS Helius (renvoie garbage quand épuisé). Fix en attente user (top-up Helius / RPC fallback / sweep manuel).
- Lentille d'analyse par défaut : **`v_strategy_faithful_perf`** (jamais le shadow brut).
- Paper + shadow tournent normalement (inchangés). 8 crons GH verts (vérifié 05/06).

## 🔬 PISTE OUVERTE — tester le bon cheval en live (2026-07-05)

> Découvert le 05/07 en réconciliant shadow→live (802 trades `rt_live`, avr→05/06). **On a testé les MAUVAIS chevaux en live.**

- **Constat 1 — WR de base intact** : `SCALP_TP20_NOSL` (proxy régime) = WR ~58-64% chaque semaine depuis 3 mois, médiane +19%. Les memecoins ne sont PAS morts pour les petits pumps. Ce qui manque = régime "gros runners" (seule la semaine du 22/06 a eu du jus). Edge scalp existe mais fin (moyenne shadow +3.7%).
- **Constat 2 — les strats mises en live étaient déjà négatives EN SHADOW** sur la même période : BE25_TP80_SL30 (méd shadow −13%), FAST_TP50_SL30 (−13%), FAST_MCAP_S40 (−6%)… choisies par moyenne/fat-tail, pas par médiane. Le live n'a fait que confirmer médiocre→médiocre. **Pas des faux positifs effondrés — déjà mauvais.**
- **Constat 3 — drift shadow→live MODÉRÉE (~±10pp), pas catastrophique.** Sur gros N (BE25 n=177, FAST n=158) le live est même **meilleur** de +7pp. Casse le mythe "le slippage détruit tout". Coût réel ~10pp max.
- **🎯 Le bon cheval n'a JAMAIS été testé live** : `SCALP_TP20_NOSL` (méd shadow +19%, WR 64%, N~1000, stable 14j ET 30j). Le seul SCALP live était `SCALP_TP20_SL10_S30` (SL10 → cascade sur bruit slip, déjà −12% en shadow) — PAS un test du TP20_NOSL.
- **Candidats live-positifs historiques** (petit N) : `ETH_TP80_SL40_T2H` (+20% moy, n=10), `BE15_TP100_SL50` (+7.4% moy, n=20) — TP large / timeout long.

**Next actions** :
- [ ] **(A) `/simulate-live` sur `SCALP_TP20_NOSL`** (± variantes `AGE24_SCALP_TP15_*_S35` WR 66-70%) — slip/MEV/fees/latence réels. Zéro risque. Décide si l'edge survit avant tout cash. ⬅️ FAIRE EN PREMIER.
- [ ] (B) Si (A) OK : mini-deck live test micro-positions ($0.50-1) sur SCALP_TP20_NOSL + AGE variants, viser N≥30, mesurer paired-drift companion-shadow < 5pp.
- [ ] Valider via `v_strategy_faithful_perf` + `/ground-truth-strat-perf` (jamais shadow brut).
- [ ] ⚠️ Écarter la famille DTRAIL — artefact confirmé EN LIVE (DTRAIL10_ACT15_SL70 = −$45 réel, méd live −6 à −70%).

## ✅ Fait récemment (archive — détail en git + mémoire)
- **05/06** : LIVE killé complet (`enabled=false`) après audit deck SOL perdant ; découvert sweep ATA cassé par quota Helius (44 ATA / $13.65 bloqués) ; vérifié divergence live↔paper↔shadow saine (−2 à −3pp slip normal) ; réconcilié `tasks/` (live status + ATA). Détail mémoire : `live_trading_killed_all_jun5.md`, `helius_quota_exhausted_breaks_ata_sweep.md`.
- **22/05** : v14e.68 (rent leak close_ata fix, $2.96 récupérés) ; FAST60 killé live ; couche sim fidèle (vues `v_strategy_faithful_perf` + skills câblés) ; slip buy 500→1000 ; sim-align-gate crash fix (MAX_DRIFT) ; `_audit_shadow_strategies_coverage.py` créé.
- **18/05** : v14e.64 skills refactor + `_reconcile_bankrolls.py` + dedup_rules.md + cleanup.
- **17/05** : live enabled, allocations resets SOL/ETH, bankroll reconcile, v14e.59/60.
- Backups `_backup_*` du 12/05 droppés le 20/05.

---

## 📌 Quick reference (stable)

### Slippage (single source = `strategies.py`, sauf live tolerance en JSONB)
- SOL paper : `BUY_SLIPPAGE_BPS = 225`. SOL live tolerance (JSONB) : `slippage_buy_bps=1000`, `slippage_sell_bps=500`.
- ETH paper : `BUY 350 / SELL 650`. ETH live tolerance (JSONB) : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`.

### Cron schedules (GitHub Actions)
| Workflow | Schedule |
|---|---|
| `mega-sweep-48h` (SOL) | 02:00 UTC /2j (6 shards) |
| `mega-sweep-eth-48h` | 22:00 UTC /2j |
| `sim-align-gate` | 04:00 UTC daily (crash fix 22/05) |
| `nightly-outlier-monitor` | 04:30 UTC daily |
| `nightly-shadow-audit` | daily |
| `train-models` / `outcomes` / `daily-summary` / `kol-weekly-audit` | daily / ~2-4h / daily / weekly |

### Pre-deploy + deploy
```bash
bash scripts/pre_deploy_check.sh                       # py_compile + import smoke + pytest
git push origin master                                  # gh account: Maximus0Primus (owner)
ssh vps "cd /opt/TelethonIA && git pull origin master && systemctl restart kol-scraper"
# JSONB config (scoring_config/rt_trade_config/blacklist) reload auto par cycle — pas de restart requis.
```

### VPS
- Service `kol-scraper.service`, wd `/opt/TelethonIA`, python `scraper/venv/bin/python`.
- Logs : `journalctl -u kol-scraper -f` (ou skill `/logs`).
- Wallet SOL live : `9t3yNhWUV7f3EfyMAiFHrL6qDU8oT4rA9Agt8tSmBeSM`.
- ⚠️ `SOLANA_PRIVATE_KEY` PAS dans le `.env` local — ops wallet/sweep uniquement sur le VPS.

## [Aug 5] Backtest entry-timing — chercher un edge réel

Contexte: 74-82% des runners plongent <-40% AVANT de monter (vs 10-27% en avril).
Tout SL 30-40% se fait sortir avant la hausse. Le problème est l'ENTREE, pas l'exit.

- [x] Charger price_ticks 30j (673 tokens SOL) + t0 = premier call par token
- [x] Grille ENTREE: immediate / wait 15-30-60min / dip -20/-30/-40/-50% / reclaim apres dip
- [x] Grille EXIT: reutiliser sim_engines.simulate_fixed (slip prod) TP/SL/horizon
- [x] Metriques: moyenne arith + geometrique (JAMAIS mediane), n, sum-ex-best, semaines positives
- [x] Test apparie sur intersection des tokens (jamais comparer sur N different)
- [x] Verdict honnete: si rien ne bat les couts (~0.4pp/trade a $10), le dire

### Resultat (Aug 5) — RESULTAT NUL, mais un piege data trouve

- [x] Harness construit: `scripts/_entry_timing_backtest.py` (466 tokens, 30j,
      12 regles d'entree x 7 configs d'exit, exits via sim_engines = slip prod)
- [x] 1er run: dip50 = +12.6%/trade, +19.7pp apparie vs immediate, 5/5 semaines +
- [x] Test de realisme du fill (`fill_lag=1`) => 0/5 semaines positives sur les
      21 configs, puis lag2 repositif. Oscillation = signature d'artefact.
- [x] CAUSE: `price_ticks` est un LOG multi-sources (jupiter/fast/full entrelaces
      toutes les 11-20s). jupiter->fast p1 = -85.8%, fast->jupiter p99 = +640%.
      2.63% des transitions jupiter->fast montrent < -40% en <=30s.
      Toute regle qui declenche sur un print bas selectionne la source qui cote bas.
- [x] Re-run mono-source: Jupiter => TOUT negatif (meilleur -0.0%, geo -7.3%).
      DexScreener => meilleur +0.9% arith / -4.2% geo, et immediate BAT les dips
      (ordre inverse entre sources = bruit).
- [x] Prod saine: `paper_trader._decision_price` resout UNE source par strategie;
      `sim.py._filter_ticks_by_source` filtre correctement. Le bug etait le mien.

VERDICT: aucun edge d'entry-timing. Ni retard, ni dip-buy, ni dip+reclaim.

Prochain fil (non fait): verifier que la hausse des dips <-40% mesuree sur
paper_trades n'est pas induite par un changement de `orch.source` par strategie.

### [Aug 5] Recherche large avec null de permutation — 2 pepites trouvees

Git: pas de rupture mecanique. Commits orchestration/price_source = 18-19 avril
(AVANT les bonnes semaines). v14e.65 (20 mai) ne touche que pnl_usd, pnl_pct reste
GROSS. Mix de sorties SCALP_TP20_NOSL stable sur 18 semaines, gain TP ameliore 17->28%.

- [x] Scan features token: 4778 candidats, 3 survivants REELS vs 6 sous HASARD => zero edge
- [x] Scan axe KOL: 161 survivants REELS vs 91-99 sous HASARD => signal reel
- [x] slingoorioyaps: 90 strats survivantes vs 12 au hasard (7.5x). +5.9%/+5.8% train/test
- [x] bounty_journal (blackliste): 47 vs 16 (3x) MAIS best combos = TRAIL => escompter
- [x] olympeqg: 0/358 survivants, -4.8% train / -10.1% test, 12-25% du flux

ACTIONS PROPOSEES (non appliquees, en attente validation):
- [ ] Bannir olympeqg du blacklist SOL => +1.3 a +2.3pp/trade sur toutes les strats.
      FAST_TP50_SL30_MCAP_S40 (deja dans le deck) passe -0.76% => +1.39%
- [ ] Ajouter slingoorioyaps x TP100_SL40 en shadow dedie (+18.6%, 13/17 sem+,
      ~4.9 trades/sem = +$9/sem a $10). TAILLE FIXE (geo ~= 0, ne pas composer)
- [ ] Re-examiner bounty_journal hors strategies TRAIL avant de le debannir

### [Aug 5] Scan #2 — features derivees KOL (null de permutation)

Pistes du research_log liquidees en une requete de fill-rate:
- `whale_new_entries` ("seul signal robuste confirme", correl +0.578) = **0% rempli sur RT**
- `ml_pred` = 0% (ML off), `unique_kols` = toujours 1 => inutilisables cote temps reel

Teste (531 strats, ~600k lignes, train/test):
- [x] Forme recente du KOL (moy de ses 10 calls precedents, causal, zero look-ahead)
      => **ANTI-predictif**: forme>0 donne −4.35% en test, forme<0 donne −2.91%.
      Tue l'idee de whitelist dynamique auto-adaptative. L'edge KOL est une identite
      STABLE, pas une forme recente.
- [x] Heure UTC: signes s'inversent entre train et test sur les 4 tranches => zero signal.
      Ferme la piste "heure optimale" du research_log.
- [x] **Silence du KOL (gap depuis son call precedent) = EFFET REEL, dose-response**

| bucket | moy totale | sans olympeqg | n |
|---|---|---|---|
| rafale <1h  | −8.74% | −5.43% | 129k |
| 1-6h        | −5.05% | −3.12% | 123k |
| 6-24h       | −1.27% | −1.31% | 137k |
| **24-72h**  | **+2.33%** | **+2.22%** | 124k |
| >72h        |  0.00% | +0.03% | 122k |

Monotone, survit au retrait d'olympeqg (57% du bucket rafale). Ameliore les 8
strategies testees (+0.1 a +6.1pp) mais ne rend AUCUNE strategie fiable seule
(train/test s'inversent par strategie).

### 🏆 LE COMBO — composants valides SEPAREMENT puis empiles

`slingoorioyaps` x `TP90_SL40` x `gap>=24h`:
n=33 (32 tokens uniques), **moy +31.6%**, **geom +12.2%** (1ere geom positive de la
session), WR 61%, 14 trades >+50%, somme sans le max = +894pp (=> pas un runner unique).
Version poolee TP70/90/100_SL40: n=96, **train +37.4% / test +37.7%**, WR 64%.
Cadence: **1.9 trades/semaine** => a $10 fixe ~ +$6/sem.
⚠️ n=32 tokens uniques = petit. La defense: KOL valide par permutation (90 vs 12) ET
gap valide par dose-response sur 600k lignes, INDEPENDAMMENT, avant combinaison.

- [ ] Cabler `slingoorioyaps + gap>=24h` en shadow dedie pour accumuler du N propre
- [ ] Envisager un filtre global "gap KOL >= 24h" (reduction de risque sur tout le deck)

### [Aug 5] DEPLOYE — v14e.70 (commit c252943)

- [x] **olympeqg banni** (`kol_chain_blacklist.solana`, 21 -> 22 KOLs).
      Backup `data/paper_trade_config_pre_olympeqg_ban_20260805.json`.
      Blacklist = skip MAIN uniquement, le shadow continue d'enregistrer
      (paper_trader.py:1189-1192) donc olympeqg reste re-evaluable.
- [x] **Nouveaux filtres opt-in** `kol_whitelist` + `min_kol_gap_hours`
      (absent = pas de gate => zero impact sur les strategies existantes).
      Gap lu en DB via nouvel index `idx_paper_trades_kol_created` (23ms,
      puis 0.004ms en cache), token courant exclu, fail-closed si inconnu.
- [x] **A/B 2x2 en shadow**: TP90_SL40 (base) / KOLW / GAP24 / KOLW_GAP24
- [x] 9/9 tests unitaires du gate + pre_deploy_check PASSED (141 tests)
- [x] Deploye VPS, service actif, zero erreur

### [Aug 5] Validation mois par mois — 1 confirme, 1 corrige, 0 nouveau

Donnee RT commence avril 2026 => 120j EST la fenetre maximale disponible.
Critere le plus dur possible: performance par mois civil.

| KOL | mois + | moy | pire mois | n |
|---|---|---|---|---|
| **slingoorioyaps** | **4/4** | +20.2% | **+13.4%** | 231 |
| dddegens | 3/4 | +20.8% | -0.4% | 120 |
| ALSTEIN_GEMCLUB | 3/3 | +16.3% | +11.2% | 45 |
| bounty_journal | **2/4** | **-1.4%** | -17.9% | 249 |
| olympeqg | **0/3** | -7.7% | -15.5% | 1335 |

- [x] slingoorioyaps CONFIRME: 4/4 mois, tient a travers le collapse de mai ET
      le regime de juin. Ecart vs reste du flux +18 a +40pp chaque mois.
- [x] **bounty_journal RETROGRADE** — je le proposais au deban, le prisme mensuel
      dit 2/4 mois et moyenne NEGATIVE. Ne pas debannir. (Son score au scan
      precedent venait de strategies TRAIL, deja suspectes.)
- [x] **dddegens = piege parfait**: moyenne +24.6% en test MAIS 230 survivants
      reels contre **208 sous permutation** => zero signal. Moyenne elevee,
      edge nul. A comparer a slingoorioyaps: 104 reels contre **2** au hasard.
      => La moyenne seule ne prouve rien, meme validee train/test.
- [x] ALSTEIN_GEMCLUB / letswinallgems: n insuffisant (echouent le seuil 15/15)

VERDICT: aucun nouveau KOL validable. slingoorioyaps reste le seul.

### [Aug 5] Filtre sentiment — test HORS SELECTION sur le deck pre-existant

Le balayage des sorties m'avait sorti FAST_TP50_SL30_S40 a geom +4.0% => $100 devient
$457 en 4 mois a 20% du capital. **C'ETAIT DE LA SELECTION.** Null en rebattant le
sentiment: la meilleure geometrique atteint 5.1% et 4.2% par HASARD contre 5.5% en reel.
Choisir le sommet d'un balayage de 239 strategies n'est pas justifie.

Ce qui EST valide par ce meme null: le filtre lui-meme. Moyenne de tout le lot
+7.19% reel contre -3.63 / +2.59 / +1.15 au hasard, p90 des geometriques +0.5
contre -6.3 / -1.2 / -3.7. **Toute la distribution est decalee**, pas juste le sommet.

Test propre = appliquer le filtre aux 3 strategies deja dans le deck (choisies il y a
des mois, zero selection possible), blacklist a jour (olympeqg deja banni):

| strategie | sans filtre | AVEC filtre | train | test | $ a taille fixe $10 |
|---|---|---|---|---|---|
| FAST_TP50_SL30_MCAP_S40 | +2.0% (n=621) | **+17.8%** (n=55) | +18.2 | +16.9 | **+$98** |
| FAST60_TP70_SL50_NZ_S40 | +3.8% (n=732) | **+13.7%** (n=58) | +15.6 | +10.4 | +$80 |
| TP50_SL40_S35 | +3.0% (n=927) | **+13.0%** (n=64) | +16.3 | +6.7 | +$83 |

Les 3 montent de +10 a +16 points. Les 3 positives dans les DEUX moities.
Cadence ~3.2 trades/semaine (55-64 trades sur 120j).

=> L'edge par trade est maintenant assez gros pour que **la TAILLE soit le levier**,
   plus la frequence. A $10 c'est +$25/mois; a $100 c'est +$250/mois.
   Le facteur limitant devient le slippage live, pas le signal.

- [ ] `/simulate-live` sur FAST_TP50_SL30_MCAP_S40 + filtre sentiment, a $50 et $100
      => c'est LA question ouverte: l'edge de +17.8% survit-il au slip reel?
- [ ] Cabler le filtre sentiment en shadow (meme mecanique que min_kol_gap_hours)

### [Aug 5] /simulate-live — FAST_TP50_SL30_MCAP_S40 + sentiment 0.5-0.6 => GO CONDITIONNEL

Pool: n=69 dedup 24h, blacklist a jour, 20/04 -> 05/08 (107j), 4.5 trades/sem.
Liquidite mediane $27 580 => $100 = 0.36% du pool (marge jusqu'a ~$500).

**Phase 0**: shadow_dedup n=68 vs paper_main-sans-olympeqg n=69 => structure OK
(aucune inflation par re-entrees). Ecart de moyenne 0.85pp = bruit a ce N.
Bonus: olympeqg faisait -13.86% sur 9 trades main cette semaine => ban valide a chaud.

**EV**: brut +12.38% -> net **+11.98%** apres couts live (-0.4pp). Le cout est
negligeable face a un edge de 12%. WR 46%, PF 2.19, gain median +44.1% vs perte
mediane -14.6% (R:R 3:1). Sharpe/trade 0.27 => annualise ~4.1.

**Walk-forward** (69 trades dans l'ordre reel, depuis $1000):
fixe $50 -> $1 413 | fixe $100 -> $1 827 (DD 25%) | 20% du capital -> $4 021 (DD 23%)

**Monte Carlo** 10k tirages, 100 trades (~7 mois), depuis $1000:
fixe $100 -> median $2 184, P10 $1 641, DD median 11%, **ruine 0.0%**
20% capital -> median $7 297, P10 $2 611, DD median 31%, ruine 0.0%

**Stress**: 4/5 PASS (slippage double, MEV, crise de liquidite, volume -50%).
Le "cold streak P20-P40" du skill echoue mais il est DEGENERE ici: a WR 46% cette
bande ne contient que des perdants (0/14), il simule 100 pertes d'affilee.
Remplace par un cold streak realiste = rejouer le pire tiers chronologique
(EV +2.58%): median $1 212, P10 $768, ruine 0.1% => survit.
Pire serie de pertes consecutives REELLE: 5.

**Marge de securite**: l'edge tient jusqu'a 8pp de cout additionnel (+4.38% EV).
La derive historique live<->paper twin mesuree sur SL=30% est -2 a -5pp
=> attendre **+7 a +10%/trade en live**, pas +12%.

**Confiance**: apres 30 trades, 6.1% de chances d'etre sous le capital initial;
apres 50 trades, 2.1%.

- [ ] Demarrer a $50 fixe sur 20-30 trades, mesurer la derive appariee vs le
      shadow compagnon. Si < 5pp, passer a $100 puis envisager le % du capital.
- [ ] Cabler le filtre sentiment en shadow AVANT (le filtre n'existe pas encore
      dans le code — la simulation est faite sur le pool historique reconstitue)

## ⏳ EN ATTENTE D'ACCUMULATION — ne rien decider avant

Deploye le 2026-08-05. **Ne pas toucher, ne pas promouvoir, ne pas passer en live
avant d'avoir du N propre.** Les chiffres ci-dessous viennent d'un pool historique
reconstitue par jointure SQL — RIEN n'a encore ete forward-teste.

Bras shadow qui accumulent (tous sur l'exit FAST_TP50_SL30_MCAP_S40, deja au deck,
donc le seul facteur qui varie est la bande) :

| bras | bande sentiment | cadence attendue | EV historique |
|---|---|---|---|
| SENT30_70_FAST_TP50_SL30_MCAP_S40 | 0.30-0.70 | ~12/sem | +8.6% |
| SENT45_65_FAST_TP50_SL30_MCAP_S40 | 0.45-0.65 | ~6/sem | +12.4% |
| SENT50_60_FAST_TP50_SL30_MCAP_S40 | 0.50-0.60 | ~3/sem | +17.8% |

Plus le 2x2 KOL/gap de v14e.70 (KOLW / GAP24 / KOLW_GAP24 vs TP90_SL40 nu).

**Quand revenir dessus :**
- [ ] **~1er septembre** (4 semaines) : premier point d'etape. A ~12/sem le bras
      large aura ~48 trades, les deux autres ~24 et ~12. Trop peu pour trancher,
      mais assez pour detecter une divergence grossiere vs l'historique.
- [ ] **~1er octobre** (8 semaines) : bras large a ~96 trades => decision possible
      sur la LARGEUR de bande. C'est la vraie question ouverte.
- [ ] Le bras 0.50-0.60 a ~3/sem mettra ~5 mois a atteindre n=69. Ne pas
      l'evaluer seul avant, il sera toujours dans le bruit.

**Ce qu'il faudra verifier a chaque point d'etape :**
- [ ] EV live du bras vs son EV historique (divergence > 5pp = alerte)
- [ ] Comparer au bras nu FAST_TP50_SL30_MCAP_S40 sur les MEMES tokens (paire),
      jamais en agrege sur des N differents
- [ ] Classer a la moyenne + verifier semaines positives, JAMAIS a la mediane
- [ ] Refaire tourner le null de permutation sur la nouvelle donnee

**Ne pas faire en attendant :**
- Ne pas re-balayer les features token (axe mort: 3 survivants vs 6 au hasard)
- Ne pas re-tester l'entry-timing (mort sur source coherente)
- Ne pas ajouter de bras: chaque bras en plus dilue la puissance statistique

### [Aug 5] Croisement du filtre sentiment (question: teste-t-il comme le mega sweep ?)

Reponse honnete: NON, il ne l'etait pas. Teste seul et empile avec le gap, jamais
croise systematiquement. Fait maintenant, sur 6 strategies, bande 0.30-0.70:

| axe croise | ni l'un ni l'autre | axe seul | sentiment seul | LES DEUX | n |
|---|---|---|---|---|---|
| slingoorioyaps | -1.3 | +6.0 | +3.0 | **+12.4** | 106 |
| gap KOL >= 24h | +0.2 | -2.2 | -0.4 | **+8.2** | 963 |
| score >= 45 | -0.2 | -1.5 | +1.2 | +5.6 | 1105 |
| age < 6h | -0.7 | -1.0 | -0.3 | +5.5 | 1405 |
| mcap >= 100k | -0.5 | -1.4 | +3.1 | +4.0 | 932 |
| liq >= 30k | -1.3 | 0.0 | +3.4 | +3.5 | 680 |

- Les 4 features token mortes le restent croisees: leur colonne "axe seul" est
  a -1.5/0.0 et "les deux" ne bat pas "sentiment seul". Confirmation.
- slingoorioyaps x sentiment est SUR-additif (+12.4 vs +10.3 attendu si additif).
- gap x sentiment semblait interactif en agrege (+8.2 sur n=963) MAIS ne se
  reproduit PAS sur l'exit cible: 0.30-0.70 seul = EV +8.6% / $1798 total contre
  0.30-0.70+gap = EV +6.4% / $505. Le gap DEGRADE ici.
  => **Pas de 4e bras sentiment+gap.** Les 3 bras deployes sont le bon jeu.

Lecon: une interaction vue en poolant des strategies ne survit pas forcement a
une strategie donnee. Toujours reverifier sur l'exit cible avant d'ajouter un bras.

### [Aug 5] v14e.72 — nouveaux axes dans le mega sweep + BUG 33% de calcul mort

**BUG TROUVE**: `_mega_apply_filter` lit `rt_buy_sell_ratio` et `kol_win_rate`
depuis v14e.43, mais AUCUN des deux n'etait dans le `select` des deux
constructeurs d'univers. Donc `(None or 0) >= seuil` = toujours False.
**7 arms sur 21 (33% de la dimension filtre) matchaient ZERO trade** et bruaient
du calcul CI depuis avril: BSR52, BSR55, NOZEROLIQ_BSR52, NOZEROLIQ_BSR55,
KW34, KW26, BSR_MCAP. Silencieux: pas d'erreur, juste "aucun resultat".
Verifie apres fix sur 8j reels: BSR52 passe de 0% a 82.2% de match, BSR_MCAP 24.8%.
⚠️ Toute conclusion passee "BSR/KW ne marchent pas au sweep" est donc SANS VALEUR
— ces arms n'ont jamais tourne. A rejuger au prochain sweep.

**Ajout des axes de cette session** (7 nouveaux arms):
SENT30_70, SENT45_65, SENT50_60, SENT_NOHYPE, GAP24, NOBURST, SENT30_70_GAP24.
Le sentiment est un U INVERSE donc c'est une BANDE, pas un seuil — aucun arm
`>= x` existant ne pouvait l'exprimer, d'ou de nouveaux noms.

**Enrichissement** `_mega_enrich_universe()`, appele une fois par run:
- `kol_gap_h` derive de l'univers lui-meme (deja 1 ligne/token avec kol_group +
  created_at) => zero requete supplementaire
- `sentiment` via une fetch paginee de kol_mentions, matche sur
  (kol_group, resolved_ca) premier message. **97.9% de match verifie sur 8j reels**

**Cout**: dimension filtre effective 14 -> 28 (x2). Shards passeraient de ~2h a
~4h, trop pres du cap GH 6h => strat-shard 2 -> 3 tiers, soit **9 shards** au lieu
de 6, ~2-2.7h chacun. Le merge agrege par pattern, il encaisse n shards.

- [ ] Prochain cron (tous les 2j a 02:00 UTC) => premier sweep avec les 28 arms.
      Verifier dans le CSV que les colonnes filter=SENT*/GAP24/BSR*/KW* ont bien
      des lignes avec n>0 (si n=0 sur un arm, l'enrichissement a echoue).
- [ ] Rejuger BSR/KW une fois qu'ils auront VRAIMENT tourne

### [Aug 5] ETH: mega sweep casse depuis >2 semaines + correction sharding SOL

**1. ETH sweep KO a chaque run** (8 echecs consecutifs verifies, depuis au moins
le 21/07). Cause reelle (le log d'echec pointait le merge, pas la source):

```
chain=eq.ethereum & source=eq.rt & created_at>=... & order=created_at & limit=1000
=> 500 {"code":"57014","message":"canceling statement due to statement timeout"}
```

**Aucun index sur `(chain, created_at)`.** Sur SOL la requete passe car les lignes
sont denses. ETH est tombe a ~0.5% du volume => Postgres balaie tout l'index
`created_at` sans jamais remplir sa page de 1000 lignes => timeout.
Fix: `idx_paper_trades_chain_created (chain, created_at DESC)`.
Verifie par EXPLAIN ANALYZE: Index Scan Backward, **169ms** au lieu du timeout.

⚠️ **Mais ca ne rend pas le sweep ETH utile pour autant.** Volume ETH par semaine:
3, 9, 5, 15, 13, 3, 2, 2, 9, 6 tokens. Soit ~30 tokens sur la fenetre du sweep.
Reparti sur sources x smoothings x polling x filters x strategies, chaque cellule
aura une poignee de trades. Le workflow passera au vert mais la sortie ne sera pas
exploitable tant que le flux ETH ne remonte pas.
=> **PAS un bug** (precision user 05/08): le support ETH a ete construit PENDANT
   l'ETH season. Elle est finie, et ETH est structurellement bien moins utilise
   que SOL pour les memecoins. La chute 74 tokens/sem (fin avril) -> 2-9 (juillet)
   est le MARCHE, pas la pipeline. Rien a reparer cote scraper.
   Consequence operationnelle: le sweep ETH brule 3 runners toutes les 48h pour
   ~30 tokens. Ne pas le supprimer (l'ETH season peut revenir) mais le rendre
   conditionnel au volume.

**2. Mon sharding v14e.72 etait sous-dimensionne.** Le commentaire du code
annoncait "~1.5-2h/shard" — PERIME. Mesure reelle (run 30977642076 du 05/08):
197 / 219 / 235 / 253 / 254 / **273** min. Le pire etait deja a 78% du timeout.
Avec +71% de charge estimee par taux de match, 2 tiers => ~467min = au-dela du
cap GH 6h. Corrige 2 -> **4 tiers = 12 shards**, pire shard ~233min, 33% de marge.
Lecon: ne jamais faire confiance a un commentaire de perf, mesurer le run reel.

- [ ] Verifier au prochain cron que le sweep ETH passe au vert
- [ ] ~~Investiguer le tarissement du flux ETH~~ ANNULE: c'est le marche (fin de
      l'ETH season), pas un bug. Voir precision ci-dessus.
- [x] **Garde de FRAICHEUR posee sur le sweep ETH** (v14e.72b). Pas une garde de
      volume total: l'univers du sweep compte encore 313 tokens, mais 236 sont
      anterieurs a juin. On rejouait 3 runners toutes les 48h sur un univers fige
      a 75% pour y ajouter ~1.4 token. Sans donnee neuve, un re-run ne peut pas
      produire de conclusion neuve.
      Regle: skip si < 20 tokens ETH distincts sur 14j. Actuellement **16 => SKIP**.
      Au rythme actuel (~0.7/jour) le sweep se declenchera ~1x/mois.
      Si une ETH season revient (74/sem = ~148 par 14j) il repart tout seul.
      `workflow_dispatch` a un input `force` pour outrepasser.
      La garde FAIL OPEN: si la requete casse, le sweep tourne quand meme
      (une garde en panne ne doit jamais masquer le job qu'elle protege).

### [Aug 5] Chasse au VOLUME — 3 pistes testees, 2 mortes, 1 inexploree

Constat: la strat filtree ne fait que 11 trades/sem (195 sur 4 mois). Entonnoir:
8 278 tokens mentionnes -> 2 818 tradés en RT (165/sem) -> 585 passent les gardes
de la strat (score40 + mcap 30-500k) -> 195 passent la bande de sentiment.

**1. Relacher les gardes de la strat => NON.** Sur le MEME exit, la progression
est monotone dans le mauvais sens quand on relache:
  sans garde $450 (n=336) -> +score40 $1 230 (n=250) -> +mcap $1 402 (n=195).
Les versions a fort volume (FAST60_TP50_SL30, TP50_SL30: 25/sem) font un peu plus
de dollars bruts mais perdent toutes en mai (3/4 mois au lieu de 4/4) et leur
geometrique s'effondre a -10 a -19%. **Les gardes achetent de l'edge.**

**2. Prendre les re-entrees (que le dedup 24h jette) => NON.**
Ca semblait etre $649 gratuits (231 trades, EV +2.8%). Decompose par rang dans la
bande: 1ere +7.0%, 2eme +4.4%, 3eme +3.7%, 4-5eme -1.6%, 6eme+ +3.0%.
MAIS verification sur les 7 strategies HORS bande, n=2098 sur la 2eme entree:
**EV -1.91% contre -0.80% pour la 1ere**, coherent sur les deux moities.
La population dit que la re-entree est PIRE. Mon +4.4% repose sur n=130 dont 67%
viennent du seul mois de juillet. A priori de population contraire + petit N
concentre => NON RETENU.

**3. Chemin batch => INEXPLORE.** `batch_trading_enabled=false`. Zero trade batch
sur 120j (les 439 tokens "non-rt" sont du `rt_live`, arrete le 5 juin). C'est la
seule source de volume additionnelle non testee. Necessiterait de l'activer en
shadow pour savoir. Prudence: il a ete desactive, probablement pour une raison.

**Conclusion honnete**: on ne peut pas avoir volume ET edge avec ce signal. Chaque
relachement echange de l'edge contre du volume a mauvais prix. Le vrai levier
reste la TAILLE: a +7.2%/trade et 11 trades/sem, passer de $10 a $100 puis $500
la position. Contrainte = liquidite (pool median $27.6k, donc $500 = 1.8% du pool,
encore sous le seuil de 2% ou le fill partiel commence).

- [ ] Si le sweep manuel ne sort rien: envisager d'activer le batch en SHADOW
      seulement, pour mesurer sans risque si c'est du volume additionnel utile

### [Aug 5] ⚠️ Ce que le mega sweep peut et NE PEUT PAS faire

Question user: "le sweep va vraiment permettre de trouver la meilleure strategie ?"
Reponse verifiee dans le code: **NON, pas pour designer un gagnant.**

Taille de la matrice: 3 sources x 9 smoothings x 10 polling x 28 filters x 4 age_bands
x ~500 strategies = **15 120 000 cellules**. Le code lui-meme parle de **371k tests
eligibles** sur le run du 26 avril.

Et le gate FDR est **DESACTIVE**. Commentaire dans analyze_mega_sweep.py:
  "--require-fdr ... default OFF — Bonferroni-corrected FDR on 371k tests was
   nuking the entire top_robust list to zero on the Apr 26 run"
Le workflow ne passe PAS --require-fdr. Donc `_mega_sweep_top_robust.csv` est le
**maximum de ~371 000 tests sans correction pour tests multiples**.

C'est exactement le piege que j'ai demontre aujourd'hui a plus petite echelle:
sur 239 strategies seulement, le null de permutation atteignait 5.1% et 4.2% de
geometrique contre 5.5% en reel. A 371k candidats, le sommet est du bruit par
construction. Le top-30 ne peut PAS se lire comme "les meilleures strategies".

**Ce pour quoi le sweep est valide:**
1. **Rejeter des dimensions.** Un negatif uniforme sur 371k tests est solide —
   l'asymetrie joue dans ce sens (facile de rejeter, impossible de selectionner).
2. **Tester une hypothese PRE-SPECIFIEE.** La bande de sentiment a ete formee et
   validee HORS sweep. Verifier `SENT30_70 vs NONE` est UNE comparaison prevue
   d'avance, pas un max sur 371k.
3. Les gates `cross_regime_robust` + `fragile_recent` sont de la robustesse
   TEMPORELLE, ca c'est reel et ca ne depend pas du FDR.

**Comment lire le run en cours (a faire, pas "regarder le top 30"):**
- [ ] Test APPARIE `filter=SENT30_70` vs `filter=NONE` sur les memes
      (strategy, source, smoothing, polling, age_band). Une seule hypothese.
- [ ] Idem `SENT50_60 vs NONE`, `GAP24 vs NONE`, et les 7 arms BSR/KW ressuscites.
- [ ] Verifier n>0 sur chaque nouvel arm (garde-fou contre le bug v14e.72)
- [ ] NE PAS promouvoir une strategie parce qu'elle sort en tete du CSV

- [ ] Envisager: relancer avec --require-fdr en parallele pour voir ce qui survit
      a la correction. Si rien ne survit, c'est l'information la plus utile du run.

## 🎯 [Aug 5] PISTE ML — diagnostic: le modele n'a JAMAIS vu les 2 axes qui marchent

User: "on a desactive le ML car ca ne s'ameliorait pas, mais on a 2 mois de data
en plus". Verification faite sur les 143 features de train_model.py:

| axe | prouve aujourd'hui | dans le modele ? |
|---|---|---|
| **Identite du KOL** (`kol_group`) | slingoorioyaps 90 survivants vs 12 au hasard (7.5x) | **ABSENT** |
| **Cadence des calls** (gap meme KOL) | dose-response -5.43% -> +2.22% sur 600k lignes | **ABSENT** |
| Sentiment 1er message | U inverse, bande 0.5-0.6 = +7.97% | present mais AGREGE (voir plus bas) |
| Features token (score/liq/age/mcap/bsr) | **3 survivants vs 6 au HASARD = zero edge** | ~120 features dessus |
| Forme recente du KOL | **ANTI-predictive** (forme>0 => -4.35% en test) | `kol_avg_prior_return`, `kol_historical_hit_rate` presents |

=> Le ML a ete entraine massivement sur l'axe que j'ai prouve MORT, il lui manque
totalement l'axe que j'ai prouve VIVANT, et il contient des features dont j'ai
montre qu'elles sont anti-predictives. **Ce n'est pas un probleme de volume de
donnees, c'est un probleme de features.** Ca explique proprement l'echec passe.

⚠️ Correction du postulat "2 mois de data en plus": `token_snapshots` ne contient
**0 ligne au-dela de 120 jours** (330 831 snapshots sur 120j, rien avant). La
fenetre d'entrainement est GLISSANTE, pas croissante. Re-entrainer tel quel sur
"plus de data" ne donnera rien de plus.

⚠️ Le `sentiment` du snapshot n'est PAS celui du 1er message: corr 0.820,
ecart absolu moyen 0.071, seulement 63.1% identiques. Sur un U inverse a bande
etroite (0.5-0.6), ce lissage peut suffire a effacer le signal.

### Plan — test decisif PAS CHER d'abord, refonte ensuite
- [ ] **(A) Test cheap**: pouvoir predictif de (kol_group, kol_gap_h, sentiment 1er
      message) SEULS vs les ~120 features token, meme cible, meme split temporel.
      Ne touche pas train_model.py. Decide si la refonte vaut le coup.
- [ ] (B) Si (A) confirme: ajouter kol_group (categoriel, ~72 KOLs actifs),
      kol_gap_h, et le sentiment message-level dans le pipeline
- [ ] (C) Tester le RETRAIT des features de forme KOL (anti-predictives)
- [ ] (D) Gate qualite habituel + null de permutation sur le resultat final

### [Aug 5] RESULTAT ML — NON, ce n'est pas la piste. Et ce n'est pas une question de features.

Test decisif fait (`scripts/_ml_axis_test.py`, HistGradientBoosting, split TEMPOREL
train 940 / test 628, cible = pnl_pct, jugement en termes de trading = EV realisee
sur le top-20% des picks). **12 tirages de permutation** par modele, pas un seul.

| modele | EV top20% | hasard median | fourchette hasard p10-p90 |
|---|---|---|---|
| A. axe KOL seul (gap+sentiment+identite) | -5.14% | -1.03% | -8.23% a +2.01% |
| B. features token (~ le modele actuel) | +3.64% | -0.31% | -6.95% a **+3.63%** |
| C. token + axe KOL | +1.98% | -1.09% | -5.33% a +5.57% |
| D. identite KOL seule | +1.81% | -0.39% | -2.27% a +1.28% |

**Le plancher de bruit fait ~10 POINTS de large.** B est a +3.64% contre un p90 de
+3.63% — a la virgule pres, c'est un pile ou face. Aucun modele ne bat de facon
convaincante son propre hasard, et AUCUN n'approche les **+7.2%** que la simple
bande de sentiment donne deja.

**Pourquoi le filtre bat le ML**: la bande de sentiment est un enonce CONDITIONNEL
valide sur l'agregat ("dans cette bande, EV = +7.2%", n=195, valide par permutation).
Le ML essaie de CLASSER des trades un par un. Sur une cible a queues epaisses avec
940 lignes d'entrainement, le classement par trade est hors de portee — il faudrait
des ordres de grandeur plus de donnees.

=> **La structure du signal est un FILTRE, pas un CLASSEMENT.** Ajouter kol_group et
le gap au pipeline ne changerait rien: le modele A, qui ne contient QUE ces axes,
est le PIRE des quatre. Refonte du pipeline ML: ANNULEE.
Note coherente: D (identite seule) passe de justesse — l'identite KOL porte bien du
signal, mais une whitelist l'exploite mieux qu'un modele.

- [x] (A) test cheap — FAIT, negatif
- [ ] ~~(B) ajouter kol_group/gap/sentiment au pipeline~~ ANNULE par (A)
- [ ] ~~(C) retirer les features de forme KOL~~ sans objet
- [ ] ~~(D) gate qualite sur le modele final~~ sans objet

## 🔬 [Aug 5] REFORMULATION ML — l'asymetrie downside/upside

Critique acceptee: mon 1er test ML etait la formulation la plus FAIBLE possible
(regression sur pnl_pct, cible continue a queues epaisses). Son resultat negatif
ne condamnait que ce cadrage, pas le ML. Reformule en CLASSIFICATION sur des
cibles binaires au niveau TOKEN (`scripts/_ml_reformulation.py`).

### Le resultat: on ne peut PAS predire les gagnants, on PEUT predire les survivants

| cible | prec@top20% | base | AUC | plafond hasard | verdict |
|---|---|---|---|---|---|
| 2x | 24-28% | 29.1% | 0.48-0.54 | 35.0% | bruit |
| 2x propre (sans dump -40%) | 10-14% | 11.3% | ~0.50 | 17.1% | bruit |
| **pas de dump -50%** (token+KOL) | **65.8%** | 36.1% | **0.72** | 44.4% | **SIGNAL** |

21 points au-dessus du plafond de bruit. C'est une vraie asymetrie, et elle est
COHERENTE avec la decouverte de regime du jour (74-82% des runners plongent -50%
avant de monter): le dump est structurel donc predictible, le pump ne l'est pas.

### Traduction trading (test = 262 tokens)

| selection | n | EV reelle | % survivants | best moyen |
|---|---|---|---|---|
| aucun filtre | 262 | -2.42% | 32.4% | 109.3% |
| top 50% proba survie | 131 | +3.32% | 45.0% | 100.1% |
| **top 20% proba survie** | 52 | **+5.60%** | 57.7% | 73.5% |
| **bas 20% (a eviter)** | 52 | **-10.29%** | 13.5% | **154.7%** |

Monotone. Et le detail qui valide tout: **le bas 20% a le PLUS gros potentiel**
(best moyen 154.7%) et c'est la qu'on perd le plus. Les plus gros runners sont
ceux qui flushent le plus fort — on se fait sortir avant la montee.

### MAIS: largement redondant avec la bande de sentiment

Croisement 2x2 (le modele a `sentiment` en feature, il la re-apprend en partie):

| | bande NON | bande OUI |
|---|---|---|
| survie basse | **-14.95%** (n=96) | +10.44% (n=35) |
| survie haute | -2.55% (n=79) | **+12.23%** (n=52) |

La bande DOMINE: elle fait passer de -14.95% a +10.44% a elle seule. Le modele
n'ajoute que **+1.8pp** a l'interieur de la bande (n=52, dans le bruit).
Hors bande il aide beaucoup (-14.95 -> -2.55) mais ca reste negatif: on ne
traderait pas ces tokens de toute facon.

=> **Operationnellement: garder la bande de sentiment, le modele n'ajoute pas
   assez pour justifier un pipeline ML en production.**
=> **Methodologiquement: le cadrage "predire la survie" est le BON, et c'est un
   acquis reutilisable.** AUC 0.72 contre 0.48-0.54 pour l'upside.

- [ ] Piste ouverte que cette asymetrie suggere: si le downside est predictible
      et pas l'upside, la forme optimale n'est pas "mieux choisir les entrees"
      mais "adapter le SL a la proba de dump" — SL large sur les survivants
      predits, pas de trade sur les autres. JAMAIS teste.

### [Aug 5] SL conditionnel — HYPOTHESE MORTE (bien testee)

Idee: si le dump est predictible (AUC 0.72) et pas le pump, alors le SL optimal
devrait dependre du risque de dump — SL large sur les survivants predits pour
encaisser le flush, pas de trade sur les autres. Toutes les strats ont un SL FIXE.

Test sans simulation nouvelle: la grille shadow contient deja TP50 a 6 niveaux de
SL sur les MEMES 2816 tokens. On croise avec p_survie du modele (`scripts/_conditional_sl.py`).

| bucket | SL30 | SL40 | SL50 | SL60 | SL70 | NOSL | meilleur |
|---|---|---|---|---|---|---|---|
| risque ELEVE | -5.61 | -5.73 | -5.07 | -5.53 | **-3.46** | -5.23 | SL 70% |
| moyen | 3.44 | 3.31 | 3.92 | 4.05 | 3.96 | **4.21** | NOSL |
| SURVIE probable | 4.76 | 5.93 | 5.95 | 5.83 | **6.00** | 5.79 | SL 70% |

**Meme SL optimal aux deux extremes.** Ecart entre niveaux de SL dans un bucket
~1pp seulement. Politique conditionnelle vs meilleur SL fixe: **+0.08pp**, et
c'est une BORNE HAUTE (le meilleur SL par bucket est choisi SUR le test).

⚠️ **Le controle est le vrai enseignement**: avec p_survie MELANGEE, le bucket
"survie" sort un gradient parfaitement monotone 3.58 -> 9.97 (SL30 -> NOSL).
PLUS PROPRE que le reel. Sans le controle j'aurais "decouvert" une loi du SL
conditionnel qui n'existe pas. C'est la 3e fois aujourd'hui qu'un controle tue
un resultat qui avait l'air bon.

Pourquoi ca ne marche pas: le label "n'a pas dumpe sous -50%" conditionne DEJA
sur le chemin. A l'interieur du groupe survivant il reste peu d'excursions entre
-30% et -70%, donc rien pour differencier les niveaux de SL.

=> **Le SL n'est pas le levier. Le filtre d'entree l'est.** Ce qu'on a deja.
