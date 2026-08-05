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
