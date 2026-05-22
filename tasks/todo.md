# Operational Backlog

> Reconcilié le 2026-05-22 contre l'état réel (l'ancienne version datait du 17-18 mai,
> ~moitié déjà fait/périmé). Source de vérité pour les données volatiles (allocations,
> blacklist, slippage) = `scoring_config` en DB, PAS ce fichier.
> 🎯 Décisions stratégie : `tasks/strategy_candidates.md`.

## 🟢 Live status (réel — 2026-05-22)
- **Live deck SOL = `FAST_TP50_SL30_MCAP_S40` SEUL** ($1/trade). FAST60 killé le 22/05 (live −$3.07/N=20 ; paper sur les tokens tradés −13.3%). Garde en shadow.
- `slippage_buy_bps = 1000` (monté de 500 le 22/05), `slippage_sell_bps = 500`, `max_position_sol = 0.012`, `eth_live_enabled = false`.
- Rent ATA : auto-récupéré (v14e.68, retry-read + sweep périodique). Wallet 0 ATA vide.
- Lentille d'analyse par défaut : **`v_strategy_faithful_perf`** (jamais le shadow brut).

## 🔥 Open — réellement actionnable
- [ ] **DÉCISION : drop les 6 allocations BSC/BASE** — 0 trade fermé jamais, absentes de SHADOW_STRATEGIES (confirmé par `scripts/_audit_shadow_strategies_coverage.py`). Elles polluent les listes sans rien produire. Drop = config-only. (Sinon : les laisser dormantes.)
- [ ] **FAST_MCAP : bumper la position $1 → $3-5** — seul strat live, ~breakeven à $1 car le gas fixe (~$0.036) = 3.6%/trade. À $5 le gas tombe à ~0.7% → nettement positif. La mémoire (v14e.49d) suggère aussi moins de slip à position plus grosse. Besoin : valider le wallet SOL a la marge.
- [ ] **Re-audit live continu** — surveiller FAST_MCAP réel (flux on-chain, pas pnl_usd) sur 7j ; surveiller l'effet du slip 500→1000 (fill rate + slip réalisé). Décider scale $50/trade si net positif.

## 🔒 Bloqué (data / temps / funding)
- [ ] **ETH live activation** — besoin wallet ETH $4-5k (gas ~$200/trade) + pilot SOL OK. Les 4 ETH paper main sont positifs en vue fidèle (FAST_TP100_SL20 +11%, FAST60 +14%, TP80_SL40_T2H +8%, TP100_SL50 +6%) mais c'est du brut/sim ; viabilité live ETH ≠ garantie au gas actuel.
- [ ] **Recalibration slip ETH** (`scripts/_calibrate_eth_slip.py` existe déjà) — bloqué : exige N≥30 trades live ETH, or eth_live=false. Trigger : après activation ETH live.
- [ ] **unemployedDegen / UnemployedPlays** — shadow-only, re-évaluer ~2026-06-16 (N≥100 shadow). Critère unban : WR>40% 30d ET med14≥0.
- [ ] **BSC/BASE chain pipeline (v14e.63)** — Bug 1 (kol_mentions chain) déployé (ethereum apparaît). Mais 0 row bsc/base en 48h : soit aucun call BSC/Base, soit Bug 2 (enrich resolve_evm_chain) non déployé. Faible priorité — à investiguer seulement si on veut vraiment trader BSC/Base.

## 🛠️ Tooling optionnel (faible ROI — la vue fidèle + /best-combo couvrent déjà l'essentiel)
- [ ] `scripts/_kol_blacklist_audit.py` — paired-test KOL allowed vs banned (audit hebdo auto).
- [ ] `scripts/_kol_per_strat_breakdown.py` — matrice KOL × strat ($/d, WR).
- [ ] `scripts/_blacklist_counterfactual.py` — Tier S avec/sans blacklist (sensitivity).
- [ ] `scripts/_score_v3_walk_forward.py` — valider rt_score_v3 vs v1 (AUC). Data dispo (May 9-23). Recherche, pas urgent.

## 📦 Tech/perf parqués
- [ ] RT KOL matching post-restart (`0/98 matched` log — probable artifact marked/unmarked IDs).
- [ ] Verb-proximity ticker filter #2 (`safe_scraper.py` `_FLEX_PATTERNS`) — N≥30 messages requis.
- [ ] R2 profiler `process_and_push` si lag >30s. / T3 ETH round-trip smoke si base_fee>5gwei.
- [ ] Root cause du counter drift cumulé (reconcile auto évite la rechute ; root cause non fixée).

## ✅ Fait récemment (archive — détail en git + mémoire)
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
