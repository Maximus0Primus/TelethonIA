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

- [ ] **unemployedDegen / UnemployedPlays** — shadow-only, re-évaluer ~2026-06-16 (N≥100 shadow). Critère unban : WR>40% 30d ET med14≥0.
- [x] **BSC/BASE chain pipeline (v14e.63) — INVESTIGUÉ, pas de bug.** `resolve_evm_chain` testé OK (USDT→bsc, USDC→base ; les tokens 0x labellisés ethereum SONT de vrais ETH). 0 row bsc/base = simplement aucun call BSC/Base des KOLs actuels, pas un mis-capture. Allocations BSC/BASE dormantes faute de calls. **DÉCISION en attente : drop les 6 allocations BSC/BASE (0 activité, hors SHADOW_STRATEGIES) ou garder dormantes.**

## 🛠️ Tooling (fait le 22/05 — testés, déployés)
- [x] `scripts/_kol_blacklist_audit.py` — KOL vs ban status, flag mismatches. (Blacklist SOL globalement saine, 1 mismatch mineur.)
- [x] `scripts/_kol_per_strat_breakdown.py` — matrice KOL × strat $/d. (batman_gem dominant SOL, +21.7$/d sur FAST_MCAP live.)
- [x] `scripts/_blacklist_counterfactual.py` — $/d avec/sans blacklist par strat (sensitivity).
- [x] `scripts/_score_v3_walk_forward.py` — AUC v3 vs v1. **Verdict : garder v1** (v3 lift +0.0007 = bruit ; v2=0.520 meilleur mais tous ~random 0.51).

## 📦 Tech/perf parqués
- [x] **RT KOL matching — résolu (v14e.69).** `0/99 matched` était cosmétique (unmarked vs marked id) → corrigé, confirmé `99/99` en prod. ⚠️ Reste : `unmatched chat_id=-1001255304592` en runtime = un channel PAS dans GROUPS_DATA → besoin de l'identifier (KOL à ajouter, ou bruit à ignorer).
- [x] **Counter drift cumulé — ROOT CAUSE fixé (v14e.69).** Race read-modify-write sur `rt_bankroll` (3 boucles concurrentes) → `_BANKROLL_LOCK` (threading.Lock). Reconcile cron reste en filet de sécurité.
- [ ] Verb-proximity ticker filter #2 (`safe_scraper.py` `_FLEX_PATTERNS`) — bloqué : N≥30 messages requis pour calibrer le pattern. Pas implémentable à l'aveugle.
- [ ] R2 profiler `process_and_push` (si lag >30s) / T3 ETH round-trip smoke (si base_fee>5gwei) — conditionnels, non actionnables maintenant.

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
