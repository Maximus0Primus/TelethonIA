# Operational Backlog

> 🎯 **Strategy decisions live in `tasks/strategy_candidates.md`.** This file tracks tech infrastructure, deploys, scripts to create, and operational state — pas de strategy ranking ici.

## 🛑 Live status

`live_trading.enabled = False` depuis 2026-05-02 22:12 UTC. Backup config : `data/rt_trade_config_pre_pause_20260501T221242Z.json`. Conditions de resume documentées dans `strategy_candidates.md` (§9 critères + recette stat).

## ✅ Recently shipped (May 7-12, 2026)

### May 12 — fix v14e.58 cooldown poisoning
- [x] **Fix v14e.58** : `paper_trader.py:1262` split `cooldown_combos` en `cooldown_combos_main` (is_shadow=False, gates MAIN re-entry) vs `cooldown_combos` (all, gates SHADOW re-entry). Régression v14e.57 → 14 paper main SOL strats gelées 5j (commit `8b5e4d1`).
- [x] **Backfill 14 strats SOL** first-call dedup-aware (151h frozen) — net **−$771 sur 945 trades** (regime shift réel post-May-7, pas artefact). Backup `_backup_bankroll_v14e58_backfill_20260512`.
- [x] **`candidates.md` May 12 iteration** : 8/10 Tier S/A May 7 collapsed, nouveau top filter-rich `_NZ_S40`/`_MCAP_S40` documenté + analyse dedup-diff parquée.

### May 7-8
- [x] Companion shadow post-promote — `paper_trader.py:1644` (commit `985a11d`) — ⚠️ a introduit le bug v14e.58 fixé May 12
- [x] Sim-align-gate skip clean quand N=0 (commit `d670fba`)
- [x] Cleanup 14 backup tables Supabase + 6 search_path + 10 RPC lockdown (3 migrations)
- [x] 5 v14e.57 picks ajoutés `strategies.py` (commit `4ee1cb0`) — verdict May 12 : 2/5 sim-aligned (+$10-15/d), 3/5 perdent
- [x] Combo proposer enrichi axes filter (commits `80b7f0c` + `f4bf886`)
- [x] Live blacklist Option B per-chain infra (commit `25ff5de`)
- [x] Audit blacklist + DB updates (migrations `v14e57_blacklist_audit_may7` + `v14e57_split_venom_*`)

## 🔧 Operational backlog

### 🆕 May 12 follow-ups

- [ ] **Investiguer RT KOL matching post-restart** — log `0/98 KOL groups matched` sur `get_dialogs()` après restart 14:08 UTC. Probable artifact marked vs unmarked IDs (le runtime `_rt_group_id_to_username` gère les deux, cf. `safe_scraper.py:2362-2365`), mais aucun event KOL matched en >5min post-restart à confirmer avec un message KOL réel. Si confirmé bug Telethon : check StringSession state.
- [ ] **DEMOTE 8 Tier S/A collapsed** May 7 du registry actif — voir `candidates.md` §F. Strats : BE25_LOCK10_TP60_SL30, FAST_TP40_SL30_DS, FAST_TP50_SL30_BOTH, TP50_SL15_NOLAZY, TP50_SL15_BOTH, FAST_TP50_SL30_S40, FAST60_TP50_SL50_S30, BE25_TP80_SL30_LAZYSLOW. Décider si keep en shadow (telemetry) ou retirer registry.
- [ ] **PROMOTE candidates Tier A 12 mai** (filter-rich) — BE25_LOCK10_TP100_SL30_NZ_S40 (#1 7d), TP300_SL50_4H_NZ_S40, TP200_SL40_2H_NZ_S40, FAST_TP40_SL30_S40, BE25_LOCK10_TP100_SL30_S40. N≥85 each, $/d > $40 stable. Procédure : ajouter à `hybrid_strategy.allocations` + seed bankroll. Validation 7-14j post-promote requise.
- [ ] **Re-audit shadow 14j post-v14e.58** (vers Mai 26) — données propres (companion shadow correct, pas de bug cooldown). Ré-évaluer dedup-diff, confirmer les nouveaux Tier A, re-mesurer drift sim↔real.

### Tech features

- [ ] **Fix #2 verb-proximity ticker filter** — patterns `$X (whales|holders|dev) (buying|aping) $Y`. Pas implémenté dans `_FLEX_PATTERNS` (`safe_scraper.py:2015-2049`). Demande N≥30 messages historiques pour calibrer. **À démarrer ou parquer si pas prioritaire.**
- [ ] **Score V3 walk-forward audit** — `rt_score_v3` collecté en shadow depuis v14e.55 (May 2) ✓. Script `scripts/_score_v3_walk_forward.py` **pas encore créé**. Window initial Mai 9-16 (today = mid-window, donc démarrer ASAP ou reprévoir). N attendu : >= 30 trades fermés avec `rt_score_v3` non-null. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter.
- [ ] **LOCK polling alignment** — ⚠️ **Hypothèse de base à reconsidérer** : `BE25_TP80_SL30` (la baseline à "confirmer") fait partie des 8 Tier S/A collapsed (May 7 +$57/d → May 12 −$13/d, cf. candidates.md §B). Le polling `60s + median_5` est déjà actif sur BE25 mais sa perf récente n'est pas conclusive. LOCK strats (BE15_LOCK*, BE25_LOCK*, BE50_LOCK*) sans override → utilisent defaults. **Décision : parquer jusqu'à re-audit post-v14e.58** (donnée propre).

### Perf / monitoring

- [ ] **R2 Profiler `process_and_push`** si lag >30s revient.
- [ ] **T3 ETH round-trip smoke** — `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.

### Scripts à créer (référencés depuis `strategy_candidates.md` Q1)

- [ ] `scripts/_kol_blacklist_audit.py` — paired-test KOL-allowed vs banned trades.
- [ ] `scripts/_kol_per_strat_breakdown.py` — matrice KOL × strat ($/d, WR) — répond à "blacklist optimale per-famille ?"
- [ ] `scripts/_blacklist_counterfactual.py` — recompute Tier S avec/sans blacklist active. Output : delta $/d, sensitivity score.

### Cleanup différé

- [ ] **DROP `_backup_blacklist_audit_20260507`** dans 7j (vers Mai 14) si pas de rollback nécessaire — règle `sql_backup_tables_lifecycle.md`.
- [ ] **DROP `_backup_bankroll_v14e58_backfill_20260512`** dans 7j (vers Mai 19) si bankrolls SOL stable post-fix.

## 📌 Quick reference

### Slippage (single source = `strategies.py`)

- **SOL** : `BUY_SLIPPAGE_BPS = 225`
- **ETH paper** : `BUY 350 / SELL 650`. JSONB miroir : `eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`
- **ETH live tx tolerance** : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`

### Cron schedules (GitHub Actions)

| Workflow | Schedule | Notes |
|---|---|---|
| `mega-sweep-48h` (SOL) | 02:00 UTC tous les 2j | matrix 6 shards (~1.5-2h chacun) |
| `mega-sweep-eth-48h` | 22:00 UTC tous les 2j | single job (3-shard tient) |
| `train-models` | daily | auto-deploy ML model files |
| `sim-align-gate` | 04:00 UTC daily | skip clean si N=0 (post-fix) |
| `outcomes` (Fill Labels) | every ~2-4h | label backfill |
| `nightly-shadow-audit` | daily 07:30 UTC | |
| `nightly-outlier-monitor` | daily 07:20 UTC | |
| `daily-summary` | daily | Telegram report |
| `kol-weekly-audit` | weekly | KOL stats refresh |

### Pre-deploy gate

```bash
bash scripts/pre_deploy_check.sh
# 1/3 py_compile critical modules
# 2/3 import smoke (10 modules)
# 3/3 pytest tests/ -x (~124 tests + ~367 subtests)
```

### Deploy

```bash
git push origin master
ssh vps "cd /opt/TelethonIA && git pull origin master && systemctl restart kol-scraper"
```

### VPS service

- Service : `kol-scraper.service`
- Working dir : `/opt/TelethonIA`
- Logs : `journalctl -u kol-scraper -f` (or via `/logs` skill)
- Wallet : `9t3yNhWUV7f3EfyMAiFHrL6qDU8oT4rA9Agt8tSmBeSM`
