# Operational Backlog

> 🎯 **Strategy decisions live in `tasks/strategy_candidates.md`** + récap dual-pick `tasks/live_top5_2026-05-17.md` (SOL) et `tasks/live_top_eth_2026-05-17.md` (ETH).

## 🟢 Live status

**`live_trading.enabled = True` depuis 2026-05-17 15:45 UTC** (resume après pause depuis May 2).

Live deck SOL pilot $1/trade :
- `TP50_SL40_S35` (top SOL analyse, +$3.2 médiane, all-window positif)
- `FAST_TP50_SL30_MCAP_S40` (accélérant $99/d 5d shadow)

Config courante :
- `max_position_sol: 0.012` (= ~$1 à SOL $86)
- `max_open_positions: 12`
- `min_position_usd: 1`
- `daily_loss_limit_sol: 0.5`
- `live_trading.kol_chain_blacklist.solana` = 18 KOLs (sync paper)
- `eth_live_enabled: false` (ETH live à activer post J+7 si pilot SOL OK)

Backup config pre-live : `data/live_trading_pre_enable_20260517T154500Z.json`

## ✅ Recently shipped (May 18, 2026)

### v14e.64 — Skills refactor + dedup safety net (2026-05-18)
- [x] Skill `live-perf-snapshot` créé — replace 4 one-off scripts ad-hoc
- [x] GH cron `live-perf-daily.yml` — alert auto if drift/streak anormal
- [x] `tasks/dedup_rules.md` consolidé — 5 lessons en 1 doc canonique
- [x] `scripts/_reconcile_bankrolls.py` créé — du backlog
- [x] Audit `verify_shadow_main_parity.py` — rolling-24h validated
- [x] Cleanup 12 scripts `_*_20260518.py` one-off

## ✅ Recently shipped (May 17, 2026)

### v14e.59 — SOL paper main unblock + mega-sweep keyset + align-gate (commit `20cc32b`)
- [x] **Fix v14e.59 is_shadow filter** : `paper_trader.py:1238, 1474` — shadow opens ne bloquent plus MAIN re-entry. 14 strats SOL gelées 10j refire (confirmé 5 strats live 14:23 UTC).
- [x] **sim.py keyset pagination** : `sb_get` auto-switch keyset on `created_at` quand select inclut `id+created_at` et order=`created_at`. Fixe mega-sweep timeout à offset=238000.
- [x] **sim-align-gate `set +e`** : grep no-match ne tue plus le step bash. PIPESTATUS capture + distinction crash vs drift. Plus d'alerte Telegram vide.

### v14e.60 — ETH SHADOW gap + bankroll last_updated_at (commit `5e54102`)
- [x] **`SHADOW_STRATEGIES.append("ETH_TP80_SL40_T2H")` + `ETH_FAST_TP100_SL20`** (`strategies.py:2748, 2827`) — sync convention SOL. Ces 2 strats paper main allouées depuis v14e.21 (Apr 25) n'avaient pas de shadow → impossibilité de mesurer paired-drift. Fix : shadow va firer à partir du prochain RT event.
- [x] **`safe_scraper.py:1041` per-entry `last_updated_at` write** — bug cosmétique : le code ne touchait jamais ce field, seulement les rebuild scripts. Telegram montrait Apr 26 stale pour ETH. Fix : tous chains uniformément updated à chaque trade close.

### Allocations resets
- [x] **SOL allocations reset 17 → 2** — TP50_SL40_S35 + FAST_TP50_SL30_MCAP_S40, bankrolls $1000 fresh chacune. Backup `data/rt_trade_config_pre_dual_reset_20260517T153350Z.json`.
- [x] **ETH allocations reset 8 → 4** — TP80_SL40_T2H + FAST_TP100_SL20 + FAST60_TP100_SL50_NZ_S40 + TP100_SL50 (🆕 promote), bankrolls $1000 fresh. Backup `data/rt_eth_reset_pre_20260517T180000Z.json`.

### Bankroll reconciliation (all chains)
- [x] **ETH bankroll rebuild from ground truth** — drift Apr 26 stale wiped, 19 strats rebuild. Backup `data/rt_bankroll_eth_pre_rebuild_20260517T162008Z.json`.
- [x] **SOL bankroll reconcile** — 29/40 strats avait drift, $2,444 abs PnL drift wipe, post-reconcile = 0 drift. Baseline v138.3 reset (2026-04-17 14:36 UTC). Backup `data/rt_bankroll_sol_pre_reconcile_20260517T174500Z.json`.

### Live trading enabled
- [x] **`live_trading.enabled = true`** + sync `live.kol_chain_blacklist.solana` = `paper.kol_chain_blacklist.solana` (18 KOLs). `max_position_sol: 0.012` ($1 à SOL $86). 12 max_open_positions. `min_position_usd: 1`.

### Docs
- [x] `tasks/live_top5_2026-05-17.md` — SOL dual pick recap (methodo, stress-tests, deployment phases)
- [x] `tasks/live_top_eth_2026-05-17.md` — ETH dual pick recap

---

## 🔧 Operational backlog

### 🆕 v14e.63 — Fix chain detection bugs Base/BSC (CODED, awaiting deploy 2026-05-17)

**Status** : code done, 137/137 tests pass + pre-deploy gate green.
- [x] Commit 1 — `enrich.py` : `_EVM_CHAIN_CACHE` + `resolve_evm_chain` pour 0x batch path (lines 47-54, 793-810)
- [x] Commit 2 — `push_to_supabase.py:1167` : propager `chain` dans le dict kol_mentions (inferred from resolved_ca shape, no network)
- [x] Commit 3 — `scraper/tests/test_chain_propagation_v14e63.py` : 13 tests (6 kol_mentions + 7 enrich), tous pass
- [ ] Commit + deploy VPS
- [ ] Verify J+1 : `chain='bsc'` ou `chain='base'` apparaît dans `token_snapshots` 24h post-deploy
- [ ] Verify J+1 : `chain='ethereum'` apparaît dans `kol_mentions` (était toujours 'solana' avant)
- [ ] Verify J+1 : Telegram alerts BSC/BASE commencent à apparaître (BSC_/BASE_ strategy allocations enfin firing)
- [ ] Verify J+7 : aucune régression sur SOL/ETH paper main perf



**Problem** : 3 bugs concomitants empêchent toute trace de calls Base/BSC en DB, alors que :
- L'infra est complète : `BSC_*` + `BASE_*` strats déclarées (`strategies.py:2861-2902`), filters `chain` posés, slip/gas constants, 6 allocations existantes en `rt_trade_config.hybrid_strategy.allocations`
- `chain_detect.resolve_evm_chain()` sait disambiguer ETH/BSC/Base via DexScreener
- RT path (`safe_scraper.py:2129-2137`) appelle déjà `resolve_evm_chain` correctement

**Mais** : 0 row `chain='bsc'` ou `chain='base'` dans `token_snapshots`, `tokens`, `paper_trades`, `kol_mentions` sur 30 jours. Le pipeline batch + le writer kol_mentions n'utilisent pas la résolution.

#### Bugs identifiés (audit 2026-05-17)
1. **`push_to_supabase.py:1148` `insert_kol_mentions`** : le dict d'upsert n'inclut pas `chain` → DB default `'solana'` pour tous les rows. 3,484 mentions avec `resolved_ca='0x…'` étiquetées solana sur 30j.
2. **`enrich.py:795`** : `_chain = detect_chain(known_ca) or "solana"` retourne "ethereum" par shape pour tout 0x, **n'appelle jamais `resolve_evm_chain`** → tous les BSC/Base finissent `chain='ethereum'` dans `tokens` et `token_snapshots`.
3. **Cascade** : exemple MOONPEPE `0xb701…1110` (postée par reapergamble avec `@ Four.meme` = BSC) :
   - 8 kol_mentions tagged solana
   - 34 snapshots + 1 token tagged ethereum
   - **81 paper_trades ouverts via l'adapter ethereum** alors que le token vit sur BSC (slip/gas/pool tous wrong)

#### Risk model (zero-regression)
- Bug 1 fix = pure analytique. Ajoute une colonne au write. ON CONFLICT DO NOTHING → existing rows untouched. Aucun downstream ne lit `kol_mentions.chain` pour des décisions (vérifié via grep, c'est query-only).
- Bug 2 fix = comportement nouveau pour les 0x CAs **nouvellement vus**. Ajoute 1 call DexScreener par nouveau 0x (cache 5min via TTL_DEXSCREENER déjà en place).
  - Conséquence intended : un 0x qui résout vers `bsc` ou `base` part sur les strats BSC/BASE existantes au lieu d'être (mal) traité comme ETH. C'est précisément l'effet voulu.
  - Conséquence sur ETH existant : **zéro impact**. Les true-ETH continuent de résoudre vers `ethereum` via DS (les snapshots ETH actuels passent tous par DS `chainId='ethereum'`).
  - Conséquence sur les trades ouverts misroutés (ex MOONPEPE 81 trades) : aucun impact, paper_trader lit `chain` depuis la row paper_trade existante, pas depuis le snapshot live. Les trades en cours closent normalement avec leur chain d'origine.
- **NO backfill historique** : on ne touche pas aux ~12k snapshots ETH déjà écrits (les vrais ETH restent ETH ; les BSC/Base misroutés gardent leur tag `ethereum` historique). Backfill corrompt l'analyse historique paper_trades.
- **NO touch live trading** : `eth_live_enabled=false`, BSC/Base n'ont pas d'adapter live (cf. `safe_scraper.py:1737`). Le fix ouvre seulement le paper trading BSC/Base.

#### Fix plan (3 commits atomiques)

**Commit 1 — `enrich.py` resolve_evm_chain pour 0x batch path**
- `enrich.py:793-796` : remplacer `_chain = detect_chain(known_ca) or "solana"` par :
  ```python
  shape = detect_chain(known_ca)
  if shape == "ethereum":
      from chain_detect import resolve_evm_chain
      _chain = resolve_evm_chain(known_ca) or "ethereum"
  else:
      _chain = shape or "solana"
  ```
- Réutiliser le cache `_rt_evm_chain_cache` existant ? Non — il est session-scoped dans safe_scraper. Mieux : ajouter un cache module-level dans enrich.py (clé = ca.lower(), valeur = chain string, TTL pas critique car immuable).
- Latence ajoutée : ~200ms par nouveau 0x CA. Acceptable (1× par CA puis cache hit).

**Commit 2 — `push_to_supabase.py` propager chain dans kol_mentions**
- `pipeline.py:3567` (call site) : ajouter `"chain": _infer_chain_for_mention(resolved)` où helper retourne :
  - `'solana'` si resolved est base58
  - `'ethereum'` si 0x (sans network call — le batch enrich.py downstream va trancher la vraie chain pour le snapshot)
  - `'solana'` (default) si resolved est None
- **Rationale** : kol_mentions ne devrait pas faire de network call (write hot path). Le shape suffit pour 99% des analytics (split SOL vs EVM). La granularité bsc/base reste utile dans `tokens` + `token_snapshots` (qui passent par enrich avec DS lookup).
- `push_to_supabase.py:1167-1185` : ajouter `"chain": m.get("chain") or "solana"` au dict.

**Commit 3 — Tests**
- `scraper/tests/test_chain_detect.py` : ajouter test que `resolve_evm_chain` est mocké correctement (déjà couvert ?)
- `scraper/tests/test_pipeline_eth.py` : ajouter test que kol_mentions dict inclut chain inferred depuis resolved_ca
- `scraper/tests/test_integration_recent_changes.py` : test BSC CA `0xb701…1110` mocké → enrich retourne chain='bsc', snapshot.chain='bsc'

#### Hors scope (non touché)
- ❌ Pas de backfill historique (corromprait analytics paper_trades historiques)
- ❌ Pas de nouveau live adapter BSC/Base
- ❌ Pas d'allocation changée — les 6 BSC/BASE strats déjà allouées vont juste commencer à firer en paper main
- ❌ Pas de modif `safe_scraper.py:2129-2137` (RT path déjà OK)

#### Verification post-deploy
- Wait 24h après push
- Check `SELECT chain, COUNT(*) FROM token_snapshots WHERE snapshot_at >= NOW() - INTERVAL '24h' GROUP BY chain` → doit montrer `bsc` et/ou `base` >0
- Check `SELECT chain, COUNT(*) FROM kol_mentions WHERE message_date >= NOW() - INTERVAL '24h' GROUP BY chain` → doit montrer >0 `ethereum`
- Check qu'aucun paper_trade SOL n'a régressé (`chain` distribution stable)
- Telegram alerts BSC/BASE doivent commencer à apparaître (`alert_kol_trade` reçoit `chain=ca_chain` déjà)

### 🆕 KOL groups en observation (v14e.62, 2026-05-17)

- [ ] **`unemployedDegen` + `UnemployedPlays`** — ajoutés au scraping mais **blacklist paper + live** SOL (force shadow-only). **MÊME PERSONNE** : `unemployedDegen` = channel principal, `UnemployedPlays` = channel degen low-cap plays. Vérifier J+30 (~2026-06-16) si N≥100 shadow trades par channel → analyser perf séparément (le main vs le degen channel peuvent avoir des profils très différents). Critère unban : WR>40% sur 30d shadow ET med14 ≥ 0. Sinon keep ban (cf. règle §J `strategy_candidates.md`).

### 🆕 Priority — Re-audit J+7 (~2026-05-24)

- [ ] **Re-audit SOL live pilot** — 2 strats à $1/trade depuis 2026-05-17 15:45. Mesurer :
  - Drift live↔paper main companion (target <10pp/trade)
  - PnL réel cumulé sur 7j
  - Si stats clean → décider scale à $50/trade (besoin top-up wallet SOL à ~12 SOL min)
- [ ] **Re-audit 4 ETH paper main** post-reset $1000 fresh — TP80_SL40_T2H, FAST_TP100_SL20, FAST60_TP100_SL50_NZ_S40, TP100_SL50.
  - Vérifier toutes 4 fenêtres (30→14d/14d/7d/5d) restent positives
  - Mesurer paired-drift main↔shadow (post v14e.60 SHADOW fix actif depuis 17:36)
  - Décider promote/demote chacun
- [ ] **Décision live ETH** — si pilot SOL OK + ETH paper main 7j cohérent → activer `eth_live_enabled: true` avec 2 ETH picks. Wallet ETH min $4-5k (gas $200/trade min).
- [ ] **Re-calibration ETH slippage** — la dernière calibration `v14e.49b` (Apr 30) est sur N=10 trades live Apr 27-29 seulement, data old. Empirical mean : BUY 554 bps / SELL 973 bps (vs paper config 350/650 = paper conservateur de +3.5pp total cost). **Trigger** : après activation `eth_live_enabled` + N≥30 nouveaux trades live ETH accumulés. **Action** : run `scripts/_calibrate_eth_slip.py` (à créer), si empirical mean diverge >50bps vs config → update `strategies.py ETH_BUY_SLIPPAGE_BPS / ETH_SELL_SLIPPAGE_BPS` ET miroir JSONB `scoring_config.paper_trade_config.eth_buy_slippage_bps / eth_sell_slippage_bps` (atomique sinon drift). Risque sans recalibration : `−20% live haircut` peut être sur/sous-estimé selon régime gas ETH du moment.

### Tech features (parqués mais toujours pertinents)

- [ ] **Investiguer RT KOL matching post-restart** — log `0/98 KOL groups matched` sur `get_dialogs()` post-restart. Probable artifact marked vs unmarked IDs. Si confirmé bug Telethon : check StringSession state.
- [ ] **Fix #2 verb-proximity ticker filter** — patterns `$X (whales|holders|dev) (buying|aping) $Y` pas implémenté dans `_FLEX_PATTERNS` (`safe_scraper.py:2015-2049`). N≥30 messages historiques requis pour calibrer.
- [ ] **Score V3 walk-forward audit** — `rt_score_v3` collecté en shadow depuis v14e.55 (May 2). Script `scripts/_score_v3_walk_forward.py` à créer. Window Mai 9-23 maintenant possible avec N suffisant. Si AUC V3 ≥ V1 + 0.015 → swap `min_rt_score` filter.
- [ ] **LOCK polling alignment** — parqué jusqu'à re-audit post-v14e.58/59/60 (donnée propre disponible J+7).

### Perf / monitoring

- [ ] **R2 Profiler `process_and_push`** si lag >30s revient.
- [ ] **T3 ETH round-trip smoke** — `_eth_round_trip_smoke.py --execute` mensuel ou si base_fee >5 gwei.

### Scripts à créer

- [x] **`scripts/_reconcile_bankrolls.py`** — shipped v14e.64 (2026-05-18). Dry-run par défaut, `--apply` pour write, `--alert` pour Telegram si drift >$100. Backup auto avant tout write.
- [ ] **`scripts/_audit_shadow_strategies_coverage.py`** — détecte les strats allouées en `hybrid_strategy.allocations` qui ne sont PAS dans `SHADOW_STRATEGIES`. Évite la rechute du bug v14e.60. Sortie : liste des strats à fixer + warning si companion shadow seul (blocked by MAIN).
- [ ] `scripts/_kol_blacklist_audit.py` — paired-test KOL-allowed vs banned trades (script auto pour l'audit hebdo).
- [ ] `scripts/_kol_per_strat_breakdown.py` — matrice KOL × strat ($/d, WR) — répond à "blacklist optimale per-famille ?" et confirme le pattern KOL-conditioning (cf. batman_gem positif sur TP50_SL40_S35).
- [ ] `scripts/_blacklist_counterfactual.py` — recompute Tier S avec/sans blacklist active. Output : delta $/d, sensitivity score (= edge amplification).
- [ ] **`scripts/_calibrate_eth_slip.py`** — miroir SOL `_calibrate_buy_slip.py`. Lit les trades live ETH (`source='rt_live'` + `chain='ethereum'`), extrait l'exec_price vs decision_price pour BUY et SELL, calcule empirical mean/median bps. Comparaison vs `ETH_BUY_SLIPPAGE_BPS / ETH_SELL_SLIPPAGE_BPS` actuels. Flag `--apply` pour write JSONB miroir auto. Trigger initial : N≥30 trades live post-activation ETH live (cf. Priority backlog). Caveat : data live ETH actuel = N=11 du 27-29 avril seulement, trop petit pour run utile aujourd'hui.

### Cleanup différé

- [ ] **DROP `_backup_bankroll_v14e58_backfill_20260512`** (vers Mai 19) si bankrolls SOL stables post-fix.
- [ ] **Considérer drop allocations BSC/BASE (6 strats)** — actuellement 0 activité (jamais firé depuis seed). Garder en allocations consume rien mais pollue les listes. À décider quand BSC/BASE seront prêts à activer ou jamais.
- [ ] **DROP `_backup_blacklist_audit_20260512`** vers Mai 19 si rollback blacklist non nécessaire.

### Sub-finding non urgent

- [ ] **Investiguer counter drift cumulé root cause** — résolu ponctuellement par reconcile mais cause sous-jacente non fixée. Hypothèses (per agent investigation v14e.60) : v14e.32 KeyError window, paper_sim_ev probe `_pending_sl_be` corruption (v14e.48), bankroll write race condition multi-strat. Reconcile auto via cron évite la rechute mais idéal serait root cause fix.

---

## 📌 Quick reference

### Allocations actuelles (2026-05-17 18:00 UTC)

```
SOL (2 paper main + 2 live):
  - TP50_SL40_S35              (live + main, bankroll $1028)
  - FAST_TP50_SL30_MCAP_S40    (live + main, bankroll $1034)

ETH (4 paper main, eth_live disabled):
  - ETH_TP80_SL40_T2H              (bankroll $1000 fresh)
  - ETH_FAST_TP100_SL20            (bankroll $1000 fresh)
  - ETH_FAST60_TP100_SL50_NZ_S40   (bankroll $1000 fresh)
  - ETH_TP100_SL50                 (bankroll $1000 fresh)

BSC (3 paper main, 0 activity historique) — keep as-is
BASE (3 paper main, 0 activity historique) — keep as-is
```

### Blacklist actuelle SOL (18 KOLs — sync paper = live)
`mad_apes_gambles, papicall, markdegens, leoclub69, ChairmanDN1, DegenSeals, aliensalphacalls, LevisAlpha, jadendegens, bagcalls, ryoshikdegen, TheReaperGems, zcallz, venom_gambles, robo_gambles, certifiedprintor, CSCalls, CarnagecallsGambles`

### Blacklist actuelle ETH (6 KOLs)
`jadendegens, aliensalphacalls, batman_gem, dddegens, CryptoChefCooks, degenncabal`

### Slippage (single source = `strategies.py`)
- **SOL** : `BUY_SLIPPAGE_BPS = 225`
- **ETH paper** : `BUY 350 / SELL 650`. JSONB miroir : `eth_buy_slippage_bps=350`, `eth_sell_slippage_bps=650`
- **ETH live tx tolerance** : `eth_buy_slippage_bps=500, eth_sell_slippage_bps=600`

### Cron schedules (GitHub Actions)

| Workflow | Schedule | Notes |
|---|---|---|
| `mega-sweep-48h` (SOL) | 02:00 UTC tous les 2j | matrix 6 shards (~1.5-2h chacun). Post v14e.59 keyset fix = plus de timeout offset>200k |
| `mega-sweep-eth-48h` | 22:00 UTC tous les 2j | single job (3-shard tient) |
| `train-models` | daily | auto-deploy ML model files |
| `sim-align-gate` | 04:00 UTC daily | post-v14e.59 fix : skip clean si N=0, set+e tolérant grep no-match |
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
# Note: config JSONB reload auto à chaque RT event — pas besoin de restart pour
# changes scoring_config / rt_trade_config / kol_blacklist.
```

### VPS service

- Service : `kol-scraper.service`
- Working dir : `/opt/TelethonIA`
- Logs : `journalctl -u kol-scraper -f` (or via `/logs` skill)
- Wallet SOL live : `9t3yNhWUV7f3EfyMAiFHrL6qDU8oT4rA9Agt8tSmBeSM` (balance ~0.26 SOL = $22 au 2026-05-17)
