# Lessons Learned

## 2026-08-06: "L'outil ne trouve rien" est une hypothèse à tester, pas un résultat
**Mistake:** Pendant trois runs et plusieurs sessions, le mega sweep n'a jamais rien produit
d'exploitable, et à chaque fois je l'ai lu comme un résultat sur les stratégies : classement
sous le plancher de bruit, `cross_regime_robust = 0`, portefeuille réduit à 2 configs. J'ai
même écrit « le classement est du bruit » comme une conclusion. La vraie cause était dans le
log, en clair, à 130 lignes d'écart : `Universe: 2717 unique tokens` puis `240 with ticks`.
La fenêtre de récupération des ticks était ancrée sur l'horloge (`now − 8 jours`) au lieu de
l'entrée de chaque token, donc 91 % de l'univers ressortait vide et le replay le sautait en
silence. Le sweep voyait 9 jours là où il annonçait 4 mois — et sur 9 jours il n'y a qu'un
régime, donc un portefeuille multi-régimes est invisible **par construction**. Ce n'est pas
la recherche qui l'a trouvé : c'est l'user qui a demandé pourquoi le run servait à rien.
**Rule:** Quand un instrument ne produit jamais rien, suspecter l'instrument avant les
données. Concrètement : vérifier que la taille de l'échantillon ANNONCÉE est celle réellement
utilisée à chaque étape, et rapprocher les compteurs successifs du log même quand ils sont
loin les uns des autres (`N chargés` vs `N retenus` vs `N évalués`). Un filtre qui jette en
silence est indétectable dans le résultat final. Tout étage qui peut vider l'échantillon doit
imprimer son taux de survie et alerter sous un seuil.

## 2026-08-06: Un budget de calcul dépensé sur des dimensions d'artefact
**Mistake:** Le sweep balayait 7 lissages × 10 cadences de polling = 70 combinaisons par
config réelle, soit 98.6 % du calcul. Ce ne sont pas des choix de stratégie : c'est
l'hypothèse sur la façon dont on LIT le prix, et une seule correspond à la prod. Ce budget
interdisait d'élargir la fenêtre à 4 mois sous le cap GH de 6 h — et en prime ces 70 quasi-
doublons gonflaient les tests appariés (BSR_MCAP n°1 sur 251k cellules, n°11 sur les 1 196
cellules indépendantes).
**Rule:** Séparer les dimensions qui sont des DÉCISIONS (stratégie, filtre, seuil) de celles
qui sont des hypothèses de MESURE (lissage, cadence, source). Les secondes se testent sur un
petit sous-ensemble pour la robustesse, jamais en produit cartésien complet — et elles ne
doivent jamais compter comme des observations indépendantes dans un test statistique.

## 2026-08-06: Ne pas RECALCULER ce qui vient d'arriver — le relire
**Mistake:** L'alerte KOL CALL affichait `$62934 bankroll | $300 déployé (4 pos) | $62634 dispo`
alors que le deck tourne sur trois bankrolls de $1000 et une mise fixe de $100. Cause :
`safe_scraper` reconstruisait un dict de 3 clés (chain / liquidité / rt_score) pour rejouer
`_passes_strategy_filter` et *deviner* quelles stratégies venaient d'ouvrir. Les trois `PF_*`
de v14e.75 filtrent sur `market_cap` + la bande de sentiment (qui a besoin de `kol_group` et
`token_address`) : aucune de ces clés n'était dans le dict, donc les trois étaient rejetées,
`strategy_positions` revenait vide, et l'alerte tombait sur sa branche de repli qui imprime le
solde GLOBAL de `rt_bankroll` (cumul de toutes les strats et toutes les chaînes depuis avril).
Le même compteur restait à 0 pour `total_pos`, d'où « 4 pos » pour « $300 ».
**Rule:** Quand un effet a DÉJÀ eu lieu, lire son résultat (ici les lignes `paper_trades`
ouvertes) plutôt que rejouer la logique qui l'a produit. Une re-dérivation partielle diverge
silencieusement dès qu'on ajoute un filtre à une stratégie. Et un affichage qui ne sait pas
ne doit **rien** afficher — jamais un nombre global à la place du nombre scopé.

## 2026-08-06: Un workflow GH analyse avec le SHA qui l'a DÉCLENCHÉ, pas le code du moment
**Mistake:** J'ai lu le run mega sweep `31040338036` en attendant d'y trouver le plancher de
bruit (v14e.73) et le classement argent + portefeuille (v14e.74). Ces commits ont été poussés
à 22:30 et 23:31 le 05/08 ; le run avait démarré à 19:38. Son job `merge_and_analyze` a beau
avoir tourné à 00:46 le lendemain, `actions/checkout` reprend le SHA déclencheur : l'analyse a
tourné sur l'ancien script. Les étapes attendues n'apparaissent nulle part dans `analyze.out`.
**Rule:** Un correctif au script d'analyse ne s'applique qu'aux runs **lancés après** le push.
Avant de conclure « la nouvelle métrique ne sort rien », vérifier qu'elle a **imprimé quelque
chose** : une section absente du log n'est pas un résultat vide, c'est du code qui n'a pas
tourné. En cas de doute, ré-analyser les artefacts en local avec le script courant.

## 2026-08-06: `python ... >> "$GITHUB_OUTPUT"` casse le step dès qu'on imprime `::notice::`
**Mistake:** La garde de fraîcheur ETH redirigeait tout son stdout vers `$GITHUB_OUTPUT`.
Elle imprimait aussi `::notice::SKIP — 15 tokens…`, que le runner refuse de parser comme une
paire `clé=valeur` → `Invalid format` → step en erreur → **workflow rouge alors que la garde
avait correctement décidé de sauter**. Le workflow SOL, lui, ouvrait `$GITHUB_OUTPUT` depuis
Python et n'a jamais eu le problème.
**Rule:** Ne jamais rediriger le stdout d'un programme vers `$GITHUB_OUTPUT`. Écrire les
sorties explicitement dans le fichier, laisser stdout aux humains. Test de non-régression :
`scraper/tests/test_workflow_outputs_v14e76.py`.

## 2026-05-22: "Same code path as the sweep" ≠ proven in production — verify the live invocation, on-chain
**Mistake:** Last session I declared live rent recovery good ("the next sell will recover its rent automatically — the code path is identical to the sweep that did 353/353") WITHOUT a single live `close_ata: closed` from the service. On re-audit it had recovered **0/24** sells: the manual sweep runs *after* settlement (balances read 0 → closes), but the live close runs *immediately post-sell* and the read-RPC lagged Jupiter's fill → `"still has tokens, skipping"` (DEBUG, silent) every time. Same function, different timing → opposite outcome. Also: I'd flagged `get_wallet_balance` as the "legacy-only residual" — wrong, it's Jupiter-holdings based; the real legacy-only blind spot was `_count_open_atas.py` (showed 6, hid 11 Token-2022).
**Rule:** Never claim a feature works in production from "the code is the same as X". Verify the ACTUAL production path with its real timing, and confirm on-chain (count empty ATAs across BOTH token programs) — not from logs alone (failures were DEBUG-silent) and not from a delayed/standalone proxy. When a recovery op has a settlement dependency, the immediate path and the delayed path are different tests.

## 2026-05-20: Never assert a cause without verifying — and report drift as MEDIAN not mean
**Mistake (two in one session):** (1) Told the user ETH live is off "because gas = $200/trade" — I lifted that from an old `tasks/live_top_eth_2026-05-17.md` note and stated it as the reason. User corrected me: gas was reconsidered, that's not why. (2) Headlined a "−17pp systemic live↔shadow drift" from a **mean** on N=10. The user pushed back ("hier on disait que ça allait, peut-être juste les dernières 24h, regarde vs paper"). Re-running with median + by-exit-type + paper-main twin: median drift ≈ **−1pp** (normal Jupiter slip, exactly what memory `v14e_49_drift_acted_no_divergence` already said). The −17pp mean was dragged by rug sl_hit fills + one freak case ($MONEY −144pp, likely a re-entry matching artifact).
**Rule:** Before stating WHY something is the way it is, verify it (config history / git / data) or say "I don't have a verified reason." For execution drift on memecoins, ALWAYS lead with the **median** — the mean is structurally dragged by the fat rug tail (sl_hit live fills −18pp worse than paper's clean simulated SL). And check existing MEMORY.md before claiming a regression — I contradicted my own memory.

## 2026-02-14: size_mult must scale aggressively for large-cap memecoins
**Mistake:** `size_mult` capped at 0.7x for ALL tokens >$20M. Pippin at $718M got only -16% penalty (0.84x with freshness boost) — scored 48/100 when it should be ~14. A $700M memecoin needs $700M NEW capital to 2x — near impossible.
**Fix:** Progressive tiers: $50M→0.70x, $200M→0.50x, $500M→0.35x, >$500M→0.25x. No freshness boost for >$50M. Floor lowered from 0.6 to 0.25.
**Rule:** When scoring tokens for 2x potential, the market cap penalty must reflect the actual capital required. $5M and $700M cannot receive the same penalty. Always ask: "how much new money must flow in for this to 2x?"

## 2026-02-14: entry_premium needs market-cap magnitude fallback
**Mistake:** `entry_premium` returned neutral (1.0) for $718M Pippin because: (a) KOLs don't write "at 700M mcap" for established tokens, (b) OHLCV candles only cover 24h so recent calls show ~same price. Both sources fail → neutral → no penalty.
**Fix:** Added third fallback: when mcap > $50M and both primary sources fail, compute implied premium from mcap magnitude (mcap / $1M typical launch). $718M → implied 718x → mult 0.35x.
**Rule:** Any signal with neutral fallback (1.0) must be audited for cases where "no data" actually means "obviously bad". A $700M token with no entry data is NOT neutral — it's already pumped 700x from launch.

## 2026-02-14: Lifecycle "boom" bonus must consider market cap
**Mistake:** Pippin ($718M, +25% 24h, 4 KOLs) classified as "boom" → got 1.1x BONUS. The lifecycle classifier didn't consider mcap, so a $700M established token got the same "boom" bonus as a $500K micro-cap.
**Fix:** Boom phase with mcap > $50M → 0.85x penalty instead of 1.1x bonus.
**Rule:** Lifecycle phases (boom/euphoria/displacement) must factor in token size. "Boom" for a micro-cap means explosive growth potential; for a mega-cap it means the train already left.

## 2026-02-14: XGBoost reg:squaredlogerror crashes on negative targets
**Mistake:** `reg:squaredlogerror` (RMSLE) requires all labels > -1. Our `log_return = np.log1p(max_return - 1)` produces negative values when tokens lose value (max_return < 1). Training crashed on first trial.
**Fix:** Changed to `reg:squarederror` which works with any real-valued target.
**Rule:** When using log-error objectives, verify target range first. For financial return prediction, `reg:squarederror` is safer since returns can be negative.

## 2026-02-14: supabase-py defaults to 1000 row limit
**Mistake:** `load_labeled_data()` fetched all rows with `.select("*").execute()` but supabase-py caps at 1000. With 1,677 1h labels, 677 were silently dropped. The model trained on 60% of available data.
**Fix:** Added pagination with `.range(offset, offset + page_size - 1)` in a while loop.
**Rule:** Always paginate supabase-py queries when data may exceed 1000 rows. Test with `len(result.data) == page_size` to detect truncation.

## 2026-02-14: p@5 is noisy with small test sets
**Mistake:** 12h regression model had Spearman=0.626 (excellent!) but p@5=0.000 because the walk-forward test split (last 30%) happened to have no 2x tokens in the top-5 by random chance. Only 29 winners out of 471 total (6.2%).
**Lesson:** With <500 labels and <10% positive rate, precision@5 on a single temporal split is essentially a coin flip. Need 1000+ labels for stable p@5 estimates. Consider using cross-validation with multiple temporal splits, or lowering the quality gate when Spearman is strong.

## 2026-02-14: Optimizer must explore from multiple starting points
**Mistake:** Rewrote autoOptimize to start cumulative sweep from user's current config only. If the config was already decent, the greedy sweep found no improvements (each phase compared against accumulated best using strict `>`). The old code started `bestHR = 0` per component so it always found something.
**Fix:** Dual-track optimization — run `cumulativeSweep()` from BOTH `baseConfig` AND `DEFAULT_CONFIG`, take the better result, then run random exploration on top. Feature impacts computed by reverting each feature from bestConfig to baseConfig (shows "what would we lose by going back to your settings").
**Rule:** Greedy optimizers stuck in local optima is a known problem. Always try multiple starting points. For feature impact, "revert and measure" is more meaningful than "marginal improvement during sweep".

## 2026-02-14: Optimizer must optimize for the metric the user sees
**Mistake:** Optimizer maximized `top10_hit_rate` but the user looks at `top5_hit_rate`. The optimizer found configs where 2x tokens land in positions 6-10 (great top10) but not 1-5 (terrible top5). User sees top5 drop from 40% to 20% after applying the "best" config.
**Fix:** Changed optimization target to combined score: `top5 * 0.5 + top10 * 0.3 + top20 * 0.2`. Top5 dominates, top10/top20 add granularity for tie-breaking. Button shows actual top5 hit rate (not internal score).
**Rule:** Always optimize for the metric that's most visible to the user. If displaying top5/top10/top20, the optimization target must prioritize top5. Test by verifying that the displayed metric doesn't DROP after applying the optimized config.

## 2026-02-13: Check ALL imports before committing
**Mistake:** Committed `safe_scraper.py` with a top-level `from auto_backtest import run_auto_backtest` but `auto_backtest.py` was untracked. GitHub Actions crashed immediately with `ModuleNotFoundError`.
**Fix:** Moved to lazy import inside the try/except block where it's actually used.
**Rule:** Before committing, verify that every import in modified files either (a) exists in the repo, (b) is in requirements.txt, or (c) is guarded by try/except. Run `python -c "import <module>"` as a smoke test when uncertain.

## 2026-02-14: ML training data must be deduplicated to one snapshot per token
→ See `tasks/dedup_rules.md` (Rule 1)

## 2026-02-14: SOL price leaks persist across ALL horizons
**Mistake:** outcome_tracker.py had a known SOL price leak bug (OHLCV APIs returning SOL price ~$78-87 instead of token price). A sanity check was added, but only AFTER $YEE, $ZEREBRO, $LUCE were already labeled. Found 17 corrupted rows: 1 in 12h, 11 in 6h, 5 in 1h. Max prices of $79-87 for micro-cap tokens are SOL's price.
**Fix:** Cleaned all corrupted labels to NULL for re-labeling. Scanned ALL `max_price_*` columns across all horizons (not just the one where the bug was first found).
**Rule:** When cleaning data corruption, always scan ALL related columns/horizons. A bug that corrupted 12h labels almost certainly also corrupted 6h and 1h labels if they existed at the time.

## 2026-02-14: Feature correlations computed on duplicated data are unreliable
→ See `tasks/dedup_rules.md` (Rule 2)

## 2026-02-14: Tuning Lab backtester must use per-cycle evaluation, not global ranking
**Mistake:** Tuning Lab showed 80% top5 hit rate at 12h — wildly inflated. Root causes: (1) No deduplication — same winning token ($WORDSLOP with 6 snapshots) filled multiple top-5 slots across cycles. (2) `consensus_val IS NOT NULL` filter in backtest API dropped 9 out of 13 unique winners (from before component values were saved). (3) Global ranking treated all snapshots as one pool, ignoring that the scoring system produces rankings per 15-min cycle.
**Fix:** Three changes: (a) Backtest API removed `consensus_val` filter — rescorer handles nulls with defaults. (b) `backtester.ts` rewrote to per-cycle evaluation: group snapshots into 15-min cycles, rescore each cycle independently, compute hit rates per cycle, average across valid cycles (5+ tokens). (c) Global stats (base_rate, avg_score, separation) still use dedup for honest denominators.
**Rule:** A real-time ranking system must be backtested PER DECISION POINT (cycle), not as one big pool. Per-pool evaluation lets duplicate winners inflate top-K and hides that the system makes independent decisions every 15 minutes. Always ask: "at each decision point, did we rank the winner in the top K?"

## 2026-02-15: NEVER dedup by symbol — always use token_address
→ See `tasks/dedup_rules.md` (Rule 3)

## 2026-05-18: Shadow data MUST be deduped before combo/ranking
→ See `tasks/dedup_rules.md` (Rule 4)

## 2026-05-18: Dedup MUST be rolling 24h on timestamps, not calendar day
→ See `tasks/dedup_rules.md` (Rule 5)

## 2026-02-14: unique_kols must be materialized in snapshots, not left NULL
**Mistake:** 41% of token_snapshots had `unique_kols = NULL` because the column was added after many snapshots were already created. `top_kols` JSON was always populated but `unique_kols` numeric was not extracted from it.
**Fix:** Backfilled 4,201 rows from `top_kols` JSON: `json_array_length(top_kols::json)`. Ensured push_to_supabase always writes unique_kols.
**Rule:** When adding a new computed column to snapshots, always backfill from existing data. NULL features are invisible to ML and break any feature that depends on them.

## 2026-02-22: CA cross-contamination in multi-token messages (v56 fix)
**Mistake:** v50 fallback (`len(unique_msg_cas) == 1 → assign CA to all tokens`) caused 9 CA collisions. Example: message "$KELLYCLAUDE $BLOX DSy..." — CA belongs to KELLYCLAUDE but v50 assigned it to BLOX too. Then enrich_token fetched KELLYCLAUDE's data for BLOX. Dashboard showed wrong chart/CA for affected tokens.
**Root cause 1:** `extract_tokens()` skipped CA when resolved symbol was already in `seen` set → `ca_by_symbol` was empty → fallback triggered.
**Root cause 2:** Fallback didn't check if the CA was already "owned" by another token via direct extraction.
**Fix (2 parts):** (a) `extract_tokens()` backfills CA onto existing ticker-only tuple when CA resolves to an already-seen symbol. (b) Aggregation loop filters `unowned_cas` — CAs claimed by `ca_by_symbol` can't fall back to other tokens.
**Rule:** When a single CA appears in a multi-token message, it belongs to exactly ONE token. Never assign it to all tokens. Check if the CA resolves to an already-extracted symbol first.

## 2026-02-22: Helius 429 retries caused 25min+ stalls — no global circuit breaker
**Mistake:** When Helius rate-limited (429), each of the 200 tokens independently retried 3× with exponential backoff (~10s wasted per token). With 3 parallel workers and no shared state, the scraper spent 25-35min on retries alone, exceeding the 35min GH Actions timeout → cancelled every cycle (9 in a row).
**Root cause:** No global awareness of rate limit status. Each token's retry loop was independent. `_fetch_token_accounts()` had per-token retries but no way to signal "stop trying, we're rate-limited" to the other 195 tokens.
**Fix:** Thread-safe 429 circuit breaker: global counter increments on each token's retry exhaustion, resets on any success. After 5 consecutive failures → trip → all subsequent API calls return None immediately. ~20s wasted instead of 25min.
**Rule:** Any API enrichment loop over N items MUST have a global circuit breaker. Per-item retries are necessary but insufficient — when the API is down, N × retry_time = catastrophic delay. Always add: "after K consecutive failures, abort remaining items."

## 2026-02-22: Helius budget estimate was wrong by 262× — exhausted free tier
**Mistake:** Comment said "~22K CU/month (2.2%)" but actual usage was 5.76M CU/month (576%). The estimate was written when HELIUS_TOP_N=50 and cache TTL=2h, but v36 raised TOP_N to 200 and v41 lowered TTL to 30min. Nobody recalculated: 200 tokens × 20 CU × 48 refreshes/day × 30 days = 5.76M vs 1M free tier.
**Fix:** Cache TTL 30min → 4h. New budget: ~720K CU/month (72% of free tier).
**Rule:** When changing API call frequency (cache TTL, batch size, polling interval), ALWAYS recalculate the monthly budget. Write the formula in the comment, not just the result: `TOP_N × CU_per_token × (24h / TTL_hours) × 30 = X CU/month`. Budget estimates without formulas become stale lies.

## 2026-02-22: PostgREST silently fails on wrong column names → HTTP 400
**Mistake:** `_load_kol_win_rates()` filtered on `called_at` but the column is `call_timestamp`. PostgREST returned HTTP 400 (not silently ignored). Function returned empty dict → `best_kol_win_rate` was NULL for ALL tokens → `kol_wr_mult` always 1.0. The feature appeared "deployed" (column populated with 1.0) but was completely non-functional.
**Fix:** Changed `"called_at"` to `"call_timestamp"`.
**Rule:** When writing PostgREST queries, ALWAYS verify column names against the actual schema (`information_schema.columns`). PostgREST returns 400 on unknown columns, but if the error is caught by a generic `except`, it silently degrades. Add the actual column list as a comment above the query. For new features, check logs for HTTP 400 on the first run — a "working" default value (1.0) can mask a completely broken data source.

## 2026-02-22: DexPaprika OHLCV returns SOL price for Pump.fun pools with inverted base token
**Mistake:** DexPaprika OHLCV returns the **base** token's price. Most pools (PumpSwap, Raydium) have the memecoin as base → correct. But some Pump.fun pools have SOL as base and memecoin as quote → returns ~$85 SOL price instead of $0.00002 token price. 27 tokens per outcome_tracker run rejected by sanity check → never labeled (data loss). `enrich_dexpaprika_ohlcv.py` had NO detection at all → wrong ATH/ATL/PA scoring for live tokens.
**Root cause:** Verified via API: `$CONNECTED` pool `8i3o...` has `tokens[0].address = So111...` (SOL) as base. `$BLOODNUT` pool `GkG5...` (PumpSwap) has memecoin as base → correct.
**Fix:** (1) `_is_sol_base_pool()` queries pool metadata (`/networks/solana/pools/{pool}`) and caches `tokens[0].address == SOL_MINT`. (2) `_fetch_ohlcv_candles()` and `_fetch_ohlcv_candles_kco()` skip DexPaprika for inverted pools → fall through to Birdeye (uses mint address, always correct). (3) `enrich_dexpaprika_ohlcv.py` adds `median_close > 50` heuristic as safety net.
**Rule:** When using pool-based OHLCV APIs, ALWAYS check which token is the base. Pool token ordering varies by DEX protocol (PumpSwap: memecoin=base, some Pump.fun: SOL=base). The safest approach is to query pool metadata once and cache the result. Mint-address-based APIs (Birdeye) are immune to this issue.

## 2026-03-02: Never reference cross-module globals without importing them
**Mistake:** `safe_scraper.py:_rt_ml_position_mult()` referenced `SCORING_PARAMS` (a global dict defined only in `pipeline.py`), but never imported it. Every RT token detection (96 since March 1) crashed with `NameError: name 'SCORING_PARAMS' is not defined`, silently killing all RT trades for 2+ days. Batch trades continued working because they use a different code path.
**Root cause:** Copy-paste from pipeline.py logic into safe_scraper.py without adapting the data source. The RT handler already had its own config dict (`_rt_config` via `_rt_load_config()`) but the new ML function used the pipeline global instead.
**Fix:** (1) Extended `_rt_load_config()` to also fetch `rt_ml_weights` from `scoring_config` (same DB query, just added column). (2) Changed `SCORING_PARAMS.get("rt_ml_weights")` to `config.get("rt_ml_weights")` to use the already-passed config dict.
**Rule:** When moving logic between modules, ALWAYS check that every referenced name is either imported or locally defined. A `NameError` in an async event handler (Telethon callback) doesn't crash the service -- it silently kills that handler invocation while everything else keeps running, making it hard to detect. Add a startup smoke test for critical paths.

## 2026-03-03: Asymmetric R:R kills PnL even with high WR
**Mistake:** FRESH_MICRO had TP +30% / SL -70% (R:R = 0.43). Despite 50% WR on RT, it lost -$28.88 over 7 days. Each win = +30% but each loss = -70%, so you need 70% WR just to break even.
**Fix:** Changed SL from -70% to -30% (sl_mult 0.30→0.70). Now R:R = 1.0 (symmetric). At 50% WR, should break even; any WR edge becomes profit.
**Rule:** Always check R:R ratio when designing strategies. WR alone means nothing — a 90% WR strategy loses money if each loss is 20x each win. Formula: breakeven WR = SL% / (TP% + SL%). For FRESH_MICRO old: 70/(30+70) = 70%. New: 30/(30+30) = 50%.

## 2026-03-03: Anti-predictive ML models have inverted value
**Observation:** Bot ML gate (7d, N=189): SKIP trades = 60% WR, FULL trades = 35% WR. Model is consistently anti-predictive — its "bad" signals are actually good.
**Approach:** Instead of disabling, invert: SKIP→1.5x boost, HALF→1.3x boost, FULL→0.7x reduce. Config-driven via `ml_gate_mode: "inverted"` to easily switch back.
**Rule:** An anti-predictive model has signal — just inverted. Before discarding a model, check if flipping its output adds value. This only works if the anti-correlation is stable over time (not just noise).

## 2026-04-20: TOUJOURS paired-test pour comparer variants (pas agrégat $)
**Mistake:** Comparé NOLAZY/LAZYSLOW v144 vs base LAZY en aggregate $ 7d. Conclusion : "NOLAZY perd $8/strat, retirer". REVERSED après paired-test.
**Réalité (paired sur mêmes tokens) :**
- FAST_TP40_SL30_NOLAZY : pair_N=16, med Δpp **+1.76**, sum Δ$ +$27 (10 wins / 6 losses) → NOLAZY > base
- FAST_TP80_SL25_NOLAZY : pair_N=16, med Δpp +1.55, sum +$30 (9/7) → NOLAZY > base
- TP50_SL15_NOLAZY : pair_N=15, med Δpp **+3.80**, sum +$40 (11/4) → NOLAZY domine
- FAST_TP80_SL25_LAZYSLOW : pair_N=12, med Δpp +2.62, sum +$27 (8/4) → LAZYSLOW > base
**Cause de l'erreur initiale :** v144 shadows démarrés Apr 19 → ont raté les bonnes journées du début de fenêtre 7d que la base a captées. Agrégat $ est un artefact de fenêtre, pas de qualité strat.
**Rule :** Quand on compare strat A vs strat B et qu'elles tradent des sets de tokens partiellement disjoints :
1. Calculer le set INTERSECTION (tokens tradés par les DEUX)
2. Calculer Δpp = pnl_pct_A − pnl_pct_B PAR token, puis median + win/loss/tie counts
3. Si pair_N < 10, déclarer "verdict prématuré" — ne JAMAIS conclure sur agrégat $ seul quand les set sizes diffèrent par >2×
4. Script de référence : `scripts/compare_lazy_vs_nolazy.py` (existe déjà). À étendre pour `_LAZYSLOW/_LAZYFAST/_LAZYMED/_LAZYXSLOW`.
5. Cette règle s'applique à TOUTE comparaison variant vs base (HYST, DS, MED3, NOLAZY, LAZY*, BOTH, JUPITER, S30/S40, COMBO).

## 2026-04-20: HYST nu = artefact sim, HYST + filtre qualité = vrai signal
**Observation:** Sweep sim v140 a promu 4 variants `_HYST` en mains. Paper 7d (N=38-69 chacun) confirme l'artefact :
- FAST_TP80_SL25 base +$427 vs `_HYST` −$62 → **−$489 coût HYST**
- FAST_TP100_SL20 base +$137 vs `_HYST` −$54 → **−$191**
- BE25_TP80_SL30 base +$191 vs `_HYST` +$6 → **−$185**
- BE25_TP80_SL30 base +$191 vs `_DS` −$0 → **−$191**

Mais **HYST + filtre qualité** marche très bien : `BE25_TP80_SL30_S30_HYST` (HYST + SCORE≥30) = +$312/7d, WR 52% (#2 earner). `_NZS30_HYST` (NZ + S30 + HYST) = +$77.

**Rule :**
1. Ne JAMAIS promouvoir un variant `_HYST` ou `_DS` sans pair-test paper N≥30 vs la base.
2. Le sim sur-estime systématiquement les variants smoothing (HYST/MED/DS) car le whipsaw paper-réel n'apparaît pas dans les ticks lissés du sim.
3. Préférer combo `_HYST` + filtre entrée (`_S30`/`_NZ`) — le filtre élimine les tokens où le hysteresis whipsaw.
4. Famille connue à sur-estimation sim massive (`tasks/todo.md` Apr 19) : TD2 45×, BOND_FAST 57×, HYST −2 à −6pp, DTRAIL/TRAIL.

## 2026-04-20: Spearman per-trade ≠ Spearman per-strategy — ne pas confondre
**Mistake:** todo.md a noté "Spearman sim↔paper ρ=+0.905 (N=139)" issu de `paper_sim_pnl_pct` (join per-trade), puis `ranking_compare.py` (cross-strategy) sort ρ=+0.599 — interprété à tort comme une "dérive".
**Réalité:** Ce sont DEUX métriques différentes :
- **Per-trade ρ** (=0.905) : sim_pnl vs paper_pnl pour CHAQUE trade joint — mesure si le sim est ordonnateur de trades individuels
- **Per-strategy ρ** (~0.6-0.7 stable) : rank des stratégies par avg PnL agrégé — mesure si le sim choisit les bonnes strats
**Rule :** Toujours préciser le niveau d'agrégation quand on cite un Spearman. Per-trade et per-strategy ne sont pas comparables. Le per-strategy ranking sim↔paper est ~+0.7 historiquement et c'est OK pour notre cas — pas de "drift". Bonus : v144 shadows polluent le ranking per-strategy de −0.10 ρ, donc exclure suffixes `_NOLAZY/_LAZY*/_BOTH/_JUPITER/_S30/_S40/_MED3/_DS/_HYST/_COMBO/_MCAP` dans `ranking_compare.py`.

## 2026-04-20: Shadow DTRAIL/TRAIL/DIP n'est PAS reproductible en live
**Mistake potentiel :** Voir DTRAIL10_ACT15_SL70 shadow paper +$52-370/7d (selon fenêtre/filtres) avec WR 53% et envisager promotion live.
**Réalité (audit Apr 20, 20 trades live + 108 shadow paire pair):**
- Live actual sell slip median = **9429 bps (94%)** vs paper modélise 200 bps → paper 47× trop optimiste sur coût sortie
- 13/20 trades live = status `reconciled` (pas `trail_stop`) → le `position_reconciler` ferme prématurément avant que le trail s'exécute. Exemples : `GLzhjuzxKDrw7r` live +68% vs paper +213%, `26jyBRf3nCxAs1` live 0% vs paper +107%
**Rule :** Toute stratégie famille DTRAIL/TRAIL/DIP/SPLIT (= multiples sells au cours d'un trade, ou logique trail/peak-detect) sur Solana memecoins low-liq pump.fun :
1. Ne JAMAIS promouvoir en live même si paper WR=65% et $/jour positif
2. Le `position_reconciler` interrompt typiquement 50-65% des trades avant que la logique trail/peak ne s'exécute
3. Slippage réelle sur sells fréquents = 50-100× le modèle paper
4. Pour Solana memecoins, préférer stratégies à exit unique (TP/SL/timeout/BE) sans logique trail dynamique
5. Ces shadows polluent l'analyse — candidat retrait de SHADOW_STRATEGIES dans `strategies.py`

## 2026-04-20: Slip recalibration per-exit-type, pas global
**Observation:** Apr 20, 143 pairs L/P matched 14d. Median delta L−P par cellule :
- pump × sl_hit (N=26) : −22 bps (OK)
- pump × timeout (N=29) : **+87 bps** (paper trop pessimiste)
- pump × tp_hit (N=5) : +154 bps mais N trop petit
- pump × trail_stop (N=79) : +83 bps mais 100% DTRAIL10 (strat retirée)

**Mistake potentiel :** appliquer un offset global −87 bps casserait `sl_hit` qui est calibré.
**Rule :** Ne jamais appliquer un offset slip global quand les cellules pump × exit_type divergent. Schéma futur : `_dynamic_sell_slip_factor` doit accepter `{exit_type: bps_offset}` dict, pas un scalaire. Calibration par cellule à N≥15. Script `scripts/slip_per_exit_type.py` rejoue le diagnostic.

## 2026-08-05 — Leçons de la session "chercher ce qui marche"

### L1. Classer sur la MOYENNE, jamais la médiane
Corrigé par l'user. `SCALP_TP20_NOSL` sur 120j : médiane **+18.7%**, moyenne arith
**−2.40%**, moyenne géom **−19.65%**. Mêmes lignes, trois conclusions opposées.
À taille fixe `PnL = N × taille × moyenne_arith` — la médiane n'entre pas dans la formule.
La médiane ne sert qu'au diagnostic de forme (moy >> méd = dépend de la queue droite).
Toujours reporter les trois.

### L2. Jamais de conclusion sans null de permutation
Un scan de 4778 candidats stratégie×filtre a sorti 3 "gagnants" validés train/test.
Le même scan sur pnl mélangé en sort **6**. Moins de gagnants que le bruit.
=> Tout scan multi-candidats DOIT être accompagné du même scan sur labels permutés.
Sans ça on livre du dredging. C'est ce qui a tué l'axe features et légitimé l'axe KOL
(161 réels vs 91-99 au hasard).

### L3. `price_ticks` est un LOG multi-sources, pas une série de prix
jupiter/fast/full s'entrelacent toutes les 11-20s, désaccord p1 −85.8% / p99 +640%.
Rejouer sans filtrer `source` fabrique un faux edge (+12.6%/trade, 5/5 semaines).
Test qui l'a attrapé : refaire le fill 1 tick plus tard (`fill_lag=1`) => 0/5 semaines
sur 21 configs, puis lag2 repositif. **Oscillation lag0/lag1/lag2 = signature d'artefact**
(une vraie dégradation serait monotone). Prod et sim.py filtraient déjà correctement.

### L4. Valider les composants séparément AVANT de les combiner
Le combo final (slingoorioyaps × gap≥24h) n'a pas été trouvé par recherche de combo.
Le KOL a été validé par permutation, le gap par dose-response sur 600k lignes,
indépendamment. Puis empilés. C'est la défense contre le surapprentissage :
on ne cherche pas la combinaison, on empile des effets déjà prouvés.

### L5. Vérifier le taux de remplissage AVANT de bâtir sur une feature
`whale_new_entries` = "seul signal robuste confirmé" du research_log, mais rempli à
**0%** sur les trades RT. `ml_pred` 0%. `unique_kols` toujours = 1. Trois pistes du
backlog mortes en une requête de fill-rate.

### L6. Un filtre doit être satisfiable AU MOMENT où il est évalué
Le portefeuille E30 (v14e.75) filtre sur la bande de sentiment. Le sentiment était lu
dans `kol_mentions`, table écrite par le **batch** : sur 1724 mentions/7j, **zéro** ligne
écrite en moins de 60 s après le message (médiane 29.3 min). Le RT décide en ~7 s.
Le SELECT ne trouvait donc jamais rien, `_msg_sentiment` renvoyait `None`, et le contrat
"None = gate non satisfait" rejetait **les 3 stratégies du deck à chaque call**.
Résultat : 58 détections, 0 ouverture main, 0 alerte pendant 21 h — **sans une seule
ligne d'erreur**, parce que l'alerte est gardée par `if opened > 0`.

⇒ Quand une stratégie ajoute un filtre, se demander **qui écrit ce champ, et quand**,
par rapport à l'instant où le filtre est évalué. Un champ rempli en aval du point de
décision rend le gate insatisfiable, pas juste imprécis.
⇒ Le test qui l'attrape en une requête : `percentile_cont` du lag `created_at − message_date`
sur la table source. Même famille que L5 (fill-rate), mais sur l'axe **temps** et non
sur l'axe **remplissage** : ici le champ est rempli à 100 %, juste trop tard.

### L7. Une panne silencieuse se cherche par le compteur, pas par les logs d'erreur
Aucune exception, aucun warning, bot Telegram joignable, identifiants présents, service
`active` avec 0 restart : tout était vert. Le seul indice était un **compteur à zéro**
dans une ligne INFO de routine — `opened 0 rows + 314 shadow`.
⇒ Devant "ça ne marche plus" sans erreur, ne pas grep `ERROR` : **remonter le garde**
qui décide de l'effet manquant (`if opened > 0`) et vérifier sa condition.
⇒ Corollaire : `_send()` échoue en `logger.debug` quand token/chat_id manquent — une
alerte peut mourir sans laisser de trace au niveau INFO. Vérifier l'env, pas les logs.

### L8. Corriger le symptôme rapporté ne prouve pas qu'on a corrigé la cause
v14e.76 (la veille) avait **déjà identifié** que les `PF_*` étaient rejetées faute de
champs — mais uniquement dans le chemin d'**affichage** de l'alerte, qui était le symptôme
rapporté. La même cause, dans le chemin d'**ouverture**, faisait un dégât bien plus grave
et n'a pas été cherchée.
⇒ Quand on trouve "ce filtre rejette tout", chercher **tous** les appelants de ce filtre
avant de refermer, pas seulement celui qui a produit le ticket.
