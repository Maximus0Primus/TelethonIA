# Lessons Learned

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
