"""
v67: In-memory metrics collector for monitoring scraper health.

Zero Supabase egress — all data stays in-memory. Alerts via Telegram Bot API.
Thread-safe singleton. If this module fails to import, scraper runs normally.
"""

import time
import threading
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Estimated row sizes for egress tracking (bytes)
ROW_SIZE_ESTIMATES = {
    "tokens": 800,
    "token_snapshots": 2000,
    "token_snapshots_label": 400,
    "kol_mentions": 300,
    "paper_trades": 500,
    "scoring_config": 10_000,
    "scrape_metadata": 200,
}


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class MetricsCollector:
    """Thread-safe in-memory metrics singleton."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        # API call tracking
        self._api_calls: deque = deque(maxlen=10_000)

        # Cycle tracking
        self._cycle_current_num: int | None = None
        self._cycle_current_start: float | None = None
        self._cycle_history: deque = deque(maxlen=100)
        self._cycle_errors: list = []

        # RT event tracking
        self._rt_events: deque = deque(maxlen=5_000)

        # Supabase egress tracking (per-day, per-module)
        self._egress_by_module: dict[str, int] = defaultdict(int)
        self._egress_day: str = _today_str()

        # Paper trade counters (per-day)
        self._paper_opens_today: int = 0
        self._paper_closes_today: int = 0
        self._paper_pnl_today: float = 0.0
        self._paper_day: str = _today_str()

    def _reset_day_if_needed(self):
        """Reset daily counters on day change. Caller must hold _lock."""
        today = _today_str()
        if today != self._egress_day:
            self._egress_by_module.clear()
            self._egress_day = today
        if today != self._paper_day:
            self._paper_opens_today = 0
            self._paper_closes_today = 0
            self._paper_pnl_today = 0.0
            self._paper_day = today

    # --- API calls ---

    def record_api_call(self, api: str, endpoint: str, status: int,
                        latency_ms: float, resp_bytes: int = 0, error: str | None = None):
        with self._lock:
            self._api_calls.append({
                "ts": time.time(),
                "api": api,
                "endpoint": endpoint,
                "status": status,
                "latency_ms": round(latency_ms, 1),
                "bytes": resp_bytes,
                "error": error,
            })

    # --- Cycle tracking ---

    def cycle_started(self, num: int):
        with self._lock:
            self._cycle_current_num = num
            self._cycle_current_start = time.time()

    def cycle_error(self, msg: str):
        with self._lock:
            self._cycle_errors.append({
                "ts": time.time(),
                "num": self._cycle_current_num,
                "msg": msg,
            })
            # Keep only last 50 errors
            if len(self._cycle_errors) > 50:
                self._cycle_errors = self._cycle_errors[-50:]

    def cycle_completed(self, scored: int, pushed: int, windows: list[str]):
        with self._lock:
            duration = 0.0
            if self._cycle_current_start:
                duration = time.time() - self._cycle_current_start
            self._cycle_history.append({
                "ts": time.time(),
                "num": self._cycle_current_num,
                "scored": scored,
                "pushed": pushed,
                "windows": windows,
                "duration_s": round(duration, 1),
            })
            self._cycle_current_start = None

    # --- RT events ---

    def record_rt_event(self, symbol: str, kol: str, ca: str,
                        score: float, size: float, trades: int):
        with self._lock:
            self._rt_events.append({
                "ts": time.time(),
                "symbol": symbol,
                "kol": kol,
                "ca": ca[:12],
                "score": round(score, 1),
                "size": round(size, 2),
                "trades": trades,
            })

    # --- Egress tracking ---

    def record_supabase_egress(self, module: str, bytes_count: int):
        with self._lock:
            self._reset_day_if_needed()
            self._egress_by_module[module] += bytes_count

    # --- Paper trades ---

    def record_paper_trade_open(self, count: int):
        with self._lock:
            self._reset_day_if_needed()
            self._paper_opens_today += count

    def record_paper_trade_close(self, count: int, pnl: float):
        with self._lock:
            self._reset_day_if_needed()
            self._paper_closes_today += count
            self._paper_pnl_today += pnl

    # --- Stats getters ---

    def get_api_stats(self, hours: float = 1.0) -> dict:
        """Stats per API for the last N hours."""
        cutoff = time.time() - hours * 3600
        with self._lock:
            recent = [c for c in self._api_calls if c["ts"] >= cutoff]

        by_api: dict[str, dict] = {}
        for c in recent:
            api = c["api"]
            if api not in by_api:
                by_api[api] = {"calls": 0, "errors": 0, "latency_sum": 0.0, "bytes": 0}
            by_api[api]["calls"] += 1
            if c.get("error") or c["status"] >= 400:
                by_api[api]["errors"] += 1
            by_api[api]["latency_sum"] += c["latency_ms"]
            by_api[api]["bytes"] += c.get("bytes", 0)

        result = {}
        for api, s in by_api.items():
            result[api] = {
                "calls": s["calls"],
                "errors": s["errors"],
                "error_rate": round(s["errors"] / max(1, s["calls"]), 3),
                "avg_latency_ms": round(s["latency_sum"] / max(1, s["calls"]), 1),
                "total_bytes": s["bytes"],
            }
        return result

    def get_cycle_stats(self, n: int = 10) -> dict:
        with self._lock:
            cycles = list(self._cycle_history)[-n:]
            errors = list(self._cycle_errors)[-10:]
            current = {
                "num": self._cycle_current_num,
                "running": self._cycle_current_start is not None,
                "elapsed_s": round(time.time() - self._cycle_current_start, 1) if self._cycle_current_start else 0,
            }
        return {
            "current": current,
            "recent": cycles,
            "recent_errors": errors,
            "total_completed": len(self._cycle_history),
        }

    def get_rt_stats(self, hours: float = 1.0) -> dict:
        cutoff = time.time() - hours * 3600
        with self._lock:
            recent = [e for e in self._rt_events if e["ts"] >= cutoff]
        total_trades = sum(e.get("trades", 0) for e in recent)
        kols = set(e["kol"] for e in recent)
        return {
            "events": len(recent),
            "trades_opened": total_trades,
            "unique_kols": len(kols),
            "last_event_age_s": round(time.time() - recent[-1]["ts"], 0) if recent else None,
        }

    def get_egress_estimate(self) -> dict:
        with self._lock:
            self._reset_day_if_needed()
            by_mod = dict(self._egress_by_module)
        total = sum(by_mod.values())
        return {
            "total_mb": round(total / (1024 * 1024), 2),
            "by_module": {k: round(v / (1024 * 1024), 2) for k, v in by_mod.items()},
            "day": self._egress_day,
        }

    def get_paper_stats(self) -> dict:
        with self._lock:
            self._reset_day_if_needed()
            return {
                "opens_today": self._paper_opens_today,
                "closes_today": self._paper_closes_today,
                "pnl_today": round(self._paper_pnl_today, 2),
            }

    def get_full_snapshot(self) -> dict:
        uptime_h = (time.time() - self._start_time) / 3600
        return {
            "uptime_hours": round(uptime_h, 1),
            "api_stats_1h": self.get_api_stats(1.0),
            "api_stats_24h": self.get_api_stats(24.0),
            "cycles": self.get_cycle_stats(20),
            "rt": self.get_rt_stats(24.0),
            "egress": self.get_egress_estimate(),
            "paper": self.get_paper_stats(),
            "snapshot_at": datetime.now(timezone.utc).isoformat(),
        }


# Module-level singleton
metrics = MetricsCollector()


# --- Helpers for instrumentation ---

class _ApiCallTracker:
    """Context manager to track an API call's latency, status, and bytes."""

    def __init__(self, api: str, endpoint: str):
        self.api = api
        self.endpoint = endpoint
        self._start = 0.0
        self._status = 0
        self._bytes = 0
        self._error: str | None = None

    def set_response(self, resp):
        """Call with the requests.Response object."""
        try:
            self._status = resp.status_code
            # Content-Length or actual content length
            cl = resp.headers.get("content-length")
            if cl:
                self._bytes = int(cl)
            else:
                self._bytes = len(resp.content) if hasattr(resp, 'content') else 0
        except Exception:
            pass

    def set_error(self, err: str):
        self._error = err
        if self._status == 0:
            self._status = 0


@contextmanager
def track_api_call(api: str, endpoint: str):
    """Context manager that records an API call to the metrics collector.

    Usage:
        with track_api_call("dexscreener", "/search") as t:
            resp = requests.get(url, timeout=10)
            t.set_response(resp)
    """
    tracker = _ApiCallTracker(api, endpoint)
    tracker._start = time.time()
    try:
        yield tracker
    except Exception as e:
        tracker.set_error(str(e)[:200])
        raise
    finally:
        latency_ms = (time.time() - tracker._start) * 1000
        metrics.record_api_call(
            api=tracker.api,
            endpoint=tracker.endpoint,
            status=tracker._status,
            latency_ms=latency_ms,
            resp_bytes=tracker._bytes,
            error=tracker._error,
        )


def estimate_egress(module: str, table: str, row_count: int):
    """Estimate and record Supabase egress for a SELECT returning row_count rows."""
    if row_count <= 0:
        return
    row_size = ROW_SIZE_ESTIMATES.get(table, 500)
    total_bytes = row_count * row_size
    metrics.record_supabase_egress(module, total_bytes)
