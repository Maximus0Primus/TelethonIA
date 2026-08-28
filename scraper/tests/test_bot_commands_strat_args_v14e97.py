"""v14e.97 — /stats donnait l'impression que les nouveaux bras "ne marchent pas".

Quatre fautes distinctes dans `bot_commands.py`, toutes SILENCIEUSES :

1. Nom raccourci non re-parsable. Le bot affiche `PFW_TP50S30_LM_WL`
   (`_short_strat` mange le `_SL`). Recopie tel quel, `_parse_strategy` ne
   trouvait rien.
2. Prefixe ambigu avale. `/stats PFWS` matche 3 bras => l'ancien
   `_parse_strategy` rendait None, et le handler repondait sur TOUT le deck
   sans le dire. L'utilisateur lit des chiffres qui ne sont pas ceux qu'il a
   demandes.
3. Bras shadow-only muets. `_query_trades` forcait `is_shadow=False`, donc
   `/stats PFWS_TP80_SL25_MED3_WL` (qui n'existe qu'en shadow) rendait
   "Aucun trade" alors que la table en a des centaines.
4. Plafond 1000 silencieux. PostgREST tronque un select non borne a 1000
   lignes ; tous les agregats "All-time" tournaient sur les 1000 dernieres.

Et la garde de non-regression : un token qui n'est PAS un nom de strategie
(`5`, `7d`, `sol`, `minN=20`) ne doit jamais declencher l'ambiguite, et un
handle KOL a underscores (`mad_apes_gambles`) ne doit pas passer pour un nom
de bras inconnu.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot_commands as bc


ACTIVE = [
    "PF_TP50_SL40_S35", "PF_BE25_TP80_SL30", "PFW_TP50_SL30_LM_WL",
    "PF_FAST_TP50_SL30_MCAP_S40", "ETH_TP100_SL50",
]


@pytest.fixture(autouse=True)
def _fixed_active(monkeypatch):
    monkeypatch.setattr(bc, "_get_active_strategies", lambda sb: list(ACTIVE))
    monkeypatch.setattr(bc, "_shadow_only_cache", {}, raising=False)


# ── 1. le nom raccourci affiche doit se reparser ──

def test_short_display_name_round_trips():
    short = bc._short_strat("PFW_TP50_SL30_LM_WL")
    assert short != "PFW_TP50_SL30_LM_WL", "sinon le test ne prouve rien"
    assert bc._parse_strategy(short, None) == "PFW_TP50_SL30_LM_WL"


def test_full_name_and_prefix_still_resolve():
    assert bc._parse_strategy("PFW_TP50_SL30_LM_WL", None) == "PFW_TP50_SL30_LM_WL"
    assert bc._parse_strategy("pfw", None) == "PFW_TP50_SL30_LM_WL"


# ── 2. l'ambiguite est rapportee, pas avalee ──

def test_ambiguous_prefix_reports_candidates():
    name, candidates = bc._resolve_strategy("PFWS", None)
    assert name is None
    assert len(candidates) > 1
    assert all(c.startswith("PFWS") for c in candidates)


def test_ambiguous_token_surfaces_as_error_not_global_stats():
    remaining, strat, err = bc._split_strategy_args("PFWS", None)
    assert strat is None
    assert err and "PFWS" in err


def test_unknown_strat_shaped_token_is_reported():
    _, strat, err = bc._split_strategy_args("PF_XX_YY_ZZ", None)
    assert strat is None
    assert err and "PF_XX_YY_ZZ" in err


# ── garde: les autres arguments ne sont pas des strategies ──

@pytest.mark.parametrize("tok", ["5", "10", "7d", "24h", "all", "sol", "eth", "minN=20"])
def test_non_strategy_tokens_never_raise_ambiguity(tok):
    name, candidates = bc._resolve_strategy(tok, None)
    assert (name, candidates) == (None, [])
    _, strat, err = bc._split_strategy_args(tok, None)
    assert strat is None and err is None


def test_kol_handle_with_underscores_not_flagged_in_free_text():
    # `/pnl mad_apes_gambles` : 2 underscores => strat-shaped, mais c'est un KOL.
    _, _, err = bc._split_strategy_args("mad_apes_gambles", None, free_text=True)
    assert err is None


# ── 3. un bras shadow-only est lu dans les lignes shadow ──

class _FakeQuery:
    """Enregistre les filtres poses, rend des lignes sur .execute()."""

    def __init__(self, store, rows=()):
        self.store = store
        self._rows = list(rows)

    def select(self, *a, **k):
        self.store["_select"] = a[0] if a else ""
        return self

    def order(self, *a, **k): return self
    def limit(self, n): self.store["limit"] = n; return self
    def range(self, lo, hi): self.store["range"] = (lo, hi); return self
    def neq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def in_(self, col, vals): self.store[col] = list(vals); return self

    def eq(self, col, val):
        self.store[col] = val
        return self

    def execute(self):
        return type("R", (), {"data": self._rows})()


class _FakeSB:
    """Dispatch sur les colonnes selectionnees: la sonde _is_shadow_only ne
    demande que `id`, la vraie requete demande la liste complete."""

    def __init__(self, store, main_rows=(), rows=()):
        self.store = store
        self.main_rows = main_rows
        self.rows = rows

    def table(self, _name):
        probe: dict = {}
        real = _FakeQuery(self.store, self.rows)
        outer = self

        class _Dispatch:
            def select(self, cols, *a, **k):
                if cols == "id":
                    return _FakeQuery(probe, outer.main_rows)
                return real.select(cols, *a, **k)

        return _Dispatch()


def test_shadow_only_arm_is_queried_as_shadow():
    store = {}
    sb = _FakeSB(store, main_rows=[])  # aucune ligne main => bras shadow-only
    bc._query_trades(sb, strategy="PFWS_TP80_SL25_MED3_WL")
    assert store["is_shadow"] is True


def test_arm_with_main_rows_stays_main():
    store = {}
    sb = _FakeSB(store, main_rows=[{"id": 1}])
    bc._query_trades(sb, strategy="PFW_TP50_SL30_LM_WL")
    assert store["is_shadow"] is False


# ── 4. plus de plafond 1000 silencieux ──

class _PagedQuery:
    def __init__(self, total):
        self.total = total
        self.pages = []

    def range(self, lo, hi):
        self.pages.append((lo, hi))
        n = max(0, min(hi + 1, self.total) - lo)
        self._next = [{"i": lo + k} for k in range(n)]
        return self

    def execute(self):
        return type("R", (), {"data": self._next})()


def test_fetch_all_pages_past_the_1000_row_cap():
    q = _PagedQuery(total=2500)
    rows = bc._fetch_all(q)
    assert len(rows) == 2500
    assert len(q.pages) == 3
    assert q.pages[0] == (0, 999)


def test_fetch_all_stops_on_short_page():
    q = _PagedQuery(total=42)
    assert len(bc._fetch_all(q)) == 42
    assert len(q.pages) == 1


# ── 5. /best /worst /pnl : le meme bug, sur un chemin qui ne passait pas par
#      _query_trades (requetes ecrites a la main dans chaque handler) ──

class _ExtremeQuery:
    def __init__(self, store):
        self.store = store

    def select(self, cols, *a, **k):
        if cols == "id":                     # sonde _is_shadow_only
            return _FakeQuery({}, self.store.get("_main_rows", []))
        return self

    def eq(self, col, val): self.store[col] = val; return self
    def in_(self, col, vals): self.store[col] = list(vals); return self
    def neq(self, *a, **k): return self
    def order(self, col, desc=True): self.store["order"] = (col, desc); return self
    def limit(self, n): self.store["limit"] = n; return self
    def gte(self, col, val): self.store.setdefault("gte", []).append(col); return self
    def execute(self): return type("R", (), {"data": []})()


class _ExtremeSB:
    def __init__(self, store): self.store = store
    def table(self, _n): return _ExtremeQuery(self.store)


def test_best_applies_the_period_instead_of_swallowing_it():
    # `/best 7d` rendait l'all-time sans le dire : aucun filtre exit_at pose.
    store = {"_main_rows": [{"id": 1}]}
    out = bc._handle_best(_ExtremeSB(store), "7d")
    assert "exit_at" in store.get("gte", []), "la periode doit filtrer la requete"
    assert "7d" in out


def test_best_without_period_stays_all_time():
    store = {"_main_rows": [{"id": 1}]}
    out = bc._handle_best(_ExtremeSB(store), "")
    assert "gte" not in store
    assert "All-time" in out


def test_worst_sorts_ascending_and_best_descending():
    s1, s2 = {"_main_rows": [{"id": 1}]}, {"_main_rows": [{"id": 1}]}
    bc._handle_best(_ExtremeSB(s1), "")
    bc._handle_worst(_ExtremeSB(s2), "")
    assert s1["order"] == ("pnl_usd", True)
    assert s2["order"] == ("pnl_usd", False)


def test_best_reports_ambiguous_strategy():
    out = bc._handle_best(_ExtremeSB({"_main_rows": [{"id": 1}]}), "PFWS")
    assert out.startswith("❓")


def test_best_reads_shadow_rows_for_a_shadow_only_arm():
    store = {"_main_rows": []}          # aucune ligne main => shadow-only
    bc._handle_best(_ExtremeSB(store), "PFWS_TP80_SL25_MED3_WL")
    assert store["is_shadow"] is True


def test_pnl_reads_shadow_rows_for_a_shadow_only_arm():
    store = {"_main_rows": []}
    bc._handle_pnl(_ExtremeSB(store), "mad_apes_gambles PFWS_TP80_SL25_MED3_WL")
    assert store["is_shadow"] is True


# ── 6. /shadow : 1.53 M lignes shadow. Paginer par OFFSET tuait la requete
#      (57014) ; l'ancien code rendait en silence les 1000 dernieres lignes,
#      soit ~40 min de grille presentees comme "All-time". ──

class _SeekQuery:
    def __init__(self, rows, calls):
        self.rows = rows
        self.calls = calls
        self._lte = None
        self._limit = 1000

    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def neq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, n): self._limit = n; return self

    def lte(self, col, val):
        self._lte = val
        return self

    def execute(self):
        self.calls.append(self._lte)
        rows = [r for r in self.rows if self._lte is None or r["exit_at"] <= self._lte]
        return type("R", (), {"data": rows[: self._limit]})()


def test_seek_pagination_uses_a_cursor_not_an_offset():
    rows = [{"id": i, "exit_at": f"2026-08-{28 - i // 500:02d}T00:00:{i % 60:02d}"}
            for i in range(2500)]
    rows.sort(key=lambda r: r["exit_at"], reverse=True)
    calls = []
    got = bc._fetch_all_seek(lambda: _SeekQuery(rows, calls), max_rows=5000)
    assert len(got) == 2500, "toutes les lignes, sans doublon"
    assert len({r["id"] for r in got}) == 2500
    assert calls[0] is None and calls[1] is not None, "page 2 doit porter un curseur"


def test_seek_dedups_rows_sharing_a_timestamp():
    # La grille ferme des dizaines de lignes sur le MEME exit_at : le curseur est
    # `<=`, donc la page suivante rechevauche et doit dedupliquer par id.
    rows = [{"id": i, "exit_at": "2026-08-28T00:00:00"} for i in range(1500)]
    got = bc._fetch_all_seek(lambda: _SeekQuery(rows, []), max_rows=5000)
    assert len({r["id"] for r in got}) == len(got), "aucun doublon"


def test_seek_respects_the_row_budget():
    rows = [{"id": i, "exit_at": f"2026-08-28T00:00:{i % 60:02d}"} for i in range(5000)]
    got = bc._fetch_all_seek(lambda: _SeekQuery(rows, []), max_rows=1200)
    assert len(got) == 1200


@pytest.mark.parametrize("arg,expect_clamp", [
    ("1h", False), ("24h", False), ("7d", False), ("14d", True), ("30d", True),
])
def test_shadow_window_is_clamped_and_says_so(arg, expect_clamp, monkeypatch):
    seen = {}

    def fake_seek(build_q, max_rows, cursor_col="exit_at"):
        seen["called"] = True
        return []

    monkeypatch.setattr(bc, "_fetch_all_seek", fake_seek)
    out = bc._handle_shadow(None, arg)
    assert seen.get("called")
    assert ("ramene a 7d" in out) is expect_clamp
