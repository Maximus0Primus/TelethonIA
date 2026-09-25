"""Deadlock 40P01 sur `tokens` — ordre de verrouillage des upserts.

Deux ecrivains concurrents upsertent `tokens` sur la meme cle de conflit
(symbol, time_window, token_address) :

* ``push_to_supabase.upsert_tokens``     — lignes dans l'ordre du SCORE ;
* ``price_refresh.refresh_top_tokens``   — lignes groupees token par token,
  a travers toutes les fenetres.

Postgres verrouille les lignes dans l'ordre du VALUES : deux ordres differents
sur les memes lignes => deadlock. Correctif attendu : les deux ecrivains
envoient leurs lignes triees par la cle de conflit, un ordre total unique.
"""
from unittest.mock import MagicMock, patch

import price_refresh
import push_to_supabase


def _conflict_key(row: dict) -> tuple[str, str, str]:
    return (row["symbol"], row["time_window"], row["token_address"])


class _FakeTable:
    """Query builder minimal : toute methode chainee renvoie self,
    `upsert` enregistre les lignes envoyees pour la table `tokens`."""

    def __init__(self, name: str, select_data: list[dict], upserts: list[list[dict]]):
        self._name = name
        self._select_data = select_data
        self._upserts = upserts

    def upsert(self, rows, **_kwargs):
        if self._name == "tokens":
            self._upserts.append(list(rows))
        return self

    def __getattr__(self, _method):  # select, eq, in_, order, limit, insert, delete...
        return lambda *a, **k: self

    def execute(self):
        res = MagicMock()
        res.data = self._select_data
        return res


def _fake_client(tokens_select: list[dict]) -> tuple[MagicMock, list[list[dict]]]:
    upserts: list[list[dict]] = []
    client = MagicMock()
    client.table.side_effect = lambda name: _FakeTable(
        name, tokens_select if name == "tokens" else [], upserts
    )
    return client, upserts


# ---------------------------------------------------------------------------
# push_to_supabase.upsert_tokens
# ---------------------------------------------------------------------------

def _ranked(symbol: str, score: float, address: str) -> dict:
    return {
        "symbol": symbol, "score": score, "mentions": 3, "unique_kols": 2,
        "sentiment": 0.5, "trend": "up", "token_address": address,
    }


def test_upsert_tokens_sends_rows_in_conflict_key_order():
    # Classement par score : ZZZ (90) avant MMM (70) avant AAA (50),
    # et deux CA pour AAA donnees a rebours.
    ranking = {
        "24h": [
            _ranked("ZZZ", 90, "zaddr"),
            _ranked("MMM", 70, "maddr"),
            _ranked("AAA", 60, "addr_b"),
            _ranked("AAA", 50, "addr_a"),
        ],
        "7d": [
            _ranked("ZZZ", 80, "zaddr"),
            _ranked("AAA", 40, "addr_a"),
        ],
    }
    client, upserts = _fake_client(tokens_select=[])

    with patch.object(push_to_supabase, "_get_client", return_value=client):
        push_to_supabase.upsert_tokens(ranking, stats={})

    assert len(upserts) == 2
    for rows in upserts:
        assert rows == sorted(rows, key=_conflict_key), [_conflict_key(r) for r in rows]


# ---------------------------------------------------------------------------
# price_refresh.refresh_top_tokens
# ---------------------------------------------------------------------------

def test_refresh_top_tokens_sends_rows_in_conflict_key_order():
    # Top N par score : ZZZ avant AAA avant MMM. Le fake renvoie les memes
    # lignes pour chaque select sur `tokens` => chaque token existe dans les
    # 6 fenetres, et les lignes partent groupees token par token.
    top = [
        {"symbol": "ZZZ", "score": 90, "base_score": 90, "token_address": "zaddr",
         "freshest_mention_hours": 1.0, "change_24h": 0},
        {"symbol": "AAA", "score": 70, "base_score": 70, "token_address": "aaddr",
         "freshest_mention_hours": 1.0, "change_24h": 0},
        {"symbol": "MMM", "score": 50, "base_score": 50, "token_address": "maddr",
         "freshest_mention_hours": 1.0, "change_24h": 0},
    ]
    market = {"price_usd": 1.0, "price_change_24h": 5.0, "buy_sell_ratio_1h": 0.5}
    client, upserts = _fake_client(tokens_select=top)

    with patch.object(price_refresh, "_get_supabase", return_value=client), \
         patch.object(price_refresh, "_monitoring", False), \
         patch.object(price_refresh, "_fetch_dexscreener_batch",
                      return_value={t["token_address"]: market for t in top}):
        price_refresh.refresh_top_tokens(n=3)

    assert len(upserts) == 1
    rows = upserts[0]
    assert len(rows) == 3 * 6  # 3 tokens x 6 fenetres
    assert rows == sorted(rows, key=_conflict_key), [_conflict_key(r) for r in rows]
