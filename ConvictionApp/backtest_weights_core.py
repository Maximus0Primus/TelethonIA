# backtest_weights_core.py
# v7.3 – Fix concat(liste vide) + GT pagination robuste + helpers Birdeye/CG
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ==================== LOAD DATA ====================

def _parse_json_file(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            msgs = obj.get("messages", obj) if isinstance(obj, dict) else obj
            if isinstance(msgs, list):
                rows.extend(msgs)
    except Exception:
        pass
    return rows


def try_load_app_dataset(data_dir: str) -> pd.DataFrame:
    """
    Essaie d’abord de charger le dataset en mémoire via utils.ensure_session_dataset(),
    sinon lit data/telegram/*.json|*.jsonl.
    """
    try:
        from utils import ensure_session_dataset  # type: ignore
        df = ensure_session_dataset()
        if df is not None and not df.empty:
            return _normalize_dataset(df)
    except Exception:
        pass

    base = Path(data_dir)
    rows: List[dict] = []
    if base.is_dir():
        for path in sorted(base.glob("*.json")) + sorted(base.glob("*.jsonl")):
            rows.extend(_parse_json_file(path))
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    return _normalize_dataset(df)


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "id" not in out.columns:
        out["id"] = np.arange(len(out))

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_convert(None)
    elif "date_utc" in out.columns:
        out["date"] = pd.to_datetime(out["date_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        out["date"] = pd.Timestamp.utcnow().tz_localize(None)

    if "group" not in out.columns:
        out["group"] = "unknown"

    if "tokens" in out.columns:
        def _norm_tokens(x):
            if isinstance(x, str):
                try:
                    maybe = json.loads(x)
                    if isinstance(maybe, list):
                        x = maybe
                except Exception:
                    pass
            if isinstance(x, list):
                return [str(t).upper().replace("$", "").strip() for t in x if t]
            if pd.isna(x):
                return []
            return [str(x).upper().replace("$", "").strip()]
        out["tokens"] = out["tokens"].apply(_norm_tokens)
    else:
        out["tokens"] = [[] for _ in range(len(out))]

    out = out.sort_values("date").drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    return out[["id", "date", "group", "tokens"]].copy()


# ==================== PARAMS ====================

@dataclass
class FeatureWindow:
    step_h: int = 2
    bins_n: int = 12

@dataclass
class SentimentKnobs:
    use_hf: bool = False
    w_hf: float = 0.50
    w_vader: float = 0.35
    w_crypto: float = 0.15
    rule_weight: float = 1.0
    group_alpha: float = 1.0
    alias_no_dollar: bool = True
    gain: float = 1.30


# ==================== FEATURES (branchées sur utils.*) ====================

def compute_features_from_app(
    raw_df: pd.DataFrame,
    win: FeatureWindow,
    k: SentimentKnobs,
    tau_hours: float = 12.0
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["token", "ts_signal_utc", "horizon"])

    try:
        from utils import add_sentiment, explode_tokens  # type: ignore
    except Exception:
        df = raw_df.copy()
        df["w_sentiment"] = 0.0
        df["sentiment"] = 0.0
        df["tokens"] = df["tokens"].apply(lambda x: x if isinstance(x, list) else [])
        dt = _naive_explode(df)
    else:
        df = add_sentiment(
            raw_df.copy(),
            use_hf=k.use_hf,
            w_vader=k.w_vader,
            w_crypto=k.w_crypto,
            w_hf=k.w_hf,
            rule_weight=k.rule_weight,
            group_weight_alpha=k.group_alpha,
            alias_no_dollar=k.alias_no_dollar,
            gain=k.gain,
        )
        for c in ["w_sentiment", "sentiment"]:
            if c not in df.columns:
                df[c] = 0.0
        dt = explode_tokens(df[["id", "date", "group", "w_sentiment", "sentiment", "tokens"]])

    if dt.empty:
        return pd.DataFrame(columns=["token", "ts_signal_utc", "horizon"])

    end_time = dt["date"].max()
    if pd.isna(end_time):
        end_time = pd.Timestamp.utcnow().tz_localize(None)
    start_time = end_time - pd.Timedelta(hours=win.bins_n * win.step_h)
    dt_win = dt[(dt["date"] >= start_time) & (dt["date"] <= end_time)].copy()
    if dt_win.empty:
        return pd.DataFrame(columns=["token", "ts_signal_utc", "horizon"])
    dt_win["bin"] = dt_win["date"].dt.floor(f"{win.step_h}h")

    agg = (dt_win.groupby("token")
           .agg(
               mentions=("id", "count"),
               sentiment=("w_sentiment", "mean"),
               std=("w_sentiment", "std"),
               groups_used=("group", "nunique")
           ).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(np.maximum(1, agg["mentions"]))

    total_groups = max(1, int(df["group"].nunique()))
    agg["breadth"] = (agg["groups_used"] / total_groups).clip(0, 1)

    grp_sent = dt_win.groupby(["token", "group"])["w_sentiment"].mean().reset_index()
    grp_pos = grp_sent.assign(pos=(grp_sent["w_sentiment"] > 0).astype(int)) \
                      .groupby("token")["pos"].mean().reset_index(name="p_pos_groups")
    agg = agg.merge(grp_pos, on="token", how="left")
    agg["p_pos_groups"] = agg["p_pos_groups"].fillna(0.0).clip(0, 1)
    agg["polarisation"] = 4.0 * agg["p_pos_groups"] * (1.0 - agg["p_pos_groups"])

    msg_pos = dt_win.assign(pos=(dt_win["w_sentiment"] > 0).astype(int)) \
                    .groupby("token")["pos"].sum().reset_index(name="pos_count")
    agg = agg.merge(msg_pos, on="token", how="left").fillna({"pos_count": 0})

    def wilson_lower_bound(pos: int, n: int, z: float = 1.96) -> float:
        if n <= 0: return 0.0
        p = pos / n
        denom = 1 + z*z/n
        centre = p + z*z/(2*n)
        margin = z*np.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
        return float((centre - margin) / denom)

    agg["wilson_low"] = [wilson_lower_bound(int(p), int(n)) for p, n in zip(agg["pos_count"], agg["mentions"])]

    bins_all = pd.date_range(start=start_time.floor(f"{win.step_h}h"),
                             end=end_time.ceil(f"{win.step_h}h"),
                             freq=f"{win.step_h}h")
    m = (dt_win.groupby(["token", "bin"])["id"].count()
         .reindex(pd.MultiIndex.from_product([agg["token"], bins_all], names=["token", "bin"]))
         .fillna(0.0).reset_index())
    x = np.arange(len(bins_all))

    def _slope_acc(vals):
        if len(vals) < 2: return 0.0, 0.0
        y = np.asarray(vals, dtype=float)
        try: b1 = np.polyfit(x, y, 1)[0]
        except Exception: b1 = 0.0
        try: a2 = np.polyfit(x, y, 2)[0]
        except Exception: a2 = 0.0
        return float(b1), float(2.0 * a2)

    mom = m.groupby("token")["id"].apply(lambda s: _slope_acc(s.values)[0]).reset_index(name="momentum")
    acc = m.groupby("token")["id"].apply(lambda s: _slope_acc(s.values)[1]).reset_index(name="accel")
    agg = agg.merge(mom, on="token", how="left").merge(acc, on="token", how="left")
    agg[["momentum", "accel"]] = agg[["momentum", "accel"]].fillna(0.0)

    pr_df = _token_pagerank_from_utils(df, tau_hours=tau_hours)
    agg = agg.merge(pr_df, on="token", how="left")
    agg["pagerank"] = agg["pagerank"].fillna(0.0)
    mpr = float(agg["pagerank"].max()) or 1.0
    agg["pr_norm"] = (agg["pagerank"] / mpr).clip(0, 1).fillna(0.0)

    try:
        import streamlit as st
        alpha = float(st.session_state.get("mentions_alpha", 0.6))
    except Exception:
        alpha = 0.6
    max_m = max(1, int(agg["mentions"].max()))
    norm_m = agg["mentions"] / max_m
    norm_s = (agg["sentiment"] + 1.0) / 2.0
    agg["score_conviction"] = (10.0 * (alpha * norm_m + (1 - alpha) * norm_s)).round(2)

    scg = _conviction_graph_score_from_utils(df, agg[["token", "score_conviction"]], tau_hours=tau_hours)
    agg = agg.merge(scg, on="token", how="left")

    if not agg.empty:
        bmax = max(1, agg["breadth"].max())
        pmax = max(1e-6, agg["polarisation"].max())
        m_abs = max(1e-6, agg["momentum"].abs().max(skipna=True))
        a_abs = max(1e-6, agg["accel"].abs().max(skipna=True))
        sent01 = (agg["sentiment"] + 1) / 2
        wil = agg["wilson_low"].clip(0, 1)
        br01 = (agg["breadth"] / bmax).clip(0, 1)
        pol01i = (1 - (agg["polarisation"] / pmax).clip(0, 1))
        mom01 = (agg["momentum"] / m_abs / 2 + 0.5).fillna(0.5).clip(0, 1)
        acc01 = (agg["accel"] / a_abs / 2 + 0.5).fillna(0.5).clip(0, 1)
        quick = 0.30*sent01 + 0.25*wil + 0.20*br01 + 0.15*mom01 + 0.10*acc01
        quick = quick * (0.7 + 0.3 * pol01i)
        agg["score_quick_win"] = (10.0 * quick).round(2)

    agg["ts_signal_utc"] = end_time
    agg["horizon"] = "mid"
    cols = [
        "token","ts_signal_utc","horizon",
        "mentions","sentiment","ci95","breadth","polarisation","wilson_low",
        "pagerank","pr_norm","momentum","accel",
        "score_conviction","score_conviction_graph","score_quick_win"
    ]
    return agg[[c for c in cols if c in agg.columns]].copy()


def _naive_explode(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        toks = r.get("tokens") or []
        for t in toks:
            rows.append({
                "id": r.get("id"),
                "group": r.get("group"),
                "date": r.get("date"),
                "token": t,
                "w_sentiment": r.get("w_sentiment", 0.0),
                "sentiment": r.get("sentiment", 0.0),
            })
    return pd.DataFrame(rows)


def _token_pagerank_from_utils(df: pd.DataFrame, tau_hours: float = 12.0) -> pd.DataFrame:
    try:
        from utils import graph_edges_advanced  # type: ignore
        import networkx as nx  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["token", "pagerank"])
    edges, node_sent, _ = graph_edges_advanced(df, tau_hours=float(tau_hours), group_sent_source="calc")
    if edges.empty:
        return pd.DataFrame(columns=["token", "pagerank"])
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r.get("weight", 0.0)))
    try:
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {}
    except Exception:
        pr = {}
    token_nodes = set(node_sent.loc[node_sent["kind"] == "token", "node"].tolist())
    rows = [{"token": t, "pagerank": float(pr.get(t, 0.0))} for t in token_nodes]
    return pd.DataFrame(rows)


def _conviction_graph_score_from_utils(df_raw: pd.DataFrame, base_scores: pd.DataFrame, tau_hours: float = 12.0) -> pd.DataFrame:
    out = pd.DataFrame(columns=["token", "score_conviction_graph"])
    if base_scores is None or base_scores.empty:
        return out
    try:
        from utils import graph_edges_advanced  # type: ignore
        import networkx as nx  # type: ignore
        import streamlit as st
    except Exception:
        return out

    wA = float(getattr(st.session_state, "wA", 0.60))
    wC = float(getattr(st.session_state, "wC", 0.40))
    wPRT = float(getattr(st.session_state, "wPRT", 0.20))
    gamma = float(getattr(st.session_state, "gamma_struct", 0.25))

    edges, node_sent, _ = graph_edges_advanced(df_raw, tau_hours=float(tau_hours), group_sent_source="calc")
    if edges.empty:
        return base_scores.rename(columns={"score_conviction": "score_conviction_graph"})[["token", "score_conviction_graph"]]

    e = edges.copy()
    e_gt = e[e["type"] == "group-token"]

    G = nx.Graph()
    for _, r in e.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r["weight"]))
    try:
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {}
    except Exception:
        pr = {}

    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, weight="weight", seed=42) if G.number_of_edges() > 0 else [{n} for n in G.nodes()]
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight")) if G.number_of_edges() > 0 else [{n} for n in G.nodes()]
    cluster_id = {}
    for ci, community in enumerate(comms, start=1):
        for n in community:
            cluster_id[n] = ci

    A, C, PRT = {}, {}, {}
    if not e_gt.empty:
        for t, g in e_gt.groupby("dst"):
            s = 0.0
            for _, r in g.iterrows():
                s += float(r["weight"]) * float(pr.get(str(r["src"]), 0.0))
            A[str(t)] = s
        m = max(A.values()) if A else 1.0
        A = {k: (v/m if m > 0 else 0.0) for k, v in A.items()}

        for t, g in e_gt.groupby("dst"):
            cls = set(cluster_id.get(str(r["src"]), 0) for _, r in g.iterrows())
            C[str(t)] = len([c for c in cls if c != 0])
        m = max(C.values()) if C else 1.0
        C = {k: (v/m if m > 0 else 0.0) for k, v in C.items()}

    token_nodes = set(node_sent.loc[node_sent["kind"] == "token", "node"].tolist())
    PRT = {t: float(pr.get(str(t), 0.0)) for t in token_nodes}
    if len(PRT) > 0:
        m = max(PRT.values())
        PRT = {k: (v/m if m > 0 else 0.0) for k, v in PRT.items()}

    ws = max(1e-9, wA + wC + wPRT)
    wA, wC, wPRT = wA / ws, wC / ws, wPRT / ws

    base = base_scores.copy()
    base01 = (base["score_conviction"] / 10.0).clip(0, 1)
    base["AutoritéGroupes"] = base["token"].map(A).fillna(0.0)
    base["ConvergenceClusters"] = base["token"].map(C).fillna(0.0)
    base["CentralitéPR"] = base["token"].map(PRT).fillna(0.0)

    struct = (wA*base["AutoritéGroupes"] + wC*base["ConvergenceClusters"] + wPRT*base["CentralitéPR"]).clip(0, 1)
    base["score_conviction_graph"] = (10.0 * ((1 - gamma) * base01 + gamma * struct)).round(2)
    return base[["token", "score_conviction_graph"]]


# ==================== Dexscreener / GeckoTerminal ====================

class PairCache:
    def __init__(self, path: str | Path = "cache/pair_cache.json", ttl_days: int = 7):
        self.path = Path(path)
        self.ttl_days = int(ttl_days)
        self.data: Dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, token: str) -> Optional[dict]:
        tok = token.upper().replace("$", "").strip()
        return self.data.get(tok)

    def set(self, token: str, chain_id: str, pair_address: str, network: str, address: Optional[str] = None):
        tok = token.upper().replace("$", "").strip()
        self.data[tok] = {
            "chainId": chain_id,
            "tokenAddress": address,
            "pairAddress": pair_address,
            "network": network,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self.save()

    def clear(self):
        self.data = {}
        self.save()


class OhlcvCache:
    def __init__(self, dirpath: str | Path = "cache/ohlcv", ttl_days: int = 3):
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = int(ttl_days)

    def _file(self, network: str, pool: str) -> Path:
        safe_pool = "".join(ch if ch.isalnum() else "_" for ch in pool)[:128]
        return self.dir / f"{network}__{safe_pool}.csv"

    def load(self, network: str, pool: str) -> pd.DataFrame:
        f = self._file(network, pool)
        if not f.exists():
            return pd.DataFrame(columns=["ts_price_utc", "price"])
        try:
            df = pd.read_csv(f, parse_dates=["ts_price_utc"])
            return df.dropna().sort_values("ts_price_utc").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["ts_price_utc", "price"])

    def save(self, network: str, pool: str, df: pd.DataFrame):
        f = self._file(network, pool)
        try:
            df = df.drop_duplicates(subset=["ts_price_utc"]).sort_values("ts_price_utc")
            df.to_csv(f, index=False)
        except Exception:
            pass


def _map_chain_to_gt_network(chain_id: str) -> Optional[str]:
    c = (chain_id or "").lower()
    mapping = {
        "ethereum":"eth","eth":"eth",
        "bsc":"bsc","binance-smart-chain":"bsc",
        "polygon":"polygon","matic":"polygon",
        "arbitrum":"arbitrum","arbitrum-one":"arbitrum",
        "optimism":"optimism","base":"base",
        "avalanche":"avax","avax":"avax",
        "fantom":"ftm","ftm":"ftm",
        "solana":"solana","sei":"sei",
        "blast":"blast","zksync":"zksync",
        "linea":"linea","scroll":"scroll","opbnb":"opbnb",
    }
    return mapping.get(c, None)


# ---------- GeckoTerminal helpers ----------

_GT_NETWORKS_SCAN = [
    "solana","eth","base","bsc","polygon","arbitrum","avax","optimism","ftm",
    "sei","blast","zksync","linea","scroll","opbnb"
]

def _gt_headers():
    return {"accept": "application/json;version=20230302"}

def gt_pool_info(network: str, pool_address: str) -> Tuple[bool, Optional[str], Optional[str]]:
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
    try:
        r = requests.get(url, headers=_gt_headers(), timeout=15)
        if r.status_code == 404:
            return False, None, None
        r.raise_for_status()
        js = r.json()
        data = js.get("data", {})
        attr = data.get("attributes", {})
        base = (attr.get("base_token") or {}).get("symbol")
        quote = (attr.get("quote_token") or {}).get("symbol")
        return True, str(base) if base else None, str(quote) if quote else None
    except Exception:
        return False, None, None

def gt_probe_pool_network(pool_address: str, preferred: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    tried = []
    if preferred:
        ok, b, q = gt_pool_info(preferred, pool_address)
        if ok:
            return preferred, b, q
        tried.append(preferred)
    for net in _GT_NETWORKS_SCAN:
        if net in tried:
            continue
        ok, b, q = gt_pool_info(net, pool_address)
        if ok:
            return net, b, q
    return None, None, None

def _gt_fetch_page_generic(network: str, pool_address: str, granularity: str,
                           aggregate: int, before_ts: Optional[int], limit: int) -> pd.DataFrame:
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{granularity}"
    params = {"aggregate": int(aggregate), "limit": int(limit)}
    if before_ts is not None:
        params["before_timestamp"] = int(before_ts)
    try:
        r = requests.get(url, params=params, headers=_gt_headers(), timeout=20)
        r.raise_for_status()
        js = r.json()
        data = js.get("data") if isinstance(js, dict) else None
        attrs = (data or {}).get("attributes", {})
        ohlcv = attrs.get("ohlcv_list") or []
        rows = []
        for row in ohlcv:
            try:
                ts_raw = row[0]
                close_raw = row[4]
                ts = pd.to_datetime(ts_raw, unit="ms", utc=True, errors="coerce")
                if pd.isna(ts):
                    ts = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
                ts = ts.tz_convert(None)
                c = float(close_raw)
            except Exception:
                continue
            if pd.isna(ts) or pd.isna(c):
                continue
            rows.append({"ts_price_utc": ts, "price": c})
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["ts_price_utc", "price"])
        return df.dropna().sort_values("ts_price_utc").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["ts_price_utc", "price"])

def _is_valid_ohlcv(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "ts_price_utc" not in df.columns or "price" not in df.columns:
        return False
    if df["ts_price_utc"].isna().all() or df["price"].isna().all():
        return False
    return True

def _gt_fetch_span(network: str, pool_address: str, granularity: str, aggregate: int,
                   start: pd.Timestamp, end: pd.Timestamp, max_pages: int) -> pd.DataFrame:
    """
    Pagination robuste: on accumule dans une liste, concat 1x, et on évite toute concat sur liste vide.
    """
    parts: List[pd.DataFrame] = []
    page = 0
    last_before = None
    empty_streak = 0
    t0 = time.time()

    # Pages récentes
    while page < max_pages:
        if time.time() - t0 > 25:
            break
        dfp = _gt_fetch_page_generic(network, pool_address, granularity, aggregate,
                                     before_ts=None if page == 0 else last_before, limit=300)
        if not _is_valid_ohlcv(dfp):
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            parts.append(dfp)
            new_min = dfp["ts_price_utc"].min()
            if pd.notna(new_min):
                last_before = int(new_min.timestamp()) - 1
        page += 1
        if parts and max(p["ts_price_utc"].max() for p in parts) >= end - pd.Timedelta(days=1):
            break

    # Concat provisoire
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts_price_utc","price"])
    if not df.empty:
        df = df.drop_duplicates(subset=["ts_price_utc"]).sort_values("ts_price_utc")

    # Pages anciennes
    page = 0
    empty_streak = 0
    prev_len = len(df)
    cur_min_ts = int((df["ts_price_utc"].min() if not df.empty else end).timestamp())
    while page < max_pages:
        if time.time() - t0 > 45:
            break
        dfp = _gt_fetch_page_generic(network, pool_address, granularity, aggregate,
                                     before_ts=cur_min_ts, limit=300)
        if not _is_valid_ohlcv(dfp):
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            parts.append(dfp)
            cur_min = dfp["ts_price_utc"].min()
            if pd.notna(cur_min):
                cur_min_ts = int(cur_min.timestamp()) - 1
        page += 1

        # Progression?
        if parts:
            df_tmp = pd.concat(parts, ignore_index=True)
            df_tmp = df_tmp.drop_duplicates(subset=["ts_price_utc"]).sort_values("ts_price_utc")
        else:
            df_tmp = pd.DataFrame(columns=["ts_price_utc","price"])
        if len(df_tmp) == prev_len:
            break
        prev_len = len(df_tmp)
        if not df_tmp.empty and df_tmp["ts_price_utc"].min() <= start - pd.Timedelta(days=1):
            df = df_tmp
            break
        df = df_tmp

    if not df.empty:
        df = df[(df["ts_price_utc"] >= start.floor("D") - pd.Timedelta(days=2)) &
                (df["ts_price_utc"] <= end.ceil("D") + pd.Timedelta(days=2))].reset_index(drop=True)
    return df

def gt_fetch_ohlcv_resampled_day_cached(network: str, pool_address: str,
                                        start: pd.Timestamp, end: pd.Timestamp,
                                        ohlcv_cache: Optional[OhlcvCache] = None,
                                        max_pages: int = 8) -> Tuple[pd.DataFrame, str]:
    cache = ohlcv_cache or OhlcvCache()
    cached = cache.load(network, pool_address)
    if not cached.empty and cached["ts_price_utc"].min() <= start - pd.Timedelta(days=1) and cached["ts_price_utc"].max() >= end - pd.Timedelta(days=1):
        return cached[["ts_price_utc","price"]].copy(), "cache"

    df_day = _gt_fetch_span(network, pool_address, "day", 1, start, end, max_pages=max_pages)
    if _is_valid_ohlcv(df_day):
        df_day = df_day[["ts_price_utc","price"]].copy()
        cache.save(network, pool_address, df_day)
        return df_day, "day"

    df_min = _gt_fetch_span(network, pool_address, "minute", 15, start, end, max_pages=max_pages)
    if _is_valid_ohlcv(df_min):
        dfr = (df_min.set_index("ts_price_utc").resample("1D").last().dropna().reset_index())
        if _is_valid_ohlcv(dfr):
            cache.save(network, pool_address, dfr)
            return dfr[["ts_price_utc","price"]].copy(), "minute"
    return pd.DataFrame(columns=["ts_price_utc","price"]), "empty"


# ---------- Dexscreener ----------

def _dexs_search(query: str) -> List[dict]:
    url = "https://api.dexscreener.com/latest/dex/search"
    try:
        r = requests.get(url, params={"q": query}, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        pairs = (data.get("pairs") or []) if isinstance(data, dict) else []
        return pairs if isinstance(pairs, list) else []
    except Exception:
        return []

def dexs_token_pairs(chain_id: str, token_address: str) -> List[dict]:
    url = f"https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _pairs_sorted_by_liquidity(pairs: List[dict]) -> List[dict]:
    def _liq(p):
        try:
            return float(((p.get("liquidity") or {}).get("usd") or 0.0))
        except Exception:
            return 0.0
    return sorted(pairs, key=_liq, reverse=True)


# ---------- BUILD PRICES ----------

def build_prices_auto_from_tokens_debug(tokens: List[str],
                                        tmin: pd.Timestamp,
                                        tmax: pd.Timestamp,
                                        pair_cache: Optional[PairCache] = None,
                                        ohlcv_cache: Optional[OhlcvCache] = None,
                                        max_pages: int = 10) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    prices: Dict[str, pd.DataFrame] = {}
    dbg: List[dict] = []
    pcache = pair_cache or PairCache()
    ocache = ohlcv_cache or OhlcvCache()
    hints = _load_user_hints()

    for tok in sorted(set([str(t).upper().replace("$", "").strip() for t in tokens if t])):
        origin = "auto"
        chain_id = pair_addr = ""
        network = None

        if tok in hints:
            hh = hints[tok]
            chain_id = str(hh.get("chainId","") or "")
            pair_addr = str(hh.get("pairAddress","") or "")
            network = _map_chain_to_gt_network(chain_id)
            origin = "hint"

        if not pair_addr:
            cached = pcache.get(tok)
            if cached:
                chain_id = cached.get("chainId","") or chain_id
                pair_addr = cached.get("pairAddress","") or pair_addr
                network = cached.get("network", None) or network
                origin = "cache"

        if not pair_addr:
            qs = [f"{tok}/USDC", f"{tok}/USDT", tok, f"${tok}"]
            pairs = []
            for q in qs:
                pairs.extend(_dexs_search(q))
            if pairs:
                pairs = _pairs_sorted_by_liquidity(pairs)
                chain_id = str(pairs[0].get("chainId",""))
                pair_addr = str(pairs[0].get("pairAddress",""))
                network = _map_chain_to_gt_network(chain_id)
                origin = "auto_search"

        if not pair_addr:
            dbg.append({"token": tok, "status": "no_pair", "origin": origin})
            continue

        net_found, base_sym, quote_sym = gt_probe_pool_network(pair_addr, preferred=network)
        if not net_found:
            dbg.append({"token": tok, "status": "gt_404_pool", "origin": origin, "pairAddress": pair_addr, "preferred": network})
            continue

        dfp, src = gt_fetch_ohlcv_resampled_day_cached(net_found, pair_addr, tmin, tmax, ohlcv_cache=ocache, max_pages=int(max_pages))
        if dfp.empty:
            dbg.append({"token": tok, "status": "gt_empty_ohlcv", "origin": origin,
                        "pairAddress": pair_addr, "network": net_found, "gt_source": src})
            continue

        prices[tok] = dfp
        pcache.set(tok, chain_id or "", pair_addr, net_found)
        dbg.append({"token": tok, "status": "ok", "origin": origin, "pairAddress": pair_addr,
                    "network": net_found, "base": base_sym, "quote": quote_sym,
                    "gt_source": src, "n_candles": int(len(dfp))})

    return prices, dbg


def build_prices_from_contracts(manual_map: pd.DataFrame,
                                tmin: pd.Timestamp,
                                tmax: pd.Timestamp,
                                pair_cache: Optional[PairCache] = None,
                                ohlcv_cache: Optional[OhlcvCache] = None,
                                hints_path: str | Path = "cache/token_hints.json",
                                max_pages: int = 10,
                                strict_manual: bool = True) -> Tuple[Dict[str, pd.DataFrame], List[dict], set]:
    prices: Dict[str, pd.DataFrame] = {}
    dbg: List[dict] = []
    manual_tokens: set = set()
    pcache = pair_cache or PairCache()
    ocache = ohlcv_cache or OhlcvCache()

    hints = _load_user_hints(hints_path)

    for _, row in manual_map.iterrows():
        tok = str(row.get("token","")).upper().replace("$", "").strip()
        chain_id = str(row.get("chainId","")).strip().lower()
        address = str(row.get("address","")).strip()
        pair_manual = str(row.get("pairAddress","")).strip() if pd.notna(row.get("pairAddress", "")) else ""

        if not tok or not chain_id or not address:
            dbg.append({"token": tok or "(vide)", "status": "invalid_row", "reason": "Missing token/chainId/address"})
            continue
        manual_tokens.add(tok)

        preferred_net = _map_chain_to_gt_network(chain_id)

        if pair_manual:
            net_found, b, q = gt_probe_pool_network(pair_manual, preferred=preferred_net)
            if not net_found:
                dbg.append({"token": tok, "status": "gt_404_pool", "origin": "manual",
                            "pairAddress": pair_manual, "preferred": preferred_net})
                if strict_manual:
                    continue
            else:
                dfp, src = gt_fetch_ohlcv_resampled_day_cached(net_found, pair_manual, tmin, tmax,
                                                               ohlcv_cache=ocache, max_pages=int(max_pages))
                if dfp.empty:
                    dbg.append({"token": tok, "status": "gt_empty_ohlcv", "origin": "manual",
                                "pairAddress": pair_manual, "network": net_found, "gt_source": src})
                    if strict_manual:
                        continue
                else:
                    prices[tok] = dfp
                    pcache.set(tok, chain_id, pair_manual, net_found, address=address)
                    hints[tok] = {"chainId": chain_id, "address": address, "pairAddress": pair_manual}
                    _save_user_hints(hints, hints_path)
                    dbg.append({"token": tok, "status": "ok", "origin": "manual",
                                "pairAddress": pair_manual, "network": net_found,
                                "base": b, "quote": q, "gt_source": src, "n_candles": int(len(dfp))})
                    continue

        pairs = dexs_token_pairs(chain_id, address)
        pairs_sorted = _pairs_sorted_by_liquidity(pairs)
        if not pairs_sorted:
            dbg.append({"token": tok, "status": "no_pair", "origin": "dex_token_pairs", "chainId": chain_id})
            continue

        got = False
        for p in pairs_sorted:
            pair_addr = str(p.get("pairAddress", "") or "")
            if not pair_addr:
                continue
            net_found, b, q = gt_probe_pool_network(pair_addr, preferred=preferred_net)
            if not net_found:
                continue
            dfp, src = gt_fetch_ohlcv_resampled_day_cached(net_found, pair_addr, tmin, tmax,
                                                           ohlcv_cache=ocache, max_pages=int(max_pages))
            if dfp.empty:
                continue
            prices[tok] = dfp
            pcache.set(tok, chain_id, pair_addr, net_found, address=address)
            hints[tok] = {"chainId": chain_id, "address": address, "pairAddress": pair_addr}
            _save_user_hints(hints, hints_path)
            dbg.append({"token": tok, "status": "ok", "origin": "dex_token_pairs",
                        "pairAddress": pair_addr, "network": net_found,
                        "base": b, "quote": q, "gt_source": src, "n_candles": int(len(dfp))})
            got = True
            break

        if not got:
            dbg.append({"token": tok, "status": "gt_404_or_empty_all_pools",
                        "origin": "dex_token_pairs", "n_pools_tried": len(pairs_sorted)})

    return prices, dbg, manual_tokens


# ==================== ALIGN PRICES ====================

def align_trade_prices(
    signals: pd.DataFrame,
    prices_by_token: Dict[str, pd.DataFrame],
    horizon_days: Dict[str, int]
) -> pd.DataFrame:
    rows = []
    if signals is None or signals.empty:
        return pd.DataFrame(rows)
    for _, r in signals.iterrows():
        tok = str(r["token"]).upper().replace("$", "").strip()
        t0 = pd.to_datetime(r["ts_signal_utc"])
        hz = str(r["horizon"]).lower()
        H = int(horizon_days.get(hz, 0) or 0)
        if H <= 0:
            continue
        p = prices_by_token.get(tok)
        if p is None or p.empty:
            continue
        pin = p[p["ts_price_utc"] >= t0].head(1)
        if pin.empty:
            continue
        p_in = float(pin["price"].iloc[0])
        t_out = t0 + pd.Timedelta(days=H)
        pout = p[p["ts_price_utc"] >= t_out].head(1)
        if pout.empty:
            continue
        p_out = float(pout["price"].iloc[0])
        ret = p_out / p_in - 1.0
        rows.append({
            "token": tok, "ts_signal_utc": t0, "horizon": hz,
            "p_in": p_in, "p_out": p_out, "ret": ret, "win": int(ret >= 0.0)
        })
    return pd.DataFrame(rows)


# ==================== Hints I/O ====================

def _hints_path(path: str | Path = "cache/token_hints.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_user_hints(path: str | Path = "cache/token_hints.json") -> Dict[str, dict]:
    p = _hints_path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_user_hints(hints: Dict[str, dict], path: str | Path = "cache/token_hints.json"):
    p = _hints_path(path)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(hints, f, indent=2)
    except Exception:
        pass


# ==================== ALT PROVIDERS (Birdeye / CG Pro) ====================

def fetch_history_birdeye(chain: str,
                          address: str,
                          start: pd.Timestamp,
                          end: pd.Timestamp,
                          api_key: Optional[str]) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["ts_price_utc", "price"])
    url = "https://public-api.birdeye.so/defi/history_price"
    headers = {"X-API-KEY": api_key}
    params = {
        "address": address,
        "chain": chain,
        "type": "day",
        "time_from": int(start.timestamp()),
        "time_to": int(end.timestamp())
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", {})
        items = data.get("items", [])
        rows = []
        for it in items:
            ts = pd.to_datetime(it.get("unixTime", 0), unit="s", utc=True).tz_convert(None)
            price = float(it.get("value", np.nan))
            rows.append({"ts_price_utc": ts, "price": price})
        return pd.DataFrame(rows).dropna().sort_values("ts_price_utc").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["ts_price_utc", "price"])


def fetch_history_coingecko(token_id: str,
                            start: pd.Timestamp,
                            end: pd.Timestamp,
                            api_key: Optional[str]) -> pd.DataFrame:
    if not api_key or not token_id:
        return pd.DataFrame(columns=["ts_price_utc", "price"])
    url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())}
    headers = {"x-cg-pro-api-key": api_key}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        rows = []
        for ts_ms, price in prices:
            ts = pd.to_datetime(int(ts_ms) / 1000, unit="s", utc=True).tz_convert(None)
            rows.append({"ts_price_utc": ts, "price": float(price)})
        df = pd.DataFrame(rows).dropna().sort_values("ts_price_utc")
        if not df.empty:
            df = (df.set_index("ts_price_utc").resample("1D").last().dropna().reset_index())
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["ts_price_utc", "price"])


# ---------- Robust OHLCV resolver (GT -> Birdeye fallback) ----------
import time, math, requests
import pandas as pd
from typing import Optional, Tuple, List, Dict

# --- utils mapping
def _map_chain_to_gt_network(chain_id: str) -> Optional[str]:
    c = (chain_id or "").lower()
    mp = {
        "ethereum":"eth","eth":"eth","bsc":"bsc","binance-smart-chain":"bsc",
        "polygon":"polygon","matic":"polygon","arbitrum":"arbitrum","arbitrum-one":"arbitrum",
        "optimism":"optimism","base":"base","avalanche":"avax","avax":"avax",
        "fantom":"ftm","ftm":"ftm","solana":"solana","sei":"sei","blast":"blast",
        "zksync":"zksync","linea":"linea","scroll":"scroll","opbnb":"opbnb",
    }
    return mp.get(c)

def _gt_headers():
    # GT conseille ce header pour v2
    return {"accept": "application/json;version=20230302"}

def _is_valid_ts(df: pd.DataFrame) -> bool:
    return (df is not None and not df.empty and
            "ts_price_utc" in df.columns and "price" in df.columns and
            not df["ts_price_utc"].isna().all() and not df["price"].isna().all())

# --- Dexscreener : recherche et pairs (pour ticker -> contrats)
def _dexs_search(q: str) -> List[dict]:
    try:
        r = requests.get("https://api.dexscreener.com/latest/dex/search", params={"q": q}, timeout=15)
        r.raise_for_status()
        js = r.json() or {}
        return (js.get("pairs") or []) if isinstance(js, dict) else []
    except Exception:
        return []

def _dexs_token_pairs(chain_id: str, token_address: str) -> List[dict]:
    try:
        url = f"https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}"
        r = requests.get(url, timeout=20); r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _pairs_sorted_by_liq_and_vol(pairs: List[dict]) -> List[dict]:
    def _vol24(p):
        try: return float((p.get("volume", {}) or {}).get("h24") or p.get("volume24h") or 0.0)
        except Exception: return 0.0
    def _liq(p):
        try: return float(((p.get("liquidity") or {}).get("usd") or 0.0))
        except Exception: return 0.0
    # priorité volume, puis liquidité
    return sorted(pairs, key=lambda p: (_vol24(p), _liq(p)), reverse=True)

# --- GeckoTerminal : pools du token + OHLCV agressif
def gt_token_pools(network: str, token_address: str) -> List[dict]:
    # Liste les pools *connues par GT* pour ce mint
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{token_address}/pools"
    try:
        r = requests.get(url, headers=_gt_headers(), timeout=15)
        r.raise_for_status()
        js = r.json() or {}
        data = js.get("data") or []
        pools = []
        for it in data:
            attr = (it or {}).get("attributes") or {}
            pools.append({
                "pool": str((it or {}).get("id") or ""),
                "liquidity_usd": float((attr.get("reserve_in_usd") or 0) or 0),
                "volume24h_usd": float((attr.get("volume_usd") or {}).get("h24") or 0),
                "base_symbol": (attr.get("base_token") or {}).get("symbol") or "",
                "quote_symbol": (attr.get("quote_token") or {}).get("symbol") or "",
            })
        # tri: volume puis liquidité
        pools.sort(key=lambda x: (x["volume24h_usd"], x["liquidity_usd"]), reverse=True)
        return pools
    except Exception:
        return []

def _gt_fetch_page(network: str, pool: str, gran: str, aggregate: int,
                   before_ts: Optional[int], limit: int) -> pd.DataFrame:
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool}/ohlcv/{gran}"
    params = {"aggregate": int(aggregate), "limit": int(limit)}
    if before_ts is not None: params["before_timestamp"] = int(before_ts)
    try:
        r = requests.get(url, params=params, headers=_gt_headers(), timeout=20)
        r.raise_for_status()
        js = r.json() or {}
        attrs = ((js.get("data") or {}) or {}).get("attributes") or {}
        ohlcv = attrs.get("ohlcv_list") or []
        rows = []
        for row in ohlcv:
            # GT renvoie [ts, open, high, low, close, volume] (ts en ms ou s)
            ts_raw, close_raw = row[0], row[4]
            ts = pd.to_datetime(ts_raw, unit="ms", utc=True, errors="coerce")
            if pd.isna(ts): ts = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
            if pd.isna(ts): continue
            rows.append({"ts_price_utc": ts.tz_convert(None), "price": float(close_raw)})
        df = pd.DataFrame(rows)
        return (df.dropna().sort_values("ts_price_utc").reset_index(drop=True)
                if not df.empty else pd.DataFrame(columns=["ts_price_utc","price"]))
    except Exception:
        return pd.DataFrame(columns=["ts_price_utc","price"])

def gt_fetch_ohlcv_aggressive(network: str, pool: str,
                              start: pd.Timestamp, end: pd.Timestamp,
                              max_pages: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Essaie plusieurs granularités/agrégations et pagine des deux côtés.
    Retourne (df, source_label) ; df = ts/price (UTC, trié).
    """
    tried = []
    def _scan(gran: str, aggregates: List[int]) -> Optional[pd.DataFrame]:
        nonlocal tried
        parts: List[pd.DataFrame] = []
        last_before = None
        t0 = time.time()
        pages = 0
        for agg in aggregates:
            pages = 0; last_before = None
            while pages < max_pages and time.time() - t0 < 45:
                dfp = _gt_fetch_page(network, pool, gran, agg, last_before, 300)
                tried.append((gran, agg, len(dfp)))
                if dfp.empty:
                    # pas cette page → avance
                    pages += 1
                    if last_before is None:
                        # 1ère page vide → on arrête cette agg
                        break
                    continue
                parts.append(dfp)
                new_min = dfp["ts_price_utc"].min()
                last_before = int(new_min.timestamp()) - 1 if pd.notna(new_min) else None
                pages += 1
                # assez de data ?
                if dfp["ts_price_utc"].max() >= end - pd.Timedelta(days=1):
                    break
        if not parts: return None
        df = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset=["ts_price_utc"])
                .sort_values("ts_price_utc"))
        # garde la fenêtre ±2j
        df = df[(df["ts_price_utc"] >= start.floor("D") - pd.Timedelta(days=2)) &
                (df["ts_price_utc"] <= end.ceil("D")   + pd.Timedelta(days=2))].reset_index(drop=True)
        return df

    # ordre d'essai : day → hour (1,4,6) → minute (1,5,15,30), puis resample daily si besoin
    for gran, aggs in [("day",[1]), ("hour",[1,4,6]), ("minute",[1,5,15,30])]:
        df = _scan(gran, aggs)
        if _is_valid_ts(df):
            if gran != "day":
                # resample en jour (dernier prix du jour)
                df = (df.set_index("ts_price_utc").resample("1D").last().dropna().reset_index())
            return df[["ts_price_utc","price"]].copy(), f"gt_{gran}"
    return pd.DataFrame(columns=["ts_price_utc","price"]), f"gt_empty({tried})"

# --- Birdeye fallback (Solana, nécessite API KEY)
def birdeye_history_price_solana(mint: str,
                                 start: pd.Timestamp,
                                 end: pd.Timestamp,
                                 api_key: Optional[str]) -> pd.DataFrame:
    if not api_key: return pd.DataFrame(columns=["ts_price_utc","price"])
    url = "https://public-api.birdeye.so/defi/history_price"
    params = {
        "address": mint, "chain": "solana", "type": "day",
        "time_from": int(start.timestamp()), "time_to": int(end.timestamp())
    }
    try:
        r = requests.get(url, params=params, headers={"X-API-KEY": api_key}, timeout=15)
        r.raise_for_status()
        items = (r.json().get("data", {}) or {}).get("items", []) or []
        rows = [{"ts_price_utc": pd.to_datetime(it.get("unixTime",0), unit="s", utc=True).tz_convert(None),
                 "price": float(it.get("value", math.nan))} for it in items]
        df = pd.DataFrame(rows).dropna().sort_values("ts_price_utc").reset_index(drop=True)
        return df[["ts_price_utc","price"]]
    except Exception:
        return pd.DataFrame(columns=["ts_price_utc","price"])

# --- Orchestrateur principal ---
def build_price_series_any(
    ticker_or_address: str,
    chain_hint: Optional[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    birdeye_api_key: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Entrées:
      - ticker_or_address : "$BONK" ou "DezXA..." (mint) ou adresse EVM
      - chain_hint        : "solana", "base", "eth", ... (optionnel)
    Sorties:
      - df(ts_price_utc, price)
      - debug dict (sources testées + choix retenu)
    """
    dbg = {"network": None, "token": None, "gt_pools_found": 0, "sources": []}

    # 1) Résolution contrat + pools
    token_addr = None
    network = _map_chain_to_gt_network(chain_hint or "")
    if ticker_or_address.startswith("$") or len(ticker_or_address) <= 8:
        # on est probablement sur un ticker → Dexscreener search
        pairs = []
        for q in [ticker_or_address, ticker_or_address.replace("$","")+"/USDC",
                  ticker_or_address.replace("$","")+"/USDT"]:
            pairs.extend(_dexs_search(q))
        pairs = _pairs_sorted_by_liq_and_vol(pairs)
        if not pairs:
            return pd.DataFrame(columns=["ts_price_utc","price"]), {**dbg, "error": "no_pairs_from_ticker"}
        p0 = pairs[0]
        token_addr = ((p0.get("baseToken") or {}).get("address") or
                      (p0.get("quoteToken") or {}).get("address") or "")
        chain_id = str(p0.get("chainId",""))
        network = _map_chain_to_gt_network(network or chain_id)
    else:
        # on a déjà un contrat + (éventuellement) une chain
        token_addr = ticker_or_address
        if network is None and chain_hint:
            network = _map_chain_to_gt_network(chain_hint)

    if not token_addr or not network:
        return pd.DataFrame(columns=["ts_price_utc","price"]), {**dbg, "error": "cannot_resolve_token_or_network"}

    dbg["network"] = network
    dbg["token"] = token_addr

    # 2) Pools reconnues par GT (plus fiable pour /ohlcv)
    pools = gt_token_pools(network, token_addr)
    dbg["gt_pools_found"] = len(pools)

    # fallback : si aucune pool chez GT, on tente via Dexscreener (token->pairs) puis on garde les pairAddress
    if not pools:
        pairs = _dexs_token_pairs(network if network!="eth" else "ethereum", token_addr)
        pairs = _pairs_sorted_by_liq_and_vol(pairs)
        pools = [{"pool": str(p.get("pairAddress")), "liquidity_usd": float(((p.get("liquidity") or {}).get("usd") or 0.0)),
                  "volume24h_usd": float(((p.get("volume") or {}).get("h24") or 0.0)),
                  "base_symbol": (p.get("baseToken") or {}).get("symbol") or "",
                  "quote_symbol": (p.get("quoteToken") or {}).get("symbol") or ""} for p in pairs]

    # 3) Essaie GT OHLCV agressif sur les meilleurs pools
    for i, pool in enumerate(pools[:8]):  # limite 8 pools
        df_gt, src = gt_fetch_ohlcv_aggressive(network, pool["pool"], start, end, max_pages=10)
        dbg["sources"].append({"network": network, "kind": "pool", "pool": pool["pool"],
                               "n_candles": 0 if df_gt.empty else int(len(df_gt)), "source": src,
                               "token": None})
        if _is_valid_ts(df_gt):
            return df_gt, dbg

    # 4) Fallback Solana → Birdeye (prix daily par mint)
    if network == "solana":
        df_be = birdeye_history_price_solana(token_addr, start, end, birdeye_api_key)
        dbg["sources"].append({"network": network, "kind": "token", "pool": None,
                               "n_candles": 0 if df_be.empty else int(len(df_be)), "source": "birdeye_day",
                               "token": token_addr})
        if _is_valid_ts(df_be):
            return df_be, dbg

    # 5) nada
    return pd.DataFrame(columns=["ts_price_utc","price"]), dbg
