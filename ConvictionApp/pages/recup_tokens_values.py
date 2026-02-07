# Streamlit app — Dexscreener Top Volume + OHLCV (GeckoTerminal) + Global ATH/DD + Holders (EVM & Solana)
# 
# ✅ Fallback automatique de RPCs (EVM & Solana) + rotation si un provider refuse les grosses fenêtres / renvoie 401/403/404/timeout
# ✅ Clé Ankr intégrée : 82754f2f17b9e2c18246856447dbfc01492c95a0be99783f10e47e78f57bb563
# ✅ Charts en USD ou Market Cap (approx) + ATH global (indépendant du timeframe) + drawdown
# ✅ Holders EVM (reconstruction via Transfer logs) :
#    - détection automatique du *bon* token à compter (base/quote) s'il est ERC‑20 (évite WETH/native)
#    - fenêtres adaptatives + caps par provider (Llama 1k, Cloudflare 2k, Ankr 2k)
#    - réduction dynamique du lookback si 0 logs → re‑scan plus récent
#    - limite de fenêtres pour éviter les scans interminables
# ✅ Holders Solana (getProgramAccounts) :
#    - fallback RPC renforcé (skip instantané 401/403/404)
#    - liste volontairement courte (Ankr + Officiel) pour éviter 404 de PublicNode
# ✅ UI : lookback réglable, choix de métrique (USD / MCAP), barre de progression, messages explicites
#
# Dépendances :
#   pip install streamlit requests pandas plotly web3
#
# Lancer : streamlit run app.py

import json
import time
from urllib.parse import quote, urlparse
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from web3 import Web3
from collections import defaultdict

# ==============================================================================
# CONFIG
# ==============================================================================
ANKR_KEY = "82754f2f17b9e2c18246856447dbfc01492c95a0be99783f10e47e78f57bb563"

# EVM RPC fallback lists (ordre de test)
EVM_RPCS: Dict[str, List[str]] = {
    # Ethereum
    "ethereum": [
        f"https://rpc.ankr.com/eth/{ANKR_KEY}",
        "https://cloudflare-eth.com",
        "https://eth.llamarpc.com",
    ],
    # BNB Smart Chain
    "bsc": [
        f"https://rpc.ankr.com/bsc/{ANKR_KEY}",
        "https://bsc-dataseed1.binance.org",
        "https://bsc-dataseed.binance.org",
    ],
    # Polygon PoS
    "polygon": [
        f"https://rpc.ankr.com/polygon/{ANKR_KEY}",
        "https://polygon-rpc.com",
        "https://poly-rpc.gateway.pokt.network",
    ],
    # Arbitrum One
    "arbitrum": [
        f"https://rpc.ankr.com/arbitrum/{ANKR_KEY}",
        "https://arb1.arbitrum.io/rpc",
        "https://arbitrum.llamarpc.com",
    ],
    # Optimism
    "optimism": [
        f"https://rpc.ankr.com/optimism/{ANKR_KEY}",
        "https://mainnet.optimism.io",
    ],
    # Base
    "base": [
        f"https://rpc.ankr.com/base/{ANKR_KEY}",
        "https://mainnet.base.org",
    ],
    # Avalanche C-Chain
    "avalanche": [
        f"https://rpc.ankr.com/avalanche/{ANKR_KEY}",
        "https://api.avax.network/ext/bc/C/rpc",
    ],
    # Fantom
    "fantom": [
        f"https://rpc.ankr.com/fantom/{ANKR_KEY}",
        "https://rpc.ftm.tools",
    ],
    # Linea
    "linea": [
        f"https://rpc.ankr.com/linea/{ANKR_KEY}",
        "https://rpc.linea.build",
    ],
    # zkSync Era
    "zksync": [
        "https://mainnet.era.zksync.io",
    ],
    # Blast
    "blast": [
        f"https://rpc.ankr.com/blast/{ANKR_KEY}",
        "https://rpc.blast.io",
    ],
    # Mantle
    "mantle": [
        f"https://rpc.ankr.com/mantle/{ANKR_KEY}",
        "https://rpc.mantle.xyz",
    ],
    # opBNB
    "opbnb": [
        f"https://rpc.ankr.com/op_bnb/{ANKR_KEY}",
        "https://opbnb-mainnet-rpc.bnbchain.org",
    ],
}

# Limites spécifiques de fenêtre getLogs par provider (quand connues)
PROVIDER_MAX_BLOCK_RANGE: Dict[str, int] = {
    "eth.llamarpc.com": 1000,
    "cloudflare-eth.com": 2048,
    "rpc.ankr.com": 2000,
}

# Solana RPC fallback list (ordre de test) — PublicNode retiré (404 chez toi)
SOLANA_RPCS: List[str] = [
    f"https://rpc.ankr.com/solana/{ANKR_KEY}",  # Ankr (clé fournie)
    "https://api.mainnet-beta.solana.com",      # Officiel (gratuit mais RL)
]

# Dexscreener API
DS_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?q="

# Dexscreener chainId -> GeckoTerminal "network" mapping
DS_TO_GECKO_NETWORK: Dict[str, str] = {
    "ethereum": "eth",
    "bsc": "bsc",
    "polygon": "polygon_pos",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "base": "base",
    "avalanche": "avax",
    "fantom": "fantom",
    "solana": "solana",
}

GECKO_OHLCV_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool}/ohlcv/{timeframe}"

# Approx average block times (sec) for start-block estimation
AVG_BLOCK_TIME_SEC: Dict[str, float] = {
    "ethereum": 12.0,
    "bsc": 3.0,
    "polygon": 2.1,
    "arbitrum": 0.25,
    "optimism": 2.0,
    "base": 2.0,
    "avalanche": 2.0,
    "fantom": 1.0,
    "linea": 2.0,
    "zksync": 2.0,
    "blast": 2.0,
    "mantle": 2.0,
    "opbnb": 1.0,
}

# ==============================================================================
# UTILS
# ==============================================================================

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def ms_to_dt(ms: Optional[int]) -> Optional[datetime]:
    try:
        if not ms:
            return None
        return datetime.utcfromtimestamp(ms / 1000)
    except Exception:
        return None


def fmt_num(x: Optional[float], max_digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        if abs(x) >= 1_000_000_000:
            return f"{x/1_000_000_000:.{max_digits}f}B"
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.{max_digits}f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.{max_digits}f}k"
        if abs(x) < 1:
            return f"{x:.{max_digits+4}f}"
        return f"{x:.{max_digits}f}"
    except Exception:
        return str(x)


def ui_tf_to_gecko_params(ui_tf: str):
    ui_tf = ui_tf.lower().strip()
    if ui_tf in {"1m", "5m", "15m"}:
        return "minute", int(ui_tf[:-1])
    if ui_tf in {"1h", "4h", "12h"}:
        return "hour", int(ui_tf[:-1])
    if ui_tf == "1d":
        return "day", 1
    return "hour", 1

# ==============================================================================
# Dexscreener
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=30)
def query_dexscreener(query: str) -> Dict[str, Any]:
    url = DS_SEARCH_URL + quote(query, safe="")
    headers = {"Accept": "application/json"}
    t0 = time.time()
    r = requests.get(url, headers=headers, timeout=20)
    latency_ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    data["_latency_ms"] = latency_ms
    data["_endpoint"] = url
    return data


def normalize_pairs(pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame()
    rows = []
    for p in pairs:
        base = p.get("baseToken", {}) or {}
        quote = p.get("quoteToken", {}) or {}
        liq = p.get("liquidity", {}) or {}
        vol = p.get("volume", {}) or {}
        chg = p.get("priceChange", {}) or {}
        txns = p.get("txns", {}) or {}
        rows.append({
            "raw": p,
            "chain": p.get("chainId"),
            "dex": p.get("dexId"),
            "pairAddress": p.get("pairAddress"),
            "url": p.get("url"),
            "baseSymbol": base.get("symbol"),
            "baseName": base.get("name"),
            "baseAddress": base.get("address"),
            "quoteSymbol": quote.get("symbol"),
            "quoteName": quote.get("name"),
            "quoteAddress": quote.get("address"),
            "priceUsd": safe_float(p.get("priceUsd")),
            "priceNative": safe_float(p.get("priceNative")),
            "vol_m5": safe_float(vol.get("m5")),
            "vol_h1": safe_float(vol.get("h1")),
            "vol_h6": safe_float(vol.get("h6")),
            "vol_h24": safe_float(vol.get("h24")),
            "chg_m5": safe_float(chg.get("m5")),
            "chg_h1": safe_float(chg.get("h1")),
            "chg_h6": safe_float(chg.get("h6")),
            "chg_h24": safe_float(chg.get("h24")),
            "liq_usd": safe_float(liq.get("usd")),
            "liq_base": safe_float(liq.get("base")),
            "liq_quote": safe_float(liq.get("quote")),
            "fdv": safe_float(p.get("fdv")),
            "marketCap": safe_float(p.get("marketCap")),
            "pairCreatedAt": p.get("pairCreatedAt"),
            "labels": p.get("labels", []),
            "info": p.get("info", {}) or {},
            "buys_m5": (txns.get("m5") or {}).get("buys"),
            "sells_m5": (txns.get("m5") or {}).get("sells"),
            "buys_h1": (txns.get("h1") or {}).get("buys"),
            "sells_h1": (txns.get("h1") or {}).get("sells"),
            "buys_h24": (txns.get("h24") or {}).get("buys"),
            "sells_h24": (txns.get("h24") or {}).get("sells"),
        })
    df = pd.DataFrame(rows)
    if "vol_h24" in df.columns:
        df = df.sort_values(["vol_h24", "liq_usd"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return df

# ==============================================================================
# GeckoTerminal OHLCV
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=120)
def fetch_gecko_ohlcv(network: str, pool: str, timeframe_path: str, aggregate: int = 1,
                      limit: int = 1000, currency: str = "usd",
                      include_empty_intervals: bool = False, token: str = "base") -> pd.DataFrame:
    url = GECKO_OHLCV_URL.format(network=network, timeframe=timeframe_path, pool=pool)
    params = {
        "aggregate": aggregate,
        "limit": limit,
        "currency": currency,
        "include_empty_intervals": str(include_empty_intervals).lower(),
        "token": token,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json() or {}
    ohlcv = (((js.get("data") or {}).get("attributes") or {}).get("ohlcv_list") or [])
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def fetch_global_ath_usd(network: str, pool: str) -> Optional[float]:
    try:
        df_day = fetch_gecko_ohlcv(network, pool, timeframe_path="day", aggregate=1, limit=1000, currency="usd")
        if df_day.empty:
            return None
        return float(pd.to_numeric(df_day["h"], errors="coerce").max())
    except requests.RequestException:
        return None


def build_candle(fig_df: pd.DataFrame, title: str = "", y_title: str = "Prix (USD)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=fig_df["ts"], open=fig_df["o"], high=fig_df["h"], low=fig_df["l"], close=fig_df["c"],
        name="OHLCV"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date (UTC)",
        yaxis_title=y_title,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=520,
    )
    return fig

# ==============================================================================
# RPC fallback helpers
# ==============================================================================

def try_connect_evm_rpc(rpc: str) -> bool:
    try:
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        return bool(w3.is_connected())
    except Exception:
        return False


def get_working_evm_rpc(chain: str) -> Optional[str]:
    for rpc in EVM_RPCS.get(chain, []):
        if try_connect_evm_rpc(rpc):
            return rpc
    return None

# ==============================================================================
# Holders – EVM (Transfer logs replay)
# ==============================================================================
TRANSFER_TOPIC = Web3.keccak(text="Transfer(address,address,uint256)").hex()
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
NATIVE_SENTINELS = {"0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", ZERO_ADDR.lower()}


def is_valid_evm_address(addr: str) -> bool:
    try:
        return Web3.is_address(addr)
    except Exception:
        return False


def is_erc20_contract(w3: Web3, addr: str) -> bool:
    try:
        code = w3.eth.get_code(Web3.to_checksum_address(addr))
        return code is not None and len(code) > 0
    except Exception:
        return False


def pick_token_to_count(w3: Web3, base_addr: str, quote_addr: str) -> Optional[str]:
    """Choisit automatiquement le token ERC‑20 à compter (évite WETH/native)."""
    cand = []
    for a in [base_addr, quote_addr]:
        if not a:
            continue
        al = a.lower()
        if al in NATIVE_SENTINELS:
            continue
        if not is_valid_evm_address(a):
            continue
        if is_erc20_contract(w3, a):
            cand.append(a)
    # priorité: base si ERC‑20, sinon quote
    if len(cand) == 0:
        return None
    if Web3.to_checksum_address(cand[0]) == Web3.to_checksum_address(base_addr) or len(cand) == 1:
        return cand[0]
    return cand[0]


def estimate_start_block(w3: Web3, chain: str, pair_created_at_ms: Optional[int], lookback_days: int) -> int:
    latest = w3.eth.block_number
    avg = AVG_BLOCK_TIME_SEC.get(chain, 2.0)
    if not pair_created_at_ms:
        blocks = int((lookback_days * 86400) / max(0.5, avg))
        return max(0, latest - blocks)
    created_dt = ms_to_dt(pair_created_at_ms)
    if not created_dt:
        return max(0, latest - int((lookback_days * 86400) / max(0.5, avg)))
    now_ts = datetime.now(timezone.utc).timestamp()
    created_ts = created_dt.replace(tzinfo=timezone.utc).timestamp()
    delta_sec = max(0, now_ts - created_ts) + lookback_days * 86400
    back_blocks = int(delta_sec / max(0.5, avg))
    return max(0, latest - back_blocks)


def _provider_max_step(rpc_url: str, default_step: int) -> int:
    host = urlparse(rpc_url).netloc
    for key, lim in PROVIDER_MAX_BLOCK_RANGE.items():
        if key in host:
            return min(default_step, lim)
    return default_step


def _scan_evm_holders_single_rpc(
    rpc_url: str,
    token_address: str,
    start_block: int,
    init_step: int = 1500,
    min_step: int = 64,
    max_windows: int = 6000,
    progress=None,
) -> Dict[str, Any]:
    """Scan sur UN provider avec fenêtres adaptatives + caps + coupe-circuit.
    - Réduction si -32062/-32005 ou messages "too large", "max is 1k blocks"
    - Cap initial par provider (Llama 1k, Cloudflare 2k, Ankr 2k)
    - Arrêt après `max_windows` fenêtres
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise RuntimeError("RPC EVM non joignable")

    token = Web3.to_checksum_address(token_address)
    latest = w3.eth.block_number

    balances = defaultdict(int)
    transfers = 0
    cur = max(0, int(start_block))
    step = _provider_max_step(rpc_url, max(min_step, int(init_step)))

    MAX_STEP_CAP = _provider_max_step(rpc_url, 4000)
    windows = 0
    empty_windows = 0

    while cur <= latest and windows < max_windows:
        to_blk = min(cur + step, latest)
        try:
            logs = w3.eth.get_logs({
                "fromBlock": cur,
                "toBlock": to_blk,
                "address": token,
                "topics": [TRANSFER_TOPIC],
            })
        except ValueError as e:
            msg = str(e)
            code = None
            try:
                err = e.args[0]
                if isinstance(err, dict):
                    code = err.get("code")
                    msg = err.get("message", msg)
            except Exception:
                pass
            lower = msg.lower()
            if "max is 1k" in lower or "max is 1000" in lower:
                if step > 1000:
                    step = 1000
                    time.sleep(0.05)
                    continue
                step = max(min_step, step // 2)
                time.sleep(0.05)
                continue
            if (code in (-32062, -32005)) or ("too large" in lower) or ("response size" in lower) or ("range" in lower and "large" in lower) or ("log" in lower and "limit" in lower):
                step = max(min_step, step // 2)
                time.sleep(0.05)
                continue
            raise
        except requests.RequestException:
            time.sleep(0.15)
            continue

        if not logs:
            empty_windows += 1
        else:
            empty_windows = 0

        for log in logs:
            topics = log.get("topics", [])
            if len(topics) < 3:
                continue
            from_addr = "0x" + topics[1].hex()[-40:]
            to_addr = "0x" + topics[2].hex()[-40:]
            try:
                value = int(log.get("data"), 16)
            except Exception:
                continue
            if from_addr.lower() != ZERO_ADDR.lower():
                balances[from_addr] -= value
            if to_addr.lower() != ZERO_ADDR.lower():
                balances[to_addr] += value
            transfers += 1

        cur = to_blk + 1
        windows += 1

        # si aucune donnée depuis 12 fenêtres d'affilée, on réduit la fenêtre
        if empty_windows >= 12 and step > min_step:
            step = max(min_step, step // 2)

        # si ça passe bien, on augmente progressivement (cap provider)
        if step < MAX_STEP_CAP and empty_windows == 0:
            step = min(MAX_STEP_CAP, int(step * 1.5))

        if progress is not None:
            progress.progress(min(1.0, (cur - start_block) / max(1, latest - start_block)))

    holders = sum(1 for v in balances.values() if v > 0)
    return {
        "holders": holders,
        "scanned_from": int(start_block),
        "scanned_to": latest,
        "transfers": transfers,
        "unique_addresses": len(balances),
        "windows": windows,
    }


def evm_count_holders_with_fallback(
    chain: str,
    token_address: str,
    start_block: int,
    max_windows: int,
    progress=None,
) -> Tuple[Dict[str, Any], str]:
    last_err = None
    for rpc in EVM_RPCS.get(chain, []):
        if not try_connect_evm_rpc(rpc):
            last_err = f"RPC down: {rpc}"
            continue
        try:
            res = _scan_evm_holders_single_rpc(
                rpc,
                token_address,
                start_block,
                init_step=1500,
                min_step=64,
                max_windows=max_windows,
                progress=progress,
            )
            return res, rpc
        except ValueError as e:
            last_err = f"ValueError {e} sur {rpc}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} sur {rpc}"
            continue
    raise RuntimeError(last_err or "Aucun RPC EVM n'a permis le scan de holders")

# ==============================================================================
# Holders – Solana (getProgramAccounts)
# ==============================================================================
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def sol_rpc_call(rpc_url: str, method: str, params: Any) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(rpc_url, json=payload, timeout=90)
    r.raise_for_status()
    js = r.json()
    if "error" in js:
        raise RuntimeError(str(js["error"]))
    return js["result"]


@st.cache_data(show_spinner=True, ttl=600)
def sol_count_holders(rpc_url: str, mint: str) -> Dict[str, Any]:
    params = [
        TOKEN_PROGRAM_ID,
        {
            "encoding": "jsonParsed",
            "filters": [
                {"dataSize": 165},
                {"memcmp": {"offset": 0, "bytes": mint}},
            ],
        },
    ]
    res = sol_rpc_call(rpc_url, "getProgramAccounts", params)
    non_zero_accounts = 0
    unique_owners = set()
    for acc in res:
        info = acc["account"]["data"]["parsed"]["info"]
        amount = int(info["tokenAmount"]["amount"])  # base units
        if amount > 0 and info.get("state") != "frozen":
            non_zero_accounts += 1
            owner = info.get("owner")
            if owner:
                unique_owners.add(owner)
    return {
        "token_accounts_non_zero": non_zero_accounts,
        "unique_owners": len(unique_owners),
        "accounts_scanned": len(res),
    }


def sol_count_holders_with_fallback(mint: str) -> Dict[str, Any]:
    last_err = None
    for rpc in SOLANA_RPCS:
        try:
            probe = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "getSlot"}, timeout=20)
            if probe.status_code in (401, 403, 404):
                last_err = f"HTTP {probe.status_code} sur {rpc}"
                continue
            if probe.status_code != 200:
                last_err = f"HTTP {probe.status_code} sur {rpc}"
                continue
            res = sol_count_holders(rpc, mint)
            res["rpc_used"] = rpc
            return res
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            last_err = f"HTTP {status} sur {rpc}"
            time.sleep(0.2)
            continue
        except requests.Timeout:
            last_err = f"Timeout sur {rpc}"
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} sur {rpc}"
            time.sleep(0.2)
            continue
    raise RuntimeError(last_err or "Aucun RPC Solana valide")

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Dexscreener + OHLCV + Holders", page_icon="📈", layout="wide")

# Session state init
if "last_query" not in st.session_state:
    st.session_state.last_query = "SOL/USDC"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "top_pair" not in st.session_state:
    st.session_state.top_pair = None
if "meta" not in st.session_state:
    st.session_state.meta = {"endpoint": "", "latency_ms": 0}

st.title("📈 Dexscreener — Top vol 24h + Bougies OHLCV & Holders (Fallback RPC)")
st.caption("Chart en **USD** ou **Market Cap (approx)**, drawdown basé sur **ATH global**. Holders EVM & Solana via RPC gratuits avec fallback automatique.")

# Sidebar controls
with st.sidebar:
    st.header("Recherche")
    qry = st.text_input("Requête", value=st.session_state.last_query, help="Ex: SOL/USDC, 0x..., adresse de pair")
    search_btn = st.button("Rechercher", use_container_width=True)

    st.markdown("---")
    st.subheader("Chart")
    ui_tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "12h", "1d"], index=3)
    limit = st.slider("Nombre de bougies", min_value=100, max_value=1000, value=800, step=50)
    chart_metric = st.radio(
        "Métrique de chart",
        options=["Prix (USD)", "Market Cap (approx)"],
        index=0,
        help="Market Cap (approx) = prix × (marketCap / priceUsd) courant ; approximation si l'offre varie.",
    )

    st.markdown("---")
    st.subheader("Holders (EVM)")
    lookback_days = st.slider("Lookback (jours)", 1, 120, 30, help="Réduit fortement la plage scannée si besoin de vitesse.")
    max_windows = st.slider("Fenêtres max", 500, 20000, 6000, step=500, help="Stoppe le scan EVM au-delà de N fenêtres pour éviter les scans interminables.")

# Trigger search
if search_btn or st.session_state.last_result is None:
    data = query_dexscreener(qry)
    st.session_state.last_query = qry
    st.session_state.last_result = data
    df = normalize_pairs(data.get("pairs") or [])
    st.session_state.top_pair = df.iloc[0].to_dict() if not df.empty else None
    st.session_state.meta = {
        "endpoint": data.get("_endpoint", ""),
        "latency_ms": data.get("_latency_ms", 0),
        "schema": data.get("schemaVersion"),
    }

meta = st.session_state.meta
top = st.session_state.top_pair

meta_col1, meta_col2, meta_col3 = st.columns([2, 1, 1])
with meta_col1:
    st.caption("Endpoint")
    st.code(meta.get("endpoint", ""), language="text")
with meta_col2:
    st.caption("Latence")
    st.metric("Temps de réponse", f"{meta.get('latency_ms', 0)} ms")
with meta_col3:
    st.caption("Schéma")
    st.write(meta.get("schema") or "—")

if not top:
    st.warning("Aucune paire trouvée pour cette recherche.")
    st.stop()

# Header
info = top.get("info") or {}
image_url = info.get("imageUrl")
header_left, header_right = st.columns([3, 1])
with header_left:
    st.subheader(f"{top.get('baseSymbol')}/{top.get('quoteSymbol')} • {top.get('chain')} • {top.get('dex')}")
    if top.get("labels"):
        st.caption("Labels : " + ", ".join(top["labels"]))
with header_right:
    if isinstance(image_url, str) and image_url:
        st.image(image_url, caption=top.get("baseName") or "", use_container_width=True)

# Main metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Prix (USD)", f"${fmt_num(top.get('priceUsd'), 6)}")
m2.metric("Δ 24h", f"{fmt_num(top.get('chg_h24'))}%")
m3.metric("Vol 24h", f"${fmt_num(top.get('vol_h24'))}")
m4.metric("Liquidité", f"${fmt_num(top.get('liq_usd'))}")
m5.metric("FDV", f"${fmt_num(top.get('fdv'))}")
m6.metric("Market Cap", f"${fmt_num(top.get('marketCap'))}")

# ==============================================================================
# Chart + Global ATH/DD
# ==============================================================================
chain = (top.get("chain") or "").lower()
pair_address = top.get("pairAddress")
gecko_net = DS_TO_GECKO_NETWORK.get(chain)

st.subheader(f"Chart OHLCV — {chart_metric}")
if not gecko_net:
    st.warning(f"Réseau `{chain}` non mappé vers GeckoTerminal — ajoutez-le si besoin.")
    ohlcv_df = pd.DataFrame()
    global_ath = None
else:
    global_ath = fetch_global_ath_usd(gecko_net, pair_address)
    tf_path, agg = ui_tf_to_gecko_params(ui_tf)
    try:
        with st.spinner(f"Chargement OHLCV… ({tf_path}, agg {agg})"):
            ohlcv_df = fetch_gecko_ohlcv(gecko_net, pair_address, timeframe_path=tf_path, aggregate=agg, limit=limit, currency="usd", token="base")
    except requests.HTTPError as e:
        st.error(f"Erreur OHLCV GeckoTerminal (HTTP {e.response_status if hasattr(e, 'response_status') else '?'} )")
        ohlcv_df = pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"Erreur réseau OHLCV: {e}")
        ohlcv_df = pd.DataFrame()

if not ohlcv_df.empty:
    df_plot = ohlcv_df.copy()
    # Market Cap approx
    y_title = "Prix (USD)"
    if chart_metric == "Market Cap (approx)":
        price_now = top.get("priceUsd")
        mcap_now = top.get("marketCap")
        if isinstance(price_now, (int, float)) and price_now > 0 and isinstance(mcap_now, (int, float)) and mcap_now > 0:
            supply_est = mcap_now / price_now
            for col in ["o", "h", "l", "c"]:
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce") * supply_est
            y_title = "Market Cap (USD, approx)"
        else:
            st.info("Market Cap (approx) indisponible (marketCap ou priceUsd manquant). Chart en USD utilisée.")

    fig = build_candle(df_plot, title=f"{top.get('baseSymbol')}/{top.get('quoteSymbol')} — {ui_tf}", y_title=y_title)
    st.plotly_chart(fig, use_container_width=True)

    cur_price = top.get("priceUsd")
    dd_pct = None
    if isinstance(cur_price, (int, float)) and isinstance(global_ath, (int, float)) and global_ath and global_ath > 0:
        dd_pct = (cur_price / global_ath - 1.0) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric("ATH global (USD)", f"${fmt_num(global_ath, 6) if global_ath else '—'}")
    c2.metric("Prix actuel", f"${fmt_num(cur_price, 6) if cur_price is not None else '—'}")
    c3.metric("Drawdown depuis ATH", f"{fmt_num(dd_pct, 4)}%" if dd_pct is not None else "—")
else:
    st.info("Pas de données OHLCV disponibles pour cette pool / ce timeframe.")
    pair_url = top.get("url")
    if isinstance(pair_url, str) and pair_url:
        st.components.v1.html(
            f"""
            <div style=\"position:relative;padding-bottom:62%;height:0;overflow:hidden;border-radius:12px;\">
              <iframe
                src=\"{pair_url}\"
                style=\"position:absolute;top:0;left:0;width:100%;height:100%;border:0;\"
                loading=\"lazy\"
                referrerpolicy=\"no-referrer\"
                sandbox=\"allow-same-origin allow-scripts allow-popups allow-forms\"
              ></iframe>
            </div>
            """,
            height=680,
            scrolling=True,
        )
        st.caption("⚠️ Fallback iframe Dexscreener si l’OHLCV n’est pas dispo sur GeckoTerminal.")

# ==============================================================================
# Détails + HOLDERS
# ==============================================================================
left, right = st.columns([2, 1])

with left:
    st.markdown("### Détails de la paire")
    st.markdown(
        f"**Pair:** `{top.get('baseSymbol')}/{top.get('quoteSymbol')}`  \n"
        f"**Réseau:** `{top.get('chain')}`  \n"
        f"**DEX:** `{top.get('dex')}`  \n"
        f"**Adresse de paire:** `{top.get('pairAddress')}`  \n"
        f"**URL Dexscreener:** {('[ouvrir]('+top.get('url')+')' if top.get('url') else '—')}  \n"
        f"**Créée le (UTC):** `{ms_to_dt(top.get('pairCreatedAt'))}`"
    )

    st.markdown("### Tokens")
    st.markdown(
        f"**Base:** `{top.get('baseName')}` (`{top.get('baseSymbol')}`)  \n"
        f"**Adresse base:** `{top.get('baseAddress')}`  \n"
        f"**Quote:** `{top.get('quoteName')}` (`{top.get('quoteSymbol')}`)  \n"
        f"**Adresse quote:** `{top.get('quoteAddress')}`"
    )

    st.markdown("### Prix & variations")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Δ 5 min", f"{fmt_num(top.get('chg_m5'))}%")
    col_p2.metric("Δ 1 h", f"{fmt_num(top.get('chg_h1'))}%")
    col_p3.metric("Δ 6 h", f"{fmt_num(top.get('chg_h6'))}%")
    col_p4.metric("Δ 24 h", f"{fmt_num(top.get('chg_h24'))}%")

    st.markdown("### Volumes")
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric("Vol 5 min", f"${fmt_num(top.get('vol_m5'))}")
    col_v2.metric("Vol 1 h", f"${fmt_num(top.get('vol_h1'))}")
    col_v3.metric("Vol 6 h", f"${fmt_num(top.get('vol_h6'))}")
    col_v4.metric("Vol 24 h", f"${fmt_num(top.get('vol_h24'))}")

    st.markdown("### Transactions")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.write(f"**5 min**  \n🟢 {top.get('buys_m5') or 0} / 🔴 {top.get('sells_m5') or 0}")
    col_t2.write(f"**1 h**  \n🟢 {top.get('buys_h1') or 0} / 🔴 {top.get('sells_h1') or 0}")
    col_t3.write(f"**24 h**  \n🟢 {top.get('buys_h24') or 0} / 🔴 {top.get('sells_h24') or 0}")

with right:
    st.markdown("### Liquidité")
    st.write(f"**USD:** ${fmt_num(top.get('liq_usd'))}")
    st.write(f"**Base ({top.get('baseSymbol')}):** {fmt_num(top.get('liq_base'))}")
    st.write(f"**Quote ({top.get('quoteSymbol')}):** {fmt_num(top.get('liq_quote'))}")

    st.markdown("### Liens & Socials")
    websites = (info.get("websites") or [])
    socials = (info.get("socials") or [])
    if websites:
        st.write("**Sites officiels**")
        for w in websites:
            url = (w or {}).get("url")
            if url:
                st.write(f"- {url}")
    if socials:
        st.write("**Réseaux sociaux**")
        for s in socials:
            platform = (s or {}).get("platform")
            handle = (s or {}).get("handle")
            if platform or handle:
                st.write(f"- {platform or 'social'}: {handle or '—'}")
    if not websites and not socials:
        st.write("—")

# ==============================================================================
# HOLDERS (automatique, fallback RPC)
# ==============================================================================
st.markdown("## 🧮 Holders (RPC fallback automatique)")
chain_id = (top.get("chain") or "").lower()
base_addr = (top.get("baseAddress") or "").strip()
quote_addr = (top.get("quoteAddress") or "").strip()

if chain_id == "solana":
    try:
        with st.spinner("Comptage des holders Solana (fallback RPC)…"):
            res = sol_count_holders_with_fallback(base_addr)
        c1, c2, c3 = st.columns(3)
        c1.metric("Token accounts > 0", f"{res['token_accounts_non_zero']:,}")
        c2.metric("Owners uniques (≈ holders)", f"{res['unique_owners']:,}")
        c3.metric("Comptes scannés", f"{res['accounts_scanned']:,}")
        st.caption(f"RPC utilisé : {res.get('rpc_used')}")
    except Exception as e:
        st.error(f"Erreur Solana holders: {e}")
        st.info("Tous les RPC publics testés ont échoué (401/403/404/Timeout). Ajoute un endpoint privé si possible (Helius/QuickNode/Chainstack).")
else:
    try:
        rpc_probe = get_working_evm_rpc(chain_id)
        if not rpc_probe:
            raise RuntimeError(f"Aucun RPC EVM valide trouvé pour `{chain_id}`")
        w3_probe = Web3(Web3.HTTPProvider(rpc_probe, request_kwargs={"timeout": 20}))

        token_to_count = pick_token_to_count(w3_probe, base_addr, quote_addr)
        if not token_to_count:
            st.warning("Aucun contrat ERC‑20 valide détecté parmi base/quote (souvent base=WETH/native). Comptage des holders ignoré.")
        else:
            # 1ère tentative avec le lookback demandé
            start_blk = estimate_start_block(
                w3_probe,
                chain_id,
                top.get("pairCreatedAt"),
                lookback_days=lookback_days,
            )
            prog = st.progress(0.0)
            with st.spinner("Scan des Transfer logs EVM… (fenêtres adaptatives + rotation de provider)"):
                try:
                    res, used_rpc = evm_count_holders_with_fallback(chain_id, token_to_count, start_blk, max_windows=max_windows, progress=prog)
                except Exception as primary_err:
                    # Si aucun log → tenter un re‑scan recent (coupe-circuit)
                    try:
                        latest = Web3(Web3.HTTPProvider(rpc_probe)).eth.block_number
                        recent_start = max(0, latest - 200_000)  # ~ 200k blocs récents
                        res, used_rpc = evm_count_holders_with_fallback(chain_id, token_to_count, recent_start, max_windows=max_windows//2, progress=prog)
                        st.info("Premier scan sans résultat → nouvelle tentative sur ~200k blocs récents.")
                    except Exception:
                        raise primary_err

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Holders (solde > 0)", f"{res['holders']:,}")
            c2.metric("Adresses uniques vues", f"{res['unique_addresses']:,}")
            c3.metric("Transfers scannés", f"{res['transfers']:,}")
            c4.metric("Fenêtres", f"{res.get('windows', 0):,}")
            c5.metric("Plage scannée", f"{res['scanned_from']:,} → {res['scanned_to']:,}")
            st.caption(f"RPC utilisé : {used_rpc}\n\nToken compté : {token_to_count}")
            if res['transfers'] == 0:
                st.info("Zéro transfert observé. Augmente le lookback (jours) ou vérifie que le token choisi est bien l'ERC‑20 recherché (base/quote).")
    except Exception as e:
        st.error(f"Erreur EVM holders: {e}")
        st.info("Astuce: réduis le lookback pour accélérer ou augmente-le si 0 transferts; sinon, fournis un RPC privé plus permissif.")

# ==============================================================================
# JSON brut
# ==============================================================================
with st.expander("JSON brut de la paire (top volume 24h)"):
    st.code(json.dumps(top.get("raw", {}), indent=2), language="json")

# Rappels
st.info("Dexscreener: ~300 req/min • GeckoTerminal OHLCV: path=day|hour|minute + aggregate • Holders: EVM via Transfer logs (fenêtres adaptatives + caps + coupe-circuit), Solana via getProgramAccounts (fallback Ankr→Officiel).\nConseil: si la base est WETH/native, le compteur choisira automatiquement l'autre côté (quote) s'il est ERC‑20.")