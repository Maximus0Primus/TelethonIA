# -*- coding: utf-8 -*-
import json, re, math, hashlib, os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from functools import lru_cache
from summarizer_deepseek import summarize_with_deepseek as deepseek_summary



import pandas as pd
import numpy as np
from dateutil import parser as dateparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch  # pour no_grad() lors du résumé DistilBART
import streamlit as st

# optionnel en haut du fichier
import unicodedata


# =========================
# Regex / lexique / alias
# =========================
# Tokens au format $TICKER (1..15 chars alpha-num, commence par lettre)
TOKEN_REGEX = re.compile(r"(?<![A-Z0-9_])\$[A-Z][A-Z0-9]{1,14}\b")

# Lexique crypto (ajustable) – valeurs dans [-1,1]
CRYPTO_LEXICON: Dict[str, float] = {
    # positifs
    "ath": 0.60, "new ath": 0.65, "listing": 0.55, "binance": 0.50, "kraken": 0.45,
    "coinbase": 0.45, "partnership": 0.40, "audit": 0.35, "audit passed": 0.45,
    "locked lp": 0.30, "lock lp": 0.30, "buyback": 0.40, "burn": 0.35,
    "whale buy": 0.40, "roadmap delivered": 0.40, "delivered": 0.25,
    "grinding community": 0.30, "dev active": 0.30, "parabolic": 0.40,
    "pump": 0.30, "pumping": 0.35, "flippen": 0.35, "diamond hands": 0.35,
    "btfd": 0.30, "buy the dip": 0.30,
    "long term hold": 0.60, "long-term hold": 0.60, "lth": 0.55, "blue chip": 0.50,
    "dyo": 0.10, "dyor": 0.10,
    # négatifs
    "rug": -0.80, "rugged": -0.80, "scam": -0.75, "honeypot": -0.80, "exploit": -0.70,
    "dev sold": -0.50, "team sold": -0.50, "lp unlock": -0.45, "unlock": -0.35,
    "dump": -0.55, "presale dump": -0.60, "vc unlock": -0.45, "dumping": -0.50,
    "high tax": -0.35, "no utility": -0.35, "dead chat": -0.40, "team silent": -0.40,
    "bearish": -0.50, "bagholder": -0.30, "bags are heavy": -0.40,
}
# Prioriser les expressions multi-mots lors du matching
PHRASES_PRIORITY = sorted([k for k in CRYPTO_LEXICON if " " in k], key=len, reverse=True)

NEGATORS = {"not", "no", "isn't", "isnt", "ain't", "aint", "never", "without"}

# Sets pour flags rapides (affichage mot-clés)
POS_BOOST = {"ath","listing","kraken","binance","audit","locked lp","lock lp","buyback","burn",
             "partnership","new ath","delivered","dev active","parabolic","pump","pumping",
             "long term hold","long-term hold","blue chip","lth"}
NEG_BOOST = {"rug","scam","honeypot","dev sold","lp unlock","unlock","dump","presale dump",
             "vc unlock","rugged","exploit","dumping","team silent","dead chat"}

# Blacklist d'alias sans $ (majuscule)
ALIAS_BLACKLIST_UPPER = {
    "CRYPTO","COIN","TOKEN","TOKENS","MEME","MEMECOIN","MEMECOINS",
    "MARKET","WALLET","DEX","CEX","EXCHANGE","BRIDGE","CHAIN","BLOCK","BLOCKCHAIN",
    "PUMP","MOON","BULL","BEAR","LP","SWAP","BUY","SELL","TRADE","TRADING",
    "MORE",  # garder $MORE mais éviter l’alias sauvage sur "more"
}
# Contexte crypto minimal pour activer la recherche d'alias sans $
CRYPTO_CONTEXT_WORDS = {
    "chart","charts","dexscreener","gecko","cmc","coingecko","marketcap","mc",
    "contract","ca","0x","launch","listing","listed","presale","fairlaunch",
    "token","coin","lp","liquidity","swap","buy","sell","pump","dump","whale",
    "airdop","airdrop","tg","bot","utility","bridge","stake","farm"
}

def _strip_vs(s: str) -> str:
    # enlève les 'variation selectors' pour matcher les emojis avec/without VS-16
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _has_negator(window_tokens: List[str]) -> bool:
    return any(w in NEGATORS for w in window_tokens)

def rule_adjust(text: str, s: float, weight: float = 1.0) -> Tuple[float,int,int]:
    """
    Ajuste s ([-1,1]) via motifs lexicaux positifs/négatifs.
    """
    t = (text or "").lower()
    pos_hits = sum(1 for k in POS_BOOST if k in t)
    neg_hits = sum(1 for k in NEG_BOOST if k in t)
    adj = weight * (0.05*pos_hits - 0.07*neg_hits)
    s2 = max(-1.0, min(1.0, s + adj))
    return s2, pos_hits, neg_hits

# =========================
# Parsing & tokens
# =========================
def _extract_tokens_from_text(text: str) -> List[str]:
    if not text: return []
    toks = [m.group(0)[1:] for m in TOKEN_REGEX.finditer(text.upper())]
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def parse_messages_json(js: Any) -> pd.DataFrame:
    """
    Accepte 2 formats:
      1) dict { "groupA": [ {id,date,text,remark?,conviction?}, ... ], "groupB": [...] }
      2) list [ {group,id,date,text,remark?,conviction?}, ... ]
    Retourne DataFrame colonnes: id,date,group,conviction,text,remark,tokens
    """
    rows = []

    def _push(item: Dict[str, Any], fallback_group: Optional[str] = None):
        try:
            gid   = item.get("id")
            dt    = item.get("date")
            text  = item.get("text") or ""
            remark = item.get("remark", "") or item.get("remarque", "") or item.get("remarques", "")
            conv  = item.get("conviction", None)
            grp   = item.get("group") or fallback_group or "unknown"
            d     = dateparser.parse(dt) if isinstance(dt, str) else dt
            tokens = _extract_tokens_from_text(text)
            rows.append({
                "id": gid,
                "date": pd.to_datetime(d, errors="coerce", utc=True).tz_convert(None),
                "group": grp,
                "conviction": conv,
                "text": text,
                "remark": remark,
                "tokens": tokens,
            })
        except Exception:
            pass

    if isinstance(js, dict):
        for g, arr in js.items():
            if isinstance(arr, list):
                for it in arr:
                    _push(it, fallback_group=g)
    elif isinstance(js, list):
        for it in js:
            if isinstance(it, dict):
                _push(it, fallback_group=it.get("group"))
    else:
        # format inattendu
        return pd.DataFrame(columns=["id","date","group","conviction","text","remark","tokens"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id","date","group","conviction","text","remark","tokens"])
    df["conviction"] = pd.to_numeric(df["conviction"], errors="coerce").fillna(7).clip(0, 10)
    df["remark"] = df["remark"].fillna("")
    return df


def _dedupe_key_row(r: pd.Series) -> str:
    """
    Clé de déduplication robuste:
      - priorité (group|id) si id disponible
      - sinon (group|date_minute|head80(text))
    """
    grp = str(r.get("group", "")).strip().upper()
    mid = r.get("id")
    if pd.notna(mid):
        return f"{grp}|{mid}"
    dt = r.get("date")
    if isinstance(dt, pd.Timestamp) and not pd.isna(dt):
        dt_min = dt.floor("min").isoformat()
    else:
        dt_min = str(dt)
    txt = (r.get("text") or "").strip().lower()
    head = txt[:80]
    return f"{grp}|{dt_min}|{head}"


def deduplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les doublons inter-exports.
    Garde la dernière occurrence (par date si possible).
    """
    if df.empty:
        return df
    tmp = df.copy()
    # tri pour garder les plus "récentes"
    if "date" in tmp.columns:
        tmp = tmp.sort_values("date")
    tmp["__key__"] = tmp.apply(_dedupe_key_row, axis=1)
    tmp = tmp.drop_duplicates(subset="__key__", keep="last").drop(columns="__key__", errors="ignore")
    # nettoyage types
    tmp["conviction"] = pd.to_numeric(tmp["conviction"], errors="coerce").fillna(7).clip(0, 10)
    if "tokens" not in tmp.columns:
        tmp["tokens"] = [[] for _ in range(len(tmp))]
    return tmp


def load_many_jsons(json_objs: List[Any], base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Parse plusieurs JSON (formats supportés par parse_messages_json), concatène,
    fusionne avec un éventuel 'base' déjà en session, puis déduplique.
    """
    parts = []
    if base is not None and not base.empty:
        parts.append(base)
    for obj in json_objs:
        try:
            parts.append(parse_messages_json(obj))
        except Exception:
            pass
    if not parts:
        return pd.DataFrame(columns=["id","date","group","conviction","text","remark","tokens"])
    merged = pd.concat(parts, ignore_index=True)
    return deduplicate_messages(merged)


def _dollar_token_counts(df: pd.DataFrame) -> Dict[str, int]:
    cnt = defaultdict(int)
    for txt in df["text"].astype(str).tolist():
        for m in TOKEN_REGEX.finditer(txt.upper()):
            cnt[m.group(0)[1:]] += 1
    return dict(cnt)

def _apply_alias_no_dollar_strict(df: pd.DataFrame, min_len: int = 3, min_occ_with_dollar: int = 2) -> pd.DataFrame:
    """
    Si un token $T est suffisamment vu avec $, autoriser "T" nu en alias seulement
    si le message contient un contexte crypto. Filtrage strict + blacklist.
    """
    if df.empty: return df
    base_counts = _dollar_token_counts(df)
    allow = {tok for tok,c in base_counts.items()
             if c >= min_occ_with_dollar and tok not in ALIAS_BLACKLIST_UPPER and len(tok) >= min_len}
    if not allow: return df
    patterns = {tok: re.compile(rf"\b{re.escape(tok.lower())}\b") for tok in allow}
    new_tokens_col = []
    for _, r in df.iterrows():
        txt = f"{r.get('text','')} {r.get('remark','')}".lower()
        cur = set(r.get("tokens") or [])
        # activer seulement si on voit du vocabulaire crypto dans le message
        if not any(w in txt for w in CRYPTO_CONTEXT_WORDS):
            new_tokens_col.append(sorted(cur)); continue
        for tok, pat in patterns.items():
            if tok in cur: continue
            if pat.search(txt):
                cur.add(tok)
        new_tokens_col.append(sorted(cur))
    out = df.copy()
    out["tokens"] = new_tokens_col
    return out

def ensure_session_dataset():
    """Initialise les clés de session partagées pour toutes les pages (une seule fois)."""
    ss = st.session_state
    ss.setdefault("RAW_ALL", pd.DataFrame())
    ss.setdefault("RAW_DF", pd.DataFrame())          # si tu caches df enrichi
    ss.setdefault("_files_merged_count", 0)
    ss.setdefault("_dataset_version", 0)             # incrément pour cache éventuel

def _dataset_basic_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, groups=0, tokens=0, start=None, end=None)
    toks = set()
    if "tokens" in df.columns:
        for lst in df["tokens"]:
            if isinstance(lst, list):
                toks.update(lst)
    return dict(
        n=len(df),
        groups=(0 if "group" not in df.columns else int(df["group"].nunique())),
        tokens=len(toks),
        start=(None if "date" not in df.columns or df["date"].isna().all() else pd.to_datetime(df["date"]).min()),
        end=(None if "date" not in df.columns or df["date"].isna().all() else pd.to_datetime(df["date"]).max()),
    )

def ui_dataset_loader(page_key: str = "default") -> pd.DataFrame:
    """
    Rendu standardisé dans la sidebar :
    - Uploader multi-JSON (fusion + dédup)
    - Bouton reset
    - Compteur "N fichiers fusionnés, M messages uniques"
    Retourne RAW_ALL mis à jour (et stocké dans st.session_state).
    """
    ensure_session_dataset()
    with st.sidebar.expander("Sources & chargement", expanded=True):
        uploads = st.file_uploader(
            "Importer un ou plusieurs JSON",
            type=["json"],
            accept_multiple_files=True,
            key=f"uploader_{page_key}",
            help="Tu peux fusionner plusieurs exports; déduplication automatique."
        )

        def _reset_dataset():
            st.session_state["RAW_ALL"] = pd.DataFrame()
            st.session_state["RAW_DF"] = pd.DataFrame()
            st.session_state["_files_merged_count"] = 0
            st.session_state["_dataset_version"] += 1
            st.success("Dataset réinitialisé.")

        st.button("🗑️ Réinitialiser dataset", on_click=_reset_dataset)

        if uploads:
            try:
                objs: List[Any] = []
                for uf in uploads:
                    try:
                        objs.append(json.loads(uf.read().decode("utf-8")))
                    except Exception:
                        pass
                before = len(st.session_state["RAW_ALL"]) if not st.session_state["RAW_ALL"].empty else 0
                st.session_state["RAW_ALL"] = load_many_jsons(objs, base=st.session_state.get("RAW_ALL"))
                after = len(st.session_state["RAW_ALL"])
                st.session_state["_files_merged_count"] += len(uploads)
                st.session_state["_dataset_version"] += 1
                st.success(f"✅ Fusion: {after:,} messages uniques (Δ {after-before:+})")
            except Exception as e:
                st.error(f"Erreur de parsing: {e}")

        total_uniques = len(st.session_state["RAW_ALL"]) if not st.session_state["RAW_ALL"].empty else 0
        st.caption(f"📦 **{st.session_state['_files_merged_count']}** fichiers fusionnés — **{total_uniques:,}** messages **uniques**")

    return st.session_state["RAW_ALL"]

def ui_status_banner(compact: bool = True):
    """
    Petite bannière en haut de page avec des stats globales sur le dataset courant.
    """
    ensure_session_dataset()
    df = st.session_state["RAW_ALL"]
    stats = _dataset_basic_stats(df)
    if df.empty:
        st.info("Aucun dataset chargé. Utilise **Sources & chargement** dans la sidebar.")
        return
    if compact:
        st.success(
            f"📊 **{stats['n']:,}** messages • **{stats['groups']}** groupes • **{stats['tokens']}** tokens "
            + (f"• fenêtre: _{stats['start']:%Y-%m-%d %H:%M} → {stats['end']:%Y-%m-%d %H:%M}_"
               if (stats['start'] and stats['end']) else "")
        )
    else:
        with st.container(border=True):
            st.markdown("#### 📊 État du dataset")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Messages uniques", f"{stats['n']:,}")
            c2.metric("Groupes", stats["groups"])
            c3.metric("Tokens uniques", stats["tokens"])
            if stats['start'] and stats['end']:
                c4.write(f"**Fenêtre**\n\n{stats['start']:%Y-%m-%d %H:%M} → {stats['end']:%Y-%m-%d %H:%M}")
            st.caption(f"Fichiers fusionnés: {st.session_state['_files_merged_count']}")

# =========================
# Sentiment
# =========================
_vader = SentimentIntensityAnalyzer()

def vader_sentiment_score(text: str) -> float:
    if not text or not str(text).strip(): return 0.0
    s = _vader.polarity_scores(text)
    return float(s.get("compound", 0.0))

EMOJI_WEIGHTS = {
    "🚀": 0.22, "🌕": 0.20, "🔥": 0.10, "💎": 0.12, "✅": 0.10,
    "💀": -0.22, "☠️": -0.22, "🪦": -0.18, "🤡": -0.16, "⚠️": -0.12
}

def crypto_lexicon_score(text: str, custom_lexicon: Optional[Dict[str,float]] = None) -> float:
    """
    Score lexique crypto : accumulation de termes/phrases pondérés (+/-), avec gestion basique de négation.
    """
    if not text: return 0.0
    t = text.lower()
    # Emojis (rendements décroissants + cap +/- 0.6)
    raw = 0.0
    for ch in _strip_vs(text):
        if ch in EMOJI_WEIGHTS:
            raw += EMOJI_WEIGHTS[ch]
    # compresse la somme: tanh garde le signe, limite l'amplitude
    emoji_sum = 0.6 * math.tanh(raw / 0.6)

    lex = dict(CRYPTO_LEXICON)
    if custom_lexicon: lex.update(custom_lexicon)
    tokens = re.findall(r"[a-z0-9\-']+", t)
    score = 0.0
    # phrases en priorité
    for p in PHRASES_PRIORITY:
        if p in t:
            idx = t.find(p)
            left = re.findall(r"[a-z0-9\-']+", t[:max(0, idx)])
            neg = _has_negator(left[-3:])
            val = lex[p]
            score += (-val if neg else val)
            t = t.replace(p, " ")  # éviter double comptage
    # mots unitaires
    for w in re.findall(r"[a-zA-Z][a-zA-Z\-']+", t):
        lw = w.lower()
        if lw in lex:
            try:
                i = tokens.index(lw)
                neg = (i>0 and tokens[i-1] in NEGATORS)
            except Exception:
                neg = False
            val = lex[lw]
            score += (-val if neg else val)
    score += emoji_sum
    return float(max(-1.0, min(1.0, score)))

# ---- Cache disque pour sentiments
def _text_hash(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t.lower())
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

def _load_cache(cache_dir: Path) -> Dict[str, Dict[str,float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "sentiment_cache.parquet"
    res = {"vader":{}, "lex":{}, "hf":{}}
    if not p.exists(): return res
    try:
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            h = r["hash"]
            for col, key in [("vader","vader"),("lex","lex"),("hf","hf")]:
                val = r.get(col, None)
                if pd.notna(val): res[key][h] = float(val)
    except Exception:
        pass
    return res

def _save_cache(cache_dir: Path, cache: Dict[str, Dict[str,float]]):
    rows = []
    all_hashes = set().union(*[set(d.keys()) for d in cache.values()]) if cache else set()
    for h in all_hashes:
        rows.append({
            "hash": h,
            "vader": cache.get("vader",{}).get(h, np.nan),
            "lex":   cache.get("lex",{}).get(h, np.nan),
            "hf":    cache.get("hf",{}).get(h, np.nan),
        })
    df = pd.DataFrame(rows)
    p = Path(cache_dir) / "sentiment_cache.parquet"
    try:
        df.to_parquet(p, index=False)
    except Exception:
        df.to_csv(Path(cache_dir) / "sentiment_cache.csv", index=False)

# ---- Résumeur (DistilBART) + cache
_SUMMARY_READY: Optional[bool] = None
_sum_tok = None; _sum_mod = None
_SUM_CACHE: Dict[str, str] = {}
_SUMMARY_CACHE: Dict[str, str] = {}  # cache par token+période

def _ensure_summarizer():
    """
    Charge DistilBART en CPU "réel" (pas de meta device), prêt pour generate().
    """
    global _SUMMARY_READY, _sum_tok, _sum_mod, _sum_device
    if _SUMMARY_READY is not None:
        return _SUMMARY_READY
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _sum_device = torch.device("cpu")
        _sum_tok = AutoTokenizer.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            use_fast=True
        )
        # IMPORTANT: pas de device_map, pas de low_cpu_mem_usage "meta"
        _sum_mod = AutoModelForSeq2SeqLM.from_pretrained(
            "sshleifer/distilbart-cnn-12-6",
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=torch.float32
        )
        _sum_mod.to(_sum_device)
        _sum_mod.eval()
        _SUMMARY_READY = True
    except Exception:
        _SUMMARY_READY = False
    return _SUMMARY_READY


def summarizer_available() -> bool:
    return True  # DeepSeek est toujours dispo avec la clé API fournie

# ======== Helpers de résumé (sélection, anti-doublon, MMR, langue) ========
def _norm(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

# élargir la détection 'low info'
_LOW_INFO_PAT = re.compile(
    r"^(gm|gn|we are back|let'?s go|send it|pump it|moon|lfg|gm fam|up only|btw|im |i'?m |idk|ngl|lol|haha|gg|nice|ok|okok|let'?s|soon|maybe|could still do well)[\W\d]*$",
    flags=re.I
)
_URL_PAT = re.compile(r"https?://\S+|t\.me/\S+|dexscreener\.com/\S+|coingecko\.com/\S+|cmc\.\S+", re.I)
_TAG_TICKER_PAT = re.compile(r"[@#]\w+|\$[A-Z][A-Z0-9]{1,14}", re.I)

def _mentions_token(sent: str, token: str) -> bool:
    """Vrai si la phrase mentionne explicitement le token ($TOKEN ou TOKEN nu)."""
    if not sent or not token:
        return False
    t = token.lower()
    s = sent.lower()
    # $TOKEN
    if re.search(rf"\${re.escape(t)}\b", s):
        return True
    # TOKEN nu (mot entier)
    return re.search(rf"\b{re.escape(t)}\b", s) is not None

def _is_english_ratio(text: str) -> float:
    """Part (0..1) de caractères alpha ASCII ~ proxy de phrase 'anglaise' pour BART."""
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    ascii_alpha = sum(ch.isalpha() and ch.isascii() for ch in text)
    if alpha == 0:
        return 0.0
    return ascii_alpha / alpha

def _sent_line_weight(txt: str) -> float:
    """Score de salience crypto : boosts lexicaux + longueur utile."""
    if not txt: return 0.0
    t = txt.lower()
    w = 0.0
    for k, v in CRYPTO_LEXICON.items():
        if k in t: w += abs(v)
    if "long term hold" in t or "blue chip" in t: w += 0.3
    L = len(t.split())
    if L < 6: w -= 0.5
    return max(0.0, w)

def _dedupe_keep_order(lines: List[str], sim_thresh: float = 0.85) -> List[str]:
    """Supprime doublons/near-duplicates (Jaccard mots)."""
    out = []
    seen = []
    for s in lines:
        s0 = _norm(_TAG_TICKER_PAT.sub("", _URL_PAT.sub("", s))).lower()
        if _LOW_INFO_PAT.match(s0):
            continue
        sw = set(re.findall(r"[a-z0-9']+", s0))
        if not sw:
            continue
        dup = False
        for tset in seen:
            inter = len(sw & tset)
            uni = len(sw | tset) or 1
            if inter / uni >= sim_thresh:
                dup = True; break
        if not dup:
            seen.append(sw)
            out.append(s.strip())
    return out

def _mmr_select(cands: List[str], k: int = 6, lam: float = 0.7) -> List[str]:
    """MMR simple sur bag-of-words."""
    if not cands: return []
    base = [(s, _sent_line_weight(s)) for s in cands]
    base.sort(key=lambda x: x[1], reverse=True)
    selected = []
    selected_sets = []
    while base and len(selected) < k:
        best_s, best_sc, best_gain = None, -1e9, -1e9
        for s, sc in base:
            sw = set(re.findall(r"[a-z0-9']+", s.lower()))
            sim = 0.0
            for ss in selected_sets:
                inter = len(sw & ss); uni = len(sw | ss) or 1
                sim = max(sim, inter/uni)
            gain = lam*sc - (1-lam)*sim
            if gain > best_gain:
                best_s, best_sc, best_gain = s, sc, gain
        if best_s is None: break
        selected.append(best_s)
        selected_sets.append(set(re.findall(r"[a-z0-9']+", best_s.lower())))
        base = [(s, sc) for (s, sc) in base if s != best_s]
    return selected

def summarize_text(lines_or_text, max_len: int = 110, min_len: int = 40, enabled: bool = False) -> str:
    """
    Utilise DeepSeek pour résumer soit une liste de phrases, soit un texte brut.
    """
    if isinstance(lines_or_text, list):
        text = " ".join(l for l in lines_or_text if isinstance(l, str))
    else:
        text = str(lines_or_text or "")
    text = text.strip()
    if not text:
        return "—"
    return summarize_with_deepseek(text, max_tokens=max_len)

# ---- Sentiment blend + cache HF incrémental
def add_sentiment(
    df: pd.DataFrame,
    use_hf: bool = True,
    w_vader: float = 0.10,          # ⬅️ 10%
    w_crypto: float = 0.10,         # ⬅️ 10% (lexique crypto)
    w_hf: float = 0.80,             # ⬅️ 80% (CryptoBERT)
    rule_weight: float = 0.8,       # ⬅️ règles un peu moins fortes
    group_weight_alpha: float = 1.0,
    custom_lexicon: Optional[Dict[str, float]] = None,
    cache_dir: str = ".cache",
    alias_no_dollar: bool = True,
    gain: float = 1.20,             # ⬅️ étire modestement l’échelle
) -> pd.DataFrame:
    if df.empty:
        # garantir les colonnes attendues par les pages, même sans lignes
        for c in ["sentiment", "sentiment_hf", "w_sentiment"]:
            if c not in df.columns:
                df[c] = pd.Series(dtype=float)
        return df
    df = df.copy()
    # alias sans $
    if alias_no_dollar:
        df = _apply_alias_no_dollar_strict(df, min_len=3, min_occ_with_dollar=2)

    df["text"] = df["text"].fillna("")
    df["remark"] = df.get("remark", "").fillna("")
    df["text_for_sent"] = (df["text"].astype(str) + " " + df["remark"].astype(str)).str.strip()

    cache = _load_cache(Path(cache_dir))

    # VADER + LEX
    _v, _l, _h = [], [], []
    for t in df["text_for_sent"]:
        h = _text_hash(t)
        if h in cache["vader"]:
            _v.append(cache["vader"][h])
        else:
            s = vader_sentiment_score(t); _v.append(s); cache["vader"][h]=s
        if h in cache["lex"]:
            _l.append(cache["lex"][h])
        else:
            s = crypto_lexicon_score(t, custom_lexicon=custom_lexicon); _l.append(s); cache["lex"][h]=s
    df["sentiment_vader"] = _v
    df["sentiment_crypto"] = _l

    # HF (incrémental + cache)
    if use_hf:
        try:
            from sentiment_local import batch_scores  # doit exister dans ton projet
            texts = df["text_for_sent"].tolist()
            hashes = [_text_hash(t) for t in texts]
            to_idx = [i for i,h in enumerate(hashes) if h not in cache["hf"]]
            _h = [cache["hf"].get(h, np.nan) for h in hashes]
            if to_idx:
                outs = batch_scores([texts[i] for i in to_idx])
                for k,i in enumerate(to_idx):
                    val = float(max(-1.0, min(1.0, outs[k])))
                    _h[i] = val
                    cache["hf"][hashes[i]] = val
        except Exception:
            _h = [np.nan]*len(df)
    else:
        _h = [np.nan]*len(df)
    df["sentiment_hf"] = _h
    try: _save_cache(Path(cache_dir), cache)
    except Exception: pass

    # Mix par ligne
    def mix_row(r):
        vals, ws = [], []
        if pd.notna(r.get("sentiment_hf", np.nan)) and w_hf>0:
            vals.append(float(r["sentiment_hf"])); ws.append(float(w_hf))
        if pd.notna(r.get("sentiment_vader", np.nan)) and w_vader>0:
            vals.append(float(r["sentiment_vader"])); ws.append(float(w_vader))
        if pd.notna(r.get("sentiment_crypto", np.nan)) and w_crypto>0:
            vals.append(float(r["sentiment_crypto"])); ws.append(float(w_crypto))
        if not vals:
            s=0.0
        else:
            sw=sum(ws) or len(vals); ws=[w/sw for w in ws]; s=float(np.dot(ws, vals))
        s,_,_ = rule_adjust(r.get("text_for_sent",""), s, weight=rule_weight)
        s = max(-1.0, min(1.0, s * float(gain)))
        return float(s)

    df["sentiment"] = df.apply(mix_row, axis=1)
    conv = df["conviction"].fillna(7)
    w = 1.0 + group_weight_alpha * (conv - 5.0)/10.0
    df["w_sentiment"] = df["sentiment"] * w
    df["group_weight"] = w
    return df

# --- utils.py (AJOUTER EN BAS DU FICHIER, après add_sentiment) ---

def _mix_from_components_row(
    r,
    w_vader: float,
    w_crypto: float,
    w_hf: float,
    rule_weight: float,
    gain: float,
    group_weight_alpha: float,
) -> float:
    """Remix ultra-rapide à partir des colonnes déjà calculées : sentiment_vader/crypto/hf + rule_adjust."""
    vals, ws = [], []
    sv = r.get("sentiment_vader", None)
    sl = r.get("sentiment_crypto", None)
    sh = r.get("sentiment_hf", None)

    if pd.notna(sv) and w_vader > 0:  vals.append(float(sv)); ws.append(float(w_vader))
    if pd.notna(sl) and w_crypto > 0: vals.append(float(sl)); ws.append(float(w_crypto))
    if pd.notna(sh) and w_hf > 0:     vals.append(float(sh)); ws.append(float(w_hf))

    if not vals:
        s = 0.0
    else:
        sw = sum(ws) or len(vals)
        ws = [w / sw for w in ws]
        s = float(np.dot(ws, vals))

    # Règles lexicales (pas de ré-inférence, juste réappliquer)
    s, _, _ = rule_adjust(r.get("text_for_sent", ""), s, weight=rule_weight)

    # Gain + clip
    s = max(-1.0, min(1.0, s * float(gain)))
    return float(s)


def remix_sentiment_weights(
    df_with_components: pd.DataFrame,
    w_vader: float = 0.10,
    w_crypto: float = 0.10,
    w_hf: float = 0.80,
    rule_weight: float = 0.8,
    gain: float = 1.20,
    group_weight_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    ⚡ Recalcule 'sentiment' et 'w_sentiment' à partir des colonnes déjà présentes :
      - sentiment_vader / sentiment_crypto / sentiment_hf
      - text_for_sent, conviction
    Sans ré-exécuter le modèle HF ni VADER. Ultra rapide.
    """
    if df_with_components.empty:
        return df_with_components

    df = df_with_components.copy()
    if "text_for_sent" not in df.columns:
        # si besoin, reconstituer minimalement
        df["text_for_sent"] = (df.get("text","").astype(str) + " " + df.get("remark","").astype(str)).str.strip()

    df["sentiment"] = df.apply(
        lambda r: _mix_from_components_row(r, w_vader, w_crypto, w_hf, rule_weight, gain, group_weight_alpha),
        axis=1
    )

    conv = pd.to_numeric(df.get("conviction", 7), errors="coerce").fillna(7)
    group_weight = 1.0 + group_weight_alpha * (conv - 5.0) / 10.0
    df["group_weight"] = group_weight
    df["w_sentiment"] = df["sentiment"] * df["group_weight"]
    return df

# ---- Explode / per-token
def explode_tokens(df: pd.DataFrame, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty or "tokens" not in df.columns:
        base_cols=["id","group","date","token","sentiment","w_sentiment","conviction","text","remark"]
        if extra_cols:
            for c in extra_cols:
                if c not in base_cols: base_cols.append(c)
        return pd.DataFrame(columns=base_cols)
    if extra_cols is None:
        extra_cols = [c for c in ["bin","sentiment_hf","sentiment_vader","sentiment_crypto"] if c in df.columns]
    rows=[]
    for _, r in df.iterrows():
        toks = r.get("tokens") or []
        for t in toks:
            row = {
                "id": r.get("id"),
                "group": r.get("group"),
                "date": r.get("date"),
                "token": t,
                "sentiment": r.get("sentiment"),
                "w_sentiment": r.get("w_sentiment", r.get("sentiment", np.nan)),
                "conviction": r.get("conviction"),
                "text": r.get("text",""),
                "remark": r.get("remark",""),
            }
            for c in extra_cols:
                row[c] = r.get(c, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def available_tokens(df: pd.DataFrame) -> List[str]:
    if df.empty or "tokens" not in df.columns: return []
    return sorted({t for ts in df["tokens"] for t in (ts or [])})

# ---- Cache résumés par token + période
def _summary_key(token: str, df_token: pd.DataFrame, enabled: bool) -> str:
    lines = []
    for _, r in df_token.iterrows():
        t = (str(r.get("text","")) + " " + str(r.get("remark",""))).strip().lower()
        t = re.sub(r"\s+", " ", t)
        lines.append(t)
    blob = " ".join(lines)[:5000]
    return f"{token}|{enabled}|{hashlib.sha1(blob.encode('utf-8')).hexdigest()}"

def per_token_view(df: pd.DataFrame, include_summaries: bool = True, summarizer_enabled: bool = False) -> pd.DataFrame:
    dt = explode_tokens(df, extra_cols=["sentiment_hf"] if "sentiment_hf" in df.columns else None)
    if dt.empty:
        cols=["token","mentions","sentiment","ci95","mot-clés","résumé","Sentiment_HF"]
        return pd.DataFrame(columns=cols)

    # --- helpers pour candidats de résumé (ANCRÉ AU TOKEN) ---
    def _candidate_lines(df_token: pd.DataFrame, k_max: int = 30, token: str = "") -> List[str]:
        rows = []
        for _, r in df_token.iterrows():
            sc = float(r.get("w_sentiment", r.get("sentiment", 0.0)) or 0.0)
            conv = float(r.get("conviction", 7.0))
            bonus_group = 0.10 * ((conv - 5.0) / 5.0)  # petit bonus si groupe très convaincu
            txts = []
            if r.get("text"): txts.append(str(r["text"]))
            if r.get("remark"): txts.append(str(r["remark"]))
            txt = " ".join(txts)

            # split en phrases et scoring
            for sent in re.split(r'(?<=[.!?])\s+', txt):
                s = sent.strip()
                if len(s) < 15:
                    continue
                if _LOW_INFO_PAT.match(s):
                    continue
                # ancrage: on favorise (fortement) les phrases qui citent le token
                has_tok = _mentions_token(s, token)
                tok_bonus = 0.6 if has_tok else 0.0
                w = abs(sc) + _sent_line_weight(s) + tok_bonus + bonus_group
                # si la phrase ne mentionne pas le token ET n'a pas de salience, on jette
                if not has_tok and _sent_line_weight(s) < 0.2 and abs(sc) < 0.15:
                    continue
                rows.append((s, w, has_tok))

        if not rows:
            return []

        # on sépare celles qui mentionnent le token (on en garantit ~60%)
        rows.sort(key=lambda x: x[1], reverse=True)
        primary = [s for (s, _, has_tok) in rows if has_tok][: k_max]  # cœur
        secondary = [s for (s, _, has_tok) in rows if not has_tok][: k_max]  # contexte utile

        # anti-duplicat & MMR sur l'ensemble
        pool = primary + secondary
        pool = _dedupe_keep_order(pool, sim_thresh=0.80)
        # Assurer une majorité de phrases ancrées
        anchored = [s for s in pool if _mentions_token(s, token)]
        not_anch = [s for s in pool if not _mentions_token(s, token)]
        pool = (anchored[: int(0.6*k_max)] + not_anch)[: k_max*3]
        return _mmr_select(pool, k=k_max, lam=0.7)

    def flags_for_token(df_token: pd.DataFrame) -> List[str]:
        series_text = (df_token["text"].astype(str) if "text" in df_token.columns else pd.Series([], dtype=str))
        series_remk = (df_token["remark"].astype(str) if "remark" in df_token.columns else pd.Series([], dtype=str))
        low_text = " \n".join(series_text.tolist() + series_remk.tolist()).lower()
        greens = sorted([f"+{k}" for k in POS_BOOST if k in low_text])
        reds   = sorted([f"-{k}" for k in NEG_BOOST if k in low_text])
        return greens + reds

    rows=[]
    for token, g in dt.groupby("token"):
        vals = g["w_sentiment"] if "w_sentiment" in g.columns else g["sentiment"]
        vals = pd.to_numeric(vals, errors="coerce").fillna(0.0).to_numpy()
        mean = float(np.mean(vals)) if len(vals) else 0.0
        std  = float(np.std(vals)) if len(vals) else 0.0
        n    = int(len(vals))
        ci   = float(1.96*std/np.sqrt(max(1,n)))

        keywords = ", ".join(flags_for_token(g)) or "—"

        # résumé (avec cache + gating langue pour BART)
        if include_summaries:
            key = _summary_key(token, g, bool(summarizer_enabled))
            if key in _SUMMARY_CACHE:
                summary = _SUMMARY_CACHE[key]
            else:
                cands = _candidate_lines(g, k_max=30, token=token)
                # on n’active BART que si majorité anglaise (robustesse)
                english_share = _is_english_ratio(" ".join(cands)) if cands else 0.0
                use_bart = bool(summarizer_enabled) and (english_share >= 0.5)
                summary = summarize_text(
                    cands if not use_bart else [s for s in cands if _is_english_ratio(s) >= 0.7],
                    max_len=120, min_len=50,
                    enabled=use_bart
                )
                _SUMMARY_CACHE[key] = summary
        else:
            summary = "—"

        hf_mean = float(pd.to_numeric(g.get("sentiment_hf", pd.Series([], dtype=float)), errors="coerce").mean()) if "sentiment_hf" in g.columns else np.nan

        rows.append({
            "token": token,
            "mentions": n,
            "sentiment": round(mean,3),
            "ci95": round(ci,3),
            "mot-clés": keywords,
            "résumé": summary,
            "Sentiment_HF": (round(hf_mean,3) if pd.notna(hf_mean) else np.nan),
        })
    out = pd.DataFrame(rows).sort_values(["mentions","sentiment"], ascending=[False, False])
    return out

# =========================
# Graphe avancé (τ, NPMI, Jaccard group-group)
# =========================
def _now_paris_naive():
    return pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)

def _decay_weight(ts: pd.Timestamp, now: pd.Timestamp, tau_hours: float) -> float:
    if pd.isna(ts) or tau_hours <= 0: return 1.0
    dt = (now - ts).total_seconds() / 3600.0
    if dt < 0: dt = 0.0
    return float(math.exp(-dt / tau_hours))

def _npmi_token_pairs(dt: pd.DataFrame, msg_decay: Dict[Any,float]) -> pd.DataFrame:
    freq=defaultdict(float); cooc=defaultdict(float); N=0.0
    for mid, g in dt.groupby("id"):
        w = msg_decay.get(mid, 1.0)
        toks = sorted(set(g["token"].tolist()))
        if not toks: continue
        N += w
        for t in toks: freq[t] += w
        for i in range(len(toks)):
            for j in range(i+1, len(toks)):
                cooc[(toks[i], toks[j])] += w
    eps=1e-12; rows=[]
    for (i,j), cij in cooc.items():
        p_ij = cij / max(N, eps); p_i = freq[i]/max(N,eps); p_j=freq[j]/max(N,eps)
        if p_ij <= 0: continue
        pmi  = math.log((p_ij) / (p_i*p_j + eps) + eps)
        npmi = pmi / (-math.log(p_ij + eps))
        rows.append((i,j,cij,npmi))
    return pd.DataFrame(rows, columns=["ti","tj","cooc","npmi"])

def graph_edges_advanced(df: pd.DataFrame, tau_hours: float = 12.0, group_sent_source: str = "calc"):
    """
    Construit:
      - edges: src, dst, weight, type ∈ {"group-token","token-token","group-group"}, sentiment moyen, cooc/npmi (tokens)
      - node_sent: node, kind ∈ {"group","token"}, sentiment moyen
      - rel_df: mapping (rel_type, key) -> message_id pour "Voir messages"
    """
    if df.empty:
        return (pd.DataFrame(columns=["src","dst","weight","type","sentiment","npmi","cooc"]),
                pd.DataFrame(columns=["node","kind","sentiment"]),
                pd.DataFrame(columns=["rel_type","key","message_id"]))
    now = _now_paris_naive()

    # sentiment groupe: calculé ou basé sur conviction fournie
    if group_sent_source == "provided":
        tmp = df.groupby("group")["conviction"].mean().reset_index(name="cv")
        tmp["sentiment"] = ((tmp["cv"] - 5.0) / 5.0).clip(-1,1)
        group_sent = tmp[["group","sentiment"]].rename(columns={"group":"node"})
    else:
        group_sent = df.groupby("group")["w_sentiment"].mean().reset_index(name="sentiment")
        group_sent.rename(columns={"group":"node"}, inplace=True)
    group_sent["kind"] = "group"

    dt = explode_tokens(df)
    if not dt.empty:
        token_sent = dt.groupby("token")["w_sentiment"].mean().reset_index(name="sentiment")
        token_sent.rename(columns={"token":"node"}, inplace=True)
        token_sent["kind"]="token"
    else:
        token_sent = pd.DataFrame(columns=["node","sentiment","kind"])

    node_sent = pd.concat([group_sent[["node","kind","sentiment"]], token_sent[["node","kind","sentiment"]]], ignore_index=True)

    # poids temporel par message
    m_decay={}
    for _, r in df.iterrows():
        mid = r.get("id"); m_decay[mid] = _decay_weight(r.get("date"), now, tau_hours)

    rows=[]; rel_map=[]
    if not dt.empty:
        for mid, g in dt.groupby("id"):
            toks = sorted({t for t in g["token"].tolist() if isinstance(t,str)})
            if not toks: continue
            grp = g["group"].iloc[0]
            try: ms = float(df.loc[df["id"]==mid,"w_sentiment"].iloc[0])
            except Exception: ms = 0.0
            wdec = m_decay.get(mid, 1.0)
            # group-token
            for t in toks:
                rows.append((grp, t, wdec, "group-token", ms))
                rel_map.append(("group-token", f"{grp}|{t}", mid))
            # token-token
            for i in range(len(toks)):
                for j in range(i+1, len(toks)):
                    rows.append((toks[i], toks[j], wdec, "token-token", ms))
                    rel_map.append(("token-token", f"{toks[i]}|{toks[j]}", mid))
    e1 = pd.DataFrame(rows, columns=["src","dst","weight","type","sentiment"])
    if e1.empty:
        e1g = pd.DataFrame(columns=["src","dst","weight","type","sentiment"])
    else:
        e1g = e1.groupby(["src","dst","type"]).agg(weight=("weight","sum"), sentiment=("sentiment","mean")).reset_index()

    # NPMI sur token-token
    if not dt.empty:
        npmi_df = _npmi_token_pairs(dt, m_decay)
    else:
        npmi_df = pd.DataFrame(columns=["ti","tj","cooc","npmi"])
    if not npmi_df.empty:
        npmi_df["k1"] = npmi_df["ti"] + "||" + npmi_df["tj"]
        e1g["k1"] = e1g["src"] + "||" + e1g["dst"]
        e1g = e1g.merge(npmi_df[["k1","cooc","npmi"]], on="k1", how="left").drop(columns=["k1"])
    else:
        e1g["cooc"]=np.nan; e1g["npmi"]=np.nan

    # group-group via similarité d'usage tokens (Jaccard pondéré min/max)
    gt = e1[e1["type"]=="group-token"].groupby(["src","dst"]).agg(w=("weight","sum")).reset_index()
    from collections import defaultdict as _dd
    group_token = _dd(dict)
    for _, r in gt.iterrows():
        group_token[r["src"]][r["dst"]] = float(r["w"])
    gg_rows=[]
    groups = list(group_token.keys())
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            gi, gj = groups[i], groups[j]
            toks = set(group_token[gi].keys()) | set(group_token[gj].keys())
            if not toks: continue
            num=den=0.0
            for t in toks:
                wi = group_token[gi].get(t,0.0); wj = group_token[gj].get(t,0.0)
                num += min(wi,wj); den += max(wi,wj)
            if den <= 0: continue
            si = node_sent.loc[(node_sent["node"]==gi)&(node_sent["kind"]=="group"), "sentiment"]
            sj = node_sent.loc[(node_sent["node"]==gj)&(node_sent["kind"]=="group"), "sentiment"]
            s = float((si.mean() if not si.empty else 0.0) + (sj.mean() if not sj.empty else 0.0)) / 2.0
            gg_rows.append((gi, gj, num/den, "group-group", s))
    e2 = pd.DataFrame(gg_rows, columns=["src","dst","weight","type","sentiment"])
    e2["cooc"]=np.nan; e2["npmi"]=np.nan

    edges = pd.concat([e1g, e2], ignore_index=True)
    rel_df = pd.DataFrame(rel_map, columns=["rel_type","key","message_id"])
    return edges, node_sent, rel_df

def dataset_signature(df: pd.DataFrame) -> str:
    """Empreinte stable du dataset (change seulement si le JSON change).
    • Gère les colonnes list (ex: tokens) en les normalisant en chaîne triée.
    • Indépendant de l'ordre des lignes.
    """
    if df is None or df.empty:
        return "empty"

    cols = [c for c in ["id","date","group","conviction","text","remark","tokens"] if c in df.columns]
    snap = df[cols].copy()

    # Normalisations par colonne
    if "date" in snap.columns:
        snap["date"] = pd.to_datetime(snap["date"], errors="coerce").astype("int64").fillna(0).astype("int64")

    # tokens peut être une liste → convertir en chaîne stable
    if "tokens" in snap.columns:
        def _tok_norm(v):
            try:
                if isinstance(v, (list, tuple, set)):
                    return ",".join(sorted(map(lambda x: str(x).upper().replace("$",""), v)))
                return str(v)
            except Exception:
                return str(v)
        snap["tokens"] = snap["tokens"].apply(_tok_norm)

    # text/remark/group → strip
    for c in ("text","remark","group"):
        if c in snap.columns:
            snap[c] = snap[c].astype(str).str.strip()

    # Tri pour stabilité — éviter de trier par 'tokens' (inutile + déjà normalisé)
    sort_by = [c for c in ["id","date","group","conviction","text","remark"] if c in snap.columns]
    try:
        snap = snap.sort_values(sort_by)
    except Exception:
        # Fallback: tri uniquement par id/date si dispo
        sort_fallback = [c for c in ["id","date"] if c in snap.columns]
        if sort_fallback:
            snap = snap.sort_values(sort_fallback)

    # Sérialisation compacte
    blob = ("||".join(["|".join(map(str, r)) for r in snap.to_numpy().tolist()]))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def summarize_top_tokens_deepseek(
    df,
    top_n=15,
    w_mentions=0.6,
    max_tokens=140,
    enabled=False,      # <-- nouveau param
):
    """Génère et met en cache les résumés DeepSeek pour le Top-N par score_conviction."""
    # 🔒 NE RIEN FAIRE si le toggle n'est pas activé
    if not enabled:
        return {}

    if df.empty:
        return {}

    sig = dataset_signature(df)
    st.session_state.setdefault("_SUM_CACHE_BY_DATASET", {})
    if sig not in st.session_state["_SUM_CACHE_BY_DATASET"]:
        st.session_state["_SUM_CACHE_BY_DATASET"][sig] = {}
    cache = st.session_state["_SUM_CACHE_BY_DATASET"][sig]

    dt = explode_tokens(df)
    if dt.empty:
        return {}

    agg = dt.groupby("token").agg(
        mentions=("id","count"),
        sentiment=("w_sentiment","mean")
    ).reset_index()
    agg["m_norm"] = agg["mentions"] / agg["mentions"].max()
    agg["s_norm"] = (agg["sentiment"] + 1.0) / 2.0
    agg["score_conviction"] = (w_mentions*agg["m_norm"] + (1-w_mentions)*agg["s_norm"]) * 10.0
    top_tokens = agg.sort_values("score_conviction", ascending=False).head(top_n)["token"].tolist()

    results = {}
    for tok in top_tokens:
        if tok in cache:
            results[tok] = cache[tok]
            continue

        # Récupérer phrases candidates (ancrées au token, déjà implémenté chez toi)
        lines = []
        for _, r in dt[dt["token"] == tok].iterrows():
            parts = []
            if r.get("text"): parts.append(str(r["text"]))
            if r.get("remark"): parts.append(str(r["remark"]))
            s = " ".join(parts).strip()
            if len(s) > 15:
                lines.append(s)

        if not lines:
            cache[tok] = "—"
            results[tok] = "—"
            continue

        # 👉 On fusionne en un seul blob (max 30 phrases), DeepSeek imposera la structure
        blob = "\n".join(lines[:30])

        try:
            # On passe le symbole pour contextualiser le prompt
            summary = deepseek_summary(blob, max_tokens=max_tokens, symbol=tok)
        except Exception as e:
            summary = f"[Erreur DeepSeek] {e}"

        cache[tok] = summary
        results[tok] = summary

    return results