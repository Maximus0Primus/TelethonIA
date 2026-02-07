# memecoin_dashboard.py
import re
import os
import json
import math
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
from collections import defaultdict, Counter

import pandas as pd
import pytz
import streamlit as st

# ---------- Config ----------
MAJORS = {"$BTC", "$ETH", "$SOL"}
MIN_FINAL_SCORE = 0.0  # slider UI will filter further if needed

# pondérations (conservées comme la version qui te convenait)
W_BASE = 0.55
W_UNIQUE_GROUPS = 0.35
W_MENTIONS = 0.10

# règles de lexique (FR/EN) pour le scoring heuristique message
STRONG_POS = [
    "giga bullish", "high conviction", "very confident", "très confiant", "conviction élevée",
    "ATH", "new ath", "sending", "giga send", "gigasendor", "taking off",
    "mooning", "parabolic", "hundreds of millions", "100x", "50x",
    "lock in", "locked in", "strongest", "beast",
    "comfy in my bags", "hodl season", "have some conviction",
    "insane active community", "hard working community",
]
MID_POS = [
    "accumulation zone", "looks good", "bottom bids", "uptrend",
    "motion", "starting to get motion", "gearing up",
    "adding more", "buy on dips", "consolidating",
    "featured on home page", "listed on", "new highs soon",
]
NEG = [
    "annoyed", "fding out", "dump", "dumps", "liquidated",
    "not in any of these", "won't rush", "observing", "afk",
]
REDFLAGS = [
    "prelaunch", "scam", "rug", "rugged", "cabal"
]

# pénalités/bonus par catégorie
SENT_WEIGHTS = {
    "STRONG_POS": 3,
    "MID_POS": 2,
    "NEG": -2,
}
RED_FLAG_PENALTY = -3  # appliqué en plus si red flag

# pondération contextuelle via "remarque" de groupe (tes notes de profil)
# (ex: groupe avec "winrate très haut" => léger bonus)
GROUP_REMARK_BONUS = {
    # clés en minuscules partielle => bonus
    "winrate": 0.3,
    "très bonne conviction": 0.2,
    "conviction de ouf": 0.4,
    "plus long terme": 0.1,
    "très fort": 0.2,
}

# ---------- Helpers ----------
TOKEN_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9_]*")

def is_amount_symbol(tok: str) -> bool:
    """
    Exclut les montants type $2M, $120k, $8K, $1b, $10B etc.
    """
    body = tok[1:]  # rm '$'
    return bool(re.fullmatch(r"(\d+([.,]?\d+)?)(k|m|b|K|M|B|K\d*|M\d*|B\d*)?", body))

def extract_tokens(text: str):
    tokens = set(TOKEN_RE.findall(text or ""))
    clean = []
    for t in tokens:
        if t in MAJORS:
            continue
        if is_amount_symbol(t):
            continue
        clean.append(t)
    return clean

def score_message_heuristic(msg_text: str) -> (int, list, list):
    """
    Retourne (score_brut, pos_hits, neg_hits_redflags)
    - score_brut sur l'échelle -3..+3 (avant normalisation 0..10)
    """
    t = (msg_text or "").lower()
    score = 0
    pos_hits, neg_hits = [], []

    # positives
    for p in STRONG_POS:
        if p in t:
            score += SENT_WEIGHTS["STRONG_POS"]
            pos_hits.append(p)
    for p in MID_POS:
        if p in t:
            score += SENT_WEIGHTS["MID_POS"]
            pos_hits.append(p)

    # negatives
    for n in NEG:
        if n in t:
            score += SENT_WEIGHTS["NEG"]
            neg_hits.append(n)
    # redflags
    for r in REDFLAGS:
        if r in t:
            score += RED_FLAG_PENALTY
            neg_hits.append(r + " (redflag)")

    # clamp
    score = max(-3, min(3, score))
    return score, pos_hits, neg_hits

def normalise_msg_score_to_10(x):
    # -3..+3 -> 0..10
    return (x + 3) / 6 * 10

def group_profile_bonus(remark: str) -> float:
    if not remark:
        return 0.0
    r = remark.lower()
    bonus = 0.0
    for k, v in GROUP_REMARK_BONUS.items():
        if k in r:
            bonus += v
    return bonus

def now_utc():
    return datetime.now(timezone.utc)

def parse_dt(s):
    # supporte iso strings (ex: "2025-08-07T14:50:57+00:00")
    try:
        return dateparser.parse(s)
    except Exception:
        return None

# ---------- UI ----------
st.set_page_config(page_title="Memecoin Dashboard", layout="wide")
st.title("🧭 Memecoin Dashboard (JSON Telegram)")

# Fichier JSON
json_path = st.text_input("Chemin du fichier JSON", value="messages_export_20250808_122416.json")

# Fenêtre de temps
time_window = st.selectbox("Fenêtre de temps", ["6h", "24h", "3j", "7j", "14j", "Tout"], index=1)
max_age = None
if time_window != "Tout":
    if time_window.endswith("h"):
        hours = int(time_window[:-1])
        max_age = timedelta(hours=hours)
    elif time_window.endswith("j"):
        days = int(time_window[:-1])
        max_age = timedelta(days=days)

min_final_score = st.slider("Seuil d'affichage (Score final ≥)", 0.0, 10.0, 0.0, 0.1)

# Option LLM
use_llm = st.checkbox("Affinage LLM (OpenAI) pour enrichir remarques & résumé", value=False)
if use_llm:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("OPENAI_API_KEY manquant dans l'environnement. Le pipeline LLM sera ignoré.")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Impossible d'initialiser l'API OpenAI: {e}")
            use_llm = False

# ---------- Lecture JSON ----------
if not os.path.exists(json_path):
    st.error("Fichier introuvable. Vérifie le chemin.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# data = {group_name: [ {group, conviction, remarque, id, date, text}, ... ], ...}
records = []
for gname, msgs in data.items():
    for m in msgs:
        # Nettoyage minimal – attendu comme dans tes fichiers
        records.append({
            "group": m.get("group", gname),
            "group_conviction": m.get("conviction", None),
            "group_remark": m.get("remarque", ""),
            "id": m.get("id", None),
            "date": m.get("date", None),
            "text": m.get("text", ""),
        })

df = pd.DataFrame(records)

# filtre par temps
if max_age is not None:
    now = now_utc()
    df["dt"] = df["date"].apply(parse_dt)
    df = df[df["dt"].notna()]
    df = df[df["dt"] >= (now - max_age)]
else:
    df["dt"] = df["date"].apply(parse_dt)
    df = df[df["dt"].notna()]

if df.empty:
    st.warning("Aucun message dans la fenêtre sélectionnée.")
    st.stop()

# ---------- Extraction tokens + scoring message ----------
rows = []
for _, row in df.iterrows():
    tokens = extract_tokens(row["text"] or "")
    if not tokens:
        continue
    score_raw, pos_hits, neg_hits = score_message_heuristic(row["text"] or "")
    rows.append({
        "group": row["group"],
        "group_conviction": row.get("group_conviction", None),
        "group_remark": row.get("group_remark", ""),
        "id": row["id"],
        "dt": row["dt"],
        "text": row["text"] or "",
        "tokens": list(tokens),
        "score_raw": score_raw,
        "pos_hits": pos_hits,
        "neg_hits": neg_hits,
    })

if not rows:
    st.warning("Aucun token détecté après filtrage ($, hors majors et montants).")
    st.stop()

mx = pd.DataFrame(rows).explode("tokens").rename(columns={"tokens": "token"})
# comptages globaux pour normaliser mentions / groupes uniques
mentions_by_token = mx.groupby("token")["id"].count()
groups_by_token = mx.groupby("token")["group"].nunique()
max_mentions = max(mentions_by_token.max(), 1)
max_groups = max(groups_by_token.max(), 1)

# agrégation token
agg = defaultdict(lambda: {
    "mentions": 0,
    "groups": set(),
    "msg_scores": [],        # (score_norm_10, weight)
    "pos_remarks": [],       # [(group, text)]
    "neg_remarks": [],
    "redflags": set(),
    "raw_msgs": [],          # pour LLM
})

for _, r in mx.iterrows():
    token = r["token"]
    grp = r["group"]
    gconv = r["group_conviction"] if r["group_conviction"] is not None else 6.0  # défaut soft
    gbonus = group_profile_bonus(r["group_remark"] or "")
    # message score normalisé 0..10
    s10 = normalise_msg_score_to_10(r["score_raw"])
    # pondération par conviction groupe + bonus profil
    weight = max(0.1, float(gconv) / 10.0 + gbonus)  # >= 0.1

    agg[token]["mentions"] += 1
    agg[token]["groups"].add(grp)
    agg[token]["msg_scores"].append((s10, weight))
    agg[token]["raw_msgs"].append((grp, r["text"]))

    # remarques + détection redflags
    if r["pos_hits"]:
        agg[token]["pos_remarks"].append((grp, r["text"]))
    if r["neg_hits"]:
        agg[token]["neg_remarks"].append((grp, r["text"]))
        for nh in r["neg_hits"]:
            if "redflag" in nh:
                if "prelaunch" in nh: agg[token]["redflags"].add("prelaunch")
                if "scam" in nh: agg[token]["redflags"].add("scam")
                if "rug" in nh: agg[token]["redflags"].add("rug")
                if "cabal" in nh: agg[token]["redflags"].add("cabal")

# calcul base score + final
rows_out = []
for tok, d in agg.items():
    # base_score = moyenne pondérée des s10
    if d["msg_scores"]:
        num = sum(s * w for s, w in d["msg_scores"])
        den = sum(w for _, w in d["msg_scores"])
        base = num / max(den, 1e-9)
    else:
        base = 5.0  # neutre

    uniq_groups = len(d["groups"])
    mentions = d["mentions"]

    # normalisations pour composantes quantité
    groups_norm = (uniq_groups / max_groups) * 10.0
    mentions_norm = (mentions / max_mentions) * 10.0

    final_score = W_BASE * base + W_UNIQUE_GROUPS * groups_norm + W_MENTIONS * mentions_norm
    final_score = max(0.0, min(10.0, final_score))

    # remarques positives/négatives (dédupe simples, limiter pour lisibilité)
    def pick_snippets(pairs, maxn=4):
        out = []
        seen = set()
        for g, t in pairs:
            snip = t.strip()
            key = (g, snip)
            if key in seen:
                continue
            seen.add(key)
            # coupe proprement
            if len(snip) > 220:
                snip = snip[:217] + "..."
            out.append(f"[{g}] {snip}")
            if len(out) >= maxn:
                break
        return out

    pos_list = pick_snippets(d["pos_remarks"], maxn=4)
    neg_list = pick_snippets(d["neg_remarks"], maxn=4)

    # résumé global heuristique (sans LLM)
    redflag_txt = ", ".join(sorted(d["redflags"])) if d["redflags"] else ""
    summary_parts = []
    if uniq_groups >= 3:
        summary_parts.append(f"Large consensus: {uniq_groups} groupes en parlent.")
    elif uniq_groups == 2:
        summary_parts.append("Couverture multi-groupes.")
    else:
        summary_parts.append("Signal porté par peu de groupes.")

    if base >= 7.5:
        summary_parts.append("Tonalité très positive (mentions de forte conviction/ATH/moon).")
    elif base >= 6.0:
        summary_parts.append("Tonalité globalement positive (accumulation/consolidation).")
    elif base <= 4.0:
        summary_parts.append("Tonalité prudente ou négative.")

    if redflag_txt:
        summary_parts.append(f"⚠️ Red flags détectés: {redflag_txt}")

    summary_heur = " ".join(summary_parts)

    rows_out.append({
        "token": tok,
        "mentions": mentions,
        "unique_groups": uniq_groups,
        "avg_token_conviction": round(base, 2),
        "final_score": round(final_score, 2),
        "positive_remarks": " • ".join(pos_list) if pos_list else "",
        "negative_remarks": " • ".join(neg_list) if neg_list else "",
        "red_flags": redflag_txt,
        "summary": summary_heur,
        "_raw_for_llm": d["raw_msgs"],  # pour enrichissement optionnel
    })

dash = pd.DataFrame(rows_out)
if dash.empty:
    st.warning("Aucun token après agrégation.")
    st.stop()

# affinage LLM (optionnel) – enrichit positive/negative/summary
if use_llm and "client" in locals():
    def llm_refine(token, msgs):
        # Prend quelques messages (max 6, divers groupes) pour résumer proprement
        sample = msgs[:6]
        payload = "\n".join([f"- [{g}] {t}" for g, t in sample])
        prompt = f"""
Tu es un analyste qui lit des messages Telegram sur le token {token}.
Sépare en 2 listes courtes: "POSITIF" et "NÉGATIF" (phrases courtes, sans doublons) basées sur l'intention et le sens.
Puis écris un RÉSUMÉ GLOBAL concis (2 phrases max) en mettant en évidence les signaux forts (plusieurs gros callers, accumulation, ATH)
et les risques (prelaunch, rug, scam, cabal) si visibles. Pas de score chiffré dans le résumé.

Messages:
{payload}
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content.strip()
            return content
        except Exception as e:
            return None

    with st.spinner("Affinage LLM en cours..."):
        refined_pos = {}
        refined_neg = {}
        refined_sum = {}
        for i, r in dash.iterrows():
            txt = llm_refine(r["token"], r["_raw_for_llm"])
            if not txt:
                continue
            # parsing très simple
            pos, neg, summ = [], [], []
            bloc = txt.splitlines()
            current = None
            for line in bloc:
                L = line.strip()
                if not L:
                    continue
                U = L.upper()
                if "POSITIF" in U:
                    current = "pos"; continue
                if "NÉGATIF" in U or "NEGATIF" in U:
                    current = "neg"; continue
                if "RÉSUMÉ" in U or "RESUME" in U:
                    current = "sum"; continue
                if current == "pos":
                    pos.append(L.lstrip("-• ").strip())
                elif current == "neg":
                    neg.append(L.lstrip("-• ").strip())
                elif current == "sum":
                    summ.append(L)
            if pos: refined_pos[i] = " • ".join(pos[:4])
            if neg: refined_neg[i] = " • ".join(neg[:4])
            if summ: refined_sum[i] = " ".join(summ).strip()

        if refined_pos:
            dash.loc[list(refined_pos.keys()), "positive_remarks"] = dash.loc[list(refined_pos.keys())].index.map(refined_pos)
        if refined_neg:
            dash.loc[list(refined_neg.keys()), "negative_remarks"] = dash.loc[list(refined_neg.keys())].index.map(refined_neg)
        if refined_sum:
            # concaténer l'ancien + nouveau
            for idx, txt in refined_sum.items():
                old = dash.at[idx, "summary"]
                dash.at[idx, "summary"] = (old + " | " + txt) if old else txt

# nettoyer colonnes internes
dash = dash.drop(columns=["_raw_for_llm"], errors="ignore")

# tri par final_score décroissant
dash = dash.sort_values(["final_score", "avg_token_conviction", "mentions"], ascending=[False, False, False])

# filtre UI par score
dash_view = dash[dash["final_score"] >= min_final_score].reset_index(drop=True)

st.subheader("📊 Memecoin Dashboard")
st.caption("Filtrage: tokens commençant par `$`, hors $BTC/$ETH/$SOL et hors montants ($2M, $120k...). Score final = 0.55*conviction moyenne + 0.35*groupes uniques + 0.10*mentions (toutes normalisées).")
st.dataframe(
    dash_view[[
        "token", "final_score", "avg_token_conviction", "mentions", "unique_groups",
        "positive_remarks", "negative_remarks", "red_flags", "summary"
    ]],
    use_container_width=True
)

# ---------- Onglet Top 5 par groupe ----------
st.subheader("🏆 Top 5 convictions par groupe (>= 5.6)")
THRESH = 5.6
df_tok = mx.merge(
    dash[["token", "final_score"]],
    on="token", how="left"
)
df_tok = df_tok[df_tok["final_score"] >= THRESH]

# calc score agrégé par groupe & token (moyenne des final_score ou max)
group_tok = (df_tok.groupby(["group", "token"])["final_score"]
             .mean().reset_index())
# top5 par groupe
group_ranked = group_tok.sort_values(["group", "final_score"], ascending=[True, False])
top_rows = []
for g, sub in group_ranked.groupby("group"):
    pick = sub.head(5).copy()
    # pour l'affichage "nom du groupe une seule fois"
    first = True
    for _, r in pick.iterrows():
        top_rows.append({
            "Groupe": g if first else "",
            "Token": r["token"],
            "Score final": round(r["final_score"], 2)
        })
        first = False

top_df = pd.DataFrame(top_rows)
st.dataframe(top_df, use_container_width=True)

# Petit export
st.download_button(
    "Télécharger le dashboard (CSV)",
    data=dash_view.to_csv(index=False).encode("utf-8"),
    file_name="memecoin_dashboard.csv",
    mime="text/csv"
)
