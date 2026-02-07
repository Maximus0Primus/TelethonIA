# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import plotly.express as px

from utils import (
    ensure_session_dataset,
    ui_dataset_loader,
    ui_status_banner,
    parse_messages_json,
    add_sentiment,
    explode_tokens,
    available_tokens,
    summarizer_available,
    load_many_jsons,
)

st.set_page_config(layout="wide")
st.title("⚙️ Configuration & Aperçu")
ensure_session_dataset()  # <-- garde le dataset en mémoire entre les pages

# -----------------------------
# Defaults
# -----------------------------
def ensure_defaults():
    ss = st.session_state
    ss.setdefault("period", "24h")
    ss.setdefault("use_custom_period", False)
    ss.setdefault("custom_start_date", None)
    ss.setdefault("custom_start_time", None)
    ss.setdefault("custom_end_date", None)
    ss.setdefault("custom_end_time", None)

    ss.setdefault("use_hf", False)
    ss.setdefault("w_hf", 0.50)
    ss.setdefault("w_vader", 0.35)
    ss.setdefault("w_crypto", 0.15)
    ss.setdefault("gain_sent", 1.30)

    ss.setdefault("rule_weight", 1.0)
    ss.setdefault("group_alpha", 1.0)
    ss.setdefault("alias_no_dollar", True)
    ss.setdefault("enable_summarizer", False)
ensure_defaults()

def _period_popover():
    st.session_state["period"] = st.radio(
        "Fenêtre rapide",
        ["2h","6h","12h","24h","48h","Tout"],
        index=["2h","6h","12h","24h","48h","Tout"].index(st.session_state["period"]),
        horizontal=True,
    )
    st.session_state["use_custom_period"] = st.checkbox("Utiliser une plage personnalisée")
    if st.session_state["use_custom_period"]:
        c1, c2 = st.columns(2)
        st.session_state["custom_start_date"] = c1.date_input("Date début (Europe/Paris)")
        st.session_state["custom_start_time"] = c1.time_input("Heure début")
        st.session_state["custom_end_date"]   = c2.date_input("Date fin (Europe/Paris)")
        st.session_state["custom_end_time"]   = c2.time_input("Heure fin")

def _apply_period_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    if st.session_state.get("use_custom_period"):
        try:
            sdate = st.session_state.get("custom_start_date")
            stime = st.session_state.get("custom_start_time")
            edate = st.session_state.get("custom_end_date")
            etime = st.session_state.get("custom_end_time")
            if sdate and stime and edate and etime:
                start = pd.Timestamp.combine(sdate, stime).tz_localize("Europe/Paris").tz_convert(None)
                end   = pd.Timestamp.combine(edate, etime).tz_localize("Europe/Paris").tz_convert(None)
                return df[(df["date"]>=start) & (df["date"]<=end)]
        except Exception:
            pass
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    if st.session_state["period"] == "Tout":
        return df
    hours = int(st.session_state["period"].replace("h",""))
    return df[df["date"] >= (now - pd.Timedelta(hours=hours))]

# -----------------------------
# Sidebar (unifiée)
# -----------------------------
st.sidebar.title("⚙️ Configuration")

# Loader unifié: multi-JSON + reset + compteur
raw_all = ui_dataset_loader(page_key="app")

with st.sidebar.expander("Période d'analyse", expanded=True):
    with st.popover("Choisir la période précisément"):
        _period_popover()

with st.sidebar.expander("Sentiment – canaux & poids (local)", expanded=True):
    st.session_state["use_hf"] = st.checkbox("Utiliser modèle local (HF)", value=st.session_state["use_hf"])
    st.session_state["w_hf"] = st.slider("Poids modèle local (HF)", 0.0, 1.0, st.session_state["w_hf"], 0.05)
    st.session_state["w_vader"]  = st.slider("Poids VADER", 0.0, 1.0, st.session_state["w_vader"], 0.05)
    st.session_state["w_crypto"] = st.slider("Poids Lexique crypto", 0.0, 1.0, st.session_state["w_crypto"], 0.05)
    st.session_state["gain_sent"] = st.slider("Gain dynamique (×)", 0.5, 2.0, float(st.session_state["gain_sent"]), 0.05)
    st.session_state["enable_summarizer"] = st.checkbox("Activer résumeur (DistilBART)", value=st.session_state["enable_summarizer"])

with st.sidebar.expander("Ajustements & pondération", expanded=True):
    st.session_state["rule_weight"] = st.slider("Poids règles/boosts lexicaux", 0.0, 1.0, st.session_state["rule_weight"], 0.05)
    st.session_state["group_alpha"] = st.slider("Poids conviction de groupe", 0.0, 2.0, st.session_state["group_alpha"], 0.1)
    st.session_state["alias_no_dollar"] = st.checkbox("Détecter alias sans $ (strict)", value=st.session_state["alias_no_dollar"])

# -----------------------------
# Bannière de statut dataset
# -----------------------------
ui_status_banner(compact=True)

# -----------------------------
# Data
# -----------------------------
if raw_all.empty:
    st.info("Importe au moins un JSON.")
    st.stop()

df0 = _apply_period_filter(raw_all)

with st.spinner("Calcul du sentiment..."):
    df = add_sentiment(
        df0,
        use_hf=st.session_state["use_hf"],
        w_vader=st.session_state["w_vader"],
        w_crypto=st.session_state["w_crypto"],
        w_hf=st.session_state["w_hf"],
        rule_weight=st.session_state["rule_weight"],
        group_weight_alpha=st.session_state["group_alpha"],
        alias_no_dollar=st.session_state["alias_no_dollar"],
        gain=float(st.session_state["gain_sent"]),
    )

# Indicateur résumeur
ready = summarizer_available()
if st.session_state["enable_summarizer"] and ready:
    st.markdown("🟢 **Résumeur DistilBART actif**")
elif st.session_state["enable_summarizer"] and ready is False:
    st.markdown("🟠 **Résumeur demandé mais indisponible** — fallback extractif.")
else:
    st.markdown("🔴 **Résumeur inactif**")

# -----------------------------
# Aperçu des messages (robuste)
# -----------------------------
st.subheader("Aperçu des messages")

# Garde-fous colonnes pour éviter KeyError quand le dataset est partiel
needed_cols = ["date","group","tokens","sentiment","sentiment_hf","w_sentiment","text","remark"]
for c in needed_cols:
    if c not in df.columns:
        if c == "tokens":
            df[c] = [[] for _ in range(len(df))]
        elif c in ("text","remark","group"):
            df[c] = ""
        else:
            df[c] = pd.Series(dtype=float)

st.dataframe(
    df[needed_cols].sort_values("date", ascending=False),
    use_container_width=True
)
