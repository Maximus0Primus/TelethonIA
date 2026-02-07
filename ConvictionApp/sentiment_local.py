# -*- coding: utf-8 -*-
"""
Sentiment local (remplacement HF) — CryptoBERT + heuristiques Telegram.
Expose: batch_scores(List[str]) -> List[float] in [-1, 1]
"""
from __future__ import annotations
from typing import List, Dict
import re
import math

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import emoji

# --------- Modèle principal (CryptoBERT) ----------
_MODEL_ID = "ElKulako/cryptobert"  # modèle crypto-social (Twitter/Reddit/StockTwits/Telegram)
_tok = None
_mod = None
_pipe: TextClassificationPipeline | None = None

# en haut du fichier
_HF_STRETCH = 1.8  # 1.4..2.2 selon ton ressenti

def _stretch(x: float, k: float = _HF_STRETCH) -> float:
    # préserve le signe, étire autour de 0, borne [-1,1]
    x = max(-0.999, min(0.999, float(x)))
    return float(np.tanh(k * np.arctanh(x)))


def _ensure_pipe():
    global _tok, _mod, _pipe
    if _pipe is not None:
        return _pipe
    _tok = AutoTokenizer.from_pretrained(_MODEL_ID, use_fast=True)
    _mod = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
    _mod.eval()
    _pipe = TextClassificationPipeline(
        model=_mod,
        tokenizer=_tok,
        device=-1,               # CPU
        top_k=None,              # on récupère tous les labels
        truncation=True          # tronque automatiquement
    )
    return _pipe

# --------- Heuristiques Telegram (intensité / emojis / sarcasme léger) ----------
_EMOJI_WEIGHTS: Dict[str, float] = {
    "🚀": 0.22, "🧨": 0.10, "🔥": 0.10, "💎": 0.12, "🌕": 0.20, "✨": 0.06, "✅": 0.10,
    "💀": -0.22, "☠️": -0.22, "🪦": -0.18, "🧻": -0.12, "⚠️": -0.12, "❗": 0.06, "❕": 0.04,
    "🤡": -0.16, "🤔": -0.04, "😬": -0.10, "😡": -0.16, "🤯": -0.08
}
_POS_HINTS = ("to the moon", "moonshot", "send it", "parabolic", "ath", "bullish", "pump", "lfg", "gm")
_NEG_HINTS = ("rug", "rugged", "honeypot", "scam", "dump", "rekt", "bearish", "dead", "exit liquidity")

def _emoji_score(text: str) -> float:
    if not text:
        return 0.0
    s = 0.0
    for ch in text:
        if ch in _EMOJI_WEIGHTS:
            s += _EMOJI_WEIGHTS[ch]
    # cap contribution totale des emojis pour éviter l'emballement
    return float(max(-0.6, min(0.6, s)))

def _intensity_boost(text: str, base: float) -> float:
    """
    Ajuste l'intensité selon MAJUSCULES / !!! / hints lexicaux.
    Effet modéré (±15% max) pour éviter de surcorriger.
    """
    if not text:
        return base
    t = text.strip()
    # Majuscules : ratio de lettres majuscules
    letters = [c for c in t if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in letters) / max(1, len(letters))) if letters else 0.0
    # Ponctuation forte
    bangs = t.count("!") + t.count("❗")
    # Hints
    tl = t.lower()
    pos_hit = any(k in tl for k in _POS_HINTS)
    neg_hit = any(k in tl for k in _NEG_HINTS)

    # facteur multiplicatif
    mult = 1.0
    if upper_ratio >= 0.45:
        mult += 0.08
    mult += min(0.12, 0.03 * min(6, bangs))
    if (base >= 0 and pos_hit) or (base <= 0 and neg_hit):
        mult += 0.05

    mult = float(max(0.85, min(1.15, mult)))
    out = base * mult

    # Sarcasme léger : 😂🤣 + mots très négatifs -> atténue un peu le négatif
    if any(x in tl for x in ("lol", "lmao")) or any(e in t for e in ("😂", "🤣")):
        if neg_hit and out < 0:
            out = out * 0.85
    return float(max(-1.0, min(1.0, out)))

def _map_logits_to_score(labels_scores: List[Dict[str, float]]) -> float:
    """
    labels_scores = [{'label': 'Bullish', 'score': 0.7}, {'label':'Neutral','score':0.2}, {'label':'Bearish','score':0.1}]
    Retourne un score [-1,1] ~ (p_pos - p_neg).
    Gestion robuste si labels = LABEL_0/1/2.
    """
    if not labels_scores:
        return 0.0
    # Normalise noms
    pos = neg = neu = 0.0
    for d in labels_scores:
        lab = (d.get("label") or "").lower()
        sc  = float(d.get("score", 0.0))
        if "bull" in lab or "pos" in lab:
            pos += sc
        elif "bear" in lab or "neg" in lab:
            neg += sc
        elif "neu" in lab:
            neu += sc
        else:
            # fallback si labels anonymes : on tente tri par score, extrêmes = pos/neg
            pass
    if pos == neg == 0.0:
        # Fallback générique : prend max comme extrême, min comme opposé si 3 classes
        ls = sorted(labels_scores, key=lambda x: x.get("score", 0.0), reverse=True)
        if len(ls) >= 2:
            pos = ls[0]["score"]
            neg = ls[-1]["score"]
        else:
            pos = ls[0]["score"]

    score = float(pos - neg)
    return max(-1.0, min(1.0, score))

# --------- API principale ----------
def batch_scores(texts: List[str], batch_size: int = 16) -> List[float]:
    """
    Retourne un score par texte dans [-1, 1].
    Pipeline: CryptoBERT (p_pos - p_neg) -> + emoji_offset -> * intensity_boost -> clip [-1,1]
    """
    if not texts:
        return []
    pipe = _ensure_pipe()
    # Pipeline HF accepte batch_size en paramètre d'appel
    outputs = pipe(texts, batch_size=batch_size, truncation=True)
    # HF renvoie une liste de listes (scores pour chaque label)
    scores = []
    for text, ls in zip(texts, outputs):
        s_ml = _map_logits_to_score(ls)
        s_ml = _stretch(s_ml)  # étire la sortie CryptoBERT
        s_emoji = _emoji_score(text)
        s = s_ml + s_emoji
        # intensity / sarcasme
        s = _intensity_boost(text, s)
        # clamp
        s = float(max(-1.0, min(1.0, s)))
        scores.append(s)
    return scores
