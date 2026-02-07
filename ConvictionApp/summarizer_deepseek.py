# -*- coding: utf-8 -*-
"""
Client DeepSeek (Chat Completions) pour résumés *structurés et pédagogiques*.
Sortie NORMALISÉE pour un public débutant (« normies ») :
1. Description
2. Catalyseurs
3. Risques
4. Sentiment global
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional
import requests

# ⚠️ Clé via variable d'env si dispo, sinon fallback (tu peux laisser tel quel)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c4d430a8afd947298ee4098d7a93059e")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"
SESSION_TIMEOUT = 15  # s

HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}

# ======== PROMPTS ========
SYSTEM_PROMPT = (
    "Tu es un ANALYSTE CRYPTO chargé d'expliquer des discussions Telegram à propos d'un token"
    " pour un public DÉBUTANT. Sois pédagogique, concret et synthétique.\n\n"
    "RÈGLES DE STYLE :\n"
    "- Écris en français simple, sans jargon. Si un terme technique apparaît, explique-le en 4-6 mots.\n"
    "- AUCUN lien, AUCUN pseudo @, pas d'emoji.\n"
    "- Sortie OBLIGATOIREMENT dans CE FORMAT (4 sections numérotées) :\n"
    "1. **Description** : type de token (meme coin, utilitaire, DeFi...) et ce qu'il fait en UNE phrase.\n"
    "2. **Catalyseurs** : 2-4 puces max (événements, annonces, influenceurs, burn, listings, partenariats, roadmap...).\n"
    "3. **Risques** : 2-4 puces max (rug pull, faible liquidité, dépendance à un influenceur, tokenomics douteuses...).\n"
    "4. **Sentiment global** : Optimiste / Prudent / Négatif + courte justification.\n"
    "- Pas de spéculation de prix chiffrée, pas de conseils d'investissement.\n"
)

USER_TEMPLATE = (
    "Token : {symbol}\n"
    "Contexte : messages Telegram bruts ci-dessous (bruit possible, liens supprimés).\n"
    "Consigne : respecte STRICTEMENT le format demandé (4 sections).\n\n"
    "=== MESSAGES SÉLECTIONNÉS (max 30) ===\n"
    "{messages}\n"
)

# ======== HELPERS ========

def _truncate_text(txt: str, max_chars: int = 6000) -> str:
    txt = (txt or "").strip()
    if len(txt) <= max_chars:
        return txt
    head = txt[: max_chars - 500]
    tail = txt[-400:]
    return head + "\n[...]\n" + tail


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        return s.strip("`\n ")
    return s


def _normalize_bullets(s: str) -> str:
    # Uniformise les puces dans sections 2 et 3
    s = re.sub(r"\n\s*[-•]\s*", "\n- ", s)
    return s


def _enforce_structure(text: str) -> str:
    """Tente d'imposer la structure 1..4 si le modèle dévie légèrement."""
    t = _strip_code_fences(text)
    t = _normalize_bullets(t)

    # Si déjà bien formatté, renvoie tel quel
    if re.search(r"\n?1\.\s*\*\*Description\*\*", t, re.I) and \
       re.search(r"\n?4\.\s*\*\*Sentiment global\*\*", t, re.I):
        return t

    # Sinon, essaie de détecter lignes et reconditionner sommairement
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines:
        return (
            "1. **Description** : —\n"
            "2. **Catalyseurs** : —\n"
            "3. **Risques** : —\n"
            "4. **Sentiment global** : —"
        )

    # Heuristique simple :
    desc = lines[0]
    # catalyseurs/risques : prends 2-3 puces si présentes
    cats = [l for l in lines[1:6] if len(l) > 8][:3]
    risks = [l for l in lines[6:11] if len(l) > 8][:3]
    senti = lines[-1]

    def _bullets(xs):
        return "\n".join([f"- {x.rstrip('.')}" for x in xs]) if xs else "- —"

    return (
        f"1. **Description** : {desc}\n"
        f"2. **Catalyseurs** :\n{_bullets(cats)}\n"
        f"3. **Risques** :\n{_bullets(risks)}\n"
        f"4. **Sentiment global** : {senti}"
    )


# ======== API CALL ========

def summarize_with_deepseek(text: str, max_tokens: int = 160, temperature: float = 0.2,
                             symbol: Optional[str] = None) -> str:
    """
    Résume `text` (déjà pré‑sélectionné côté utils) en 4 sections claires.
    - `symbol` : le ticker du token (ex: "STUPID"), utilisé dans le prompt utilisateur.
    """
    text = (text or "").strip()
    if not text:
        return (
            "1. **Description** : —\n"
            "2. **Catalyseurs** : —\n"
            "3. **Risques** : —\n"
            "4. **Sentiment global** : —"
        )

    user_prompt = USER_TEMPLATE.format(
        symbol=(symbol or "?"),
        messages=_truncate_text(text, 3500),
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    for attempt in range(2):
        try:
            r = requests.post(API_URL, json=payload, headers=HEADERS, timeout=SESSION_TIMEOUT)
            if r.status_code == 429 and attempt == 0:
                time.sleep(0.6); continue
            r.raise_for_status()
            data = r.json() or {}
            choices = data.get("choices") or []
            if choices and choices[0].get("message") and choices[0]["message"].get("content"):
                raw = str(choices[0]["message"]["content"]).strip()
                return _enforce_structure(raw)
            return (
                "1. **Description** : —\n"
                "2. **Catalyseurs** : —\n"
                "3. **Risques** : —\n"
                "4. **Sentiment global** : —"
            )
        except requests.HTTPError as e:
            return f"[Erreur DeepSeek] {e}"
        except Exception as e:
            err = str(e)
            if attempt == 0 and ("Read timed out" in err or "timeout" in err.lower()):
                continue
            return f"[Erreur DeepSeek] {err}"

    return (
        "1. **Description** : —\n"
        "2. **Catalyseurs** : —\n"
        "3. **Risques** : —\n"
        "4. **Sentiment global** : —"
    )
