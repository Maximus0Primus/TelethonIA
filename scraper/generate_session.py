"""
One-time helper: authenticate with Telegram and output a StringSession.
Run locally, paste the result into GitHub Secrets as TELEGRAM_STRING_SESSION.

Usage:
    python scraper/generate_session.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

load_dotenv(Path(__file__).parent / ".env")

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]


def main():
    print("=== Telegram StringSession Generator ===")
    print("You will be prompted for your phone number and a verification code.\n")

    with TelegramClient(StringSession(), API_ID, API_HASH) as client:
        session_string = client.session.save()

    print("\n" + "=" * 60)
    print("Copy the entire string below into GitHub Secrets")
    print("as TELEGRAM_STRING_SESSION:")
    print("=" * 60)
    print(session_string)
    print("=" * 60)
    print("\nNEVER share this string — it grants full Telegram access.")


if __name__ == "__main__":
    main()
