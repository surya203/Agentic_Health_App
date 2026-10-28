"""Read and write the hospital profile and chat transcript in Supabase."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Values taken from documents/hospital.txt. Inserted once when the table is empty.
HOSPITAL_PROFILE = {
    "id": 1,
    "name": "Jaiiii Surya Medical Center",
    "address": "Jaiiii Surya Medical Center, 123 Main Street, Anytown, USA",
    "phone": "123-456-7890",
    "email": "info@jaiiii-surya-medical-center.com",
    "website": "https://jaiiii-surya-medical-center.com",
    "logo_url": "https://jaiiii-surya-medical-center.com/logo.png",
    "colors": ["#000000", "#FFFFFF"],
    "fonts": ["Arial", "Helvetica"],
    "font_sizes": ["12px", "14px"],
    "font_weights": ["normal", "bold"],
    "font_styles": ["normal", "italic"],
    "font_variants": ["normal", "small-caps"],
}

NOT_CONFIGURED = (
    "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to .env, "
    "run sql/schema.sql in the Supabase SQL editor, then restart the app."
)
SCHEMA_HINT = (
    "Supabase could not read or write the tables. "
    "Run sql/schema.sql in the Supabase SQL editor, then restart the app."
)
KEY_HINT = "Supabase rejected the API key."

_client = None
_client_token: tuple[str, str] | None = None


class SupabaseStoreError(Exception):
    """A fixed message safe to show in the interface."""


def configured() -> bool:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    return bool(url and key)


def get_client():
    global _client, _client_token
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    token = (url, key)
    if _client is None or _client_token != token:
        from supabase import create_client

        _client = create_client(url, key)
        _client_token = token
    return _client


def _run(operation):
    client = get_client()
    if client is None:
        raise SupabaseStoreError(NOT_CONFIGURED)
    try:
        return operation(client)
    except SupabaseStoreError:
        raise
    except Exception as exc:
        text = str(exc).lower()
        if "401" in text or "invalid api key" in text or "jwt" in text:
            raise SupabaseStoreError(KEY_HINT) from exc
        raise SupabaseStoreError(SCHEMA_HINT) from exc


def load_hospital_profile() -> dict:
    """Return the stored profile, inserting the sample row when the table is empty."""

    def operation(client):
        found = (
            client.table("hospital_profile")
            .select("*")
            .eq("id", 1)
            .limit(1)
            .execute()
        )
        rows = found.data or []
        if rows:
            return rows[0]
        inserted = client.table("hospital_profile").insert(HOSPITAL_PROFILE).execute()
        created = inserted.data or []
        return created[0] if created else dict(HOSPITAL_PROFILE)

    return _run(operation)


def load_chat_messages(limit: int = 50) -> list[dict]:
    """Return the newest stored turns in chronological order."""

    def operation(client):
        found = (
            client.table("chat_messages")
            .select("role, content")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(reversed(found.data or []))
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    return _run(operation)


def clear_chat_messages() -> None:
    """Delete every stored chat turn."""

    def operation(client):
        client.table("chat_messages").delete().neq("role", "").execute()

    _run(operation)


def save_chat_message(role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise SupabaseStoreError("Chat role must be user or assistant.")

    def operation(client):
        client.table("chat_messages").insert(
            {"role": role, "content": content}
        ).execute()

    _run(operation)


