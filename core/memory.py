"""SQLite-backed short- and long-term memory.

Short-term (within a session): the last N messages for a session_id, read
straight from the `messages` table -- no separate in-process buffer, so
memory survives a backend restart and works the same whether the client is
Streamlit (reruns the whole script every turn) or a notebook. Once a session
passes SUMMARIZE_AFTER_TURNS, older turns are collapsed into a rolling
`session_summaries` row via one small LLM call, keeping the per-request
token cost bounded regardless of how long the conversation runs.

Long-term (across sessions): `users`, `preferences`, `car_interactions`
(viewed/liked/booked), and `leads` persist per user_id. On session start the
caller builds a short natural-language summary from these tables (see
build_user_context) to inject into the system prompt -- never raw rows, to
keep the prompt small.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from core.llm import SMALL_MODEL, call_llm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "memory.db"
LEADS_CSV_PATH = DATA_DIR / "leads.csv"

SUMMARIZE_AFTER_TURNS = 12  # messages (user+assistant combined), not conversational turns
KEEP_RECENT_MESSAGES = 6


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, key)
            );

            CREATE TABLE IF NOT EXISTS car_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                car_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,  -- viewed | liked | booked
                ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                price_range TEXT,
                needs TEXT,
                car_id INTEGER,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filters_json TEXT NOT NULL,
                last_shown_car_ids_json TEXT NOT NULL,
                lead_in_progress_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                car_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


init_db()


# --------------------------------------------------------------------------
# Users & sessions
# --------------------------------------------------------------------------

def get_or_create_user(user_id: str, name: str | None = None) -> dict:
    """user_id is whatever the client sends to identify a returning person
    (e.g. a typed name/handle). Returns the user row, updating name if a new
    one was supplied."""
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name or user_id, _now()),
            )
            return {"user_id": user_id, "name": name or user_id, "created_at": _now(), "is_returning": False}
        if name and name != row["name"]:
            conn.execute("UPDATE users SET name = ? WHERE user_id = ?", (name, user_id))
        return {**dict(row), "name": name or row["name"], "is_returning": True}


def start_or_touch_session(session_id: str, user_id: str) -> None:
    with _conn() as conn:
        row = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        now = _now()
        if row is None:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, created_at, last_active_at) VALUES (?,?,?,?)",
                (session_id, user_id, now, now),
            )
        else:
            conn.execute("UPDATE sessions SET last_active_at = ? WHERE session_id = ?", (now, session_id))


# --------------------------------------------------------------------------
# Messages & short-term memory
# --------------------------------------------------------------------------

def save_message(session_id: str, user_id: str, role: str, content: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, user_id, role, content, ts) VALUES (?,?,?,?,?)",
            (session_id, user_id, role, content, _now()),
        )


def get_recent_messages(session_id: str, limit: int = KEEP_RECENT_MESSAGES) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, ts FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_session_summary(session_id: str) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,)).fetchone()
    return row["summary"] if row else None


def _set_session_summary(session_id: str, user_id: str, summary: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO session_summaries (session_id, user_id, summary, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at""",
            (session_id, user_id, summary, _now()),
        )


def maybe_summarize_session(session_id: str, user_id: str) -> None:
    """Once a session has more than SUMMARIZE_AFTER_TURNS messages, fold
    everything except the most recent KEEP_RECENT_MESSAGES into a running
    summary. Keeps per-request prompt size bounded on long conversations
    without losing earlier context entirely."""
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM messages WHERE session_id = ?", (session_id,)).fetchone()["c"]
        if total <= SUMMARIZE_AFTER_TURNS:
            return
        to_fold = conn.execute(
            """SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC
               LIMIT ?""",
            (session_id, total - KEEP_RECENT_MESSAGES),
        ).fetchall()

    if not to_fold:
        return

    prior_summary = get_session_summary(session_id)
    transcript = "\n".join(f"{r['role']}: {r['content']}" for r in to_fold)
    prompt = (
        (f"Prior summary: {prior_summary}\n\n" if prior_summary else "")
        + f"Conversation so far:\n{transcript}\n\n"
        + "Summarize the car-shopping-relevant facts in 2-3 short sentences: "
        + "what the user is looking for, cars discussed/shown, any preferences "
        + "stated. Skip pleasantries."
    )
    try:
        response = call_llm(
            messages=[{"role": "user", "content": prompt}],
            component="session_summarization",
            model=SMALL_MODEL,
            max_tokens=200,
        )
        summary = response.choices[0].message.content.strip()
    except Exception:  # noqa: BLE001 - summarization is best-effort
        return

    _set_session_summary(session_id, user_id, summary)

    # Delete the folded messages, keep the recent tail intact.
    with _conn() as conn:
        ids_to_delete = conn.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (session_id, len(to_fold)),
        ).fetchall()
        conn.executemany("DELETE FROM messages WHERE id = ?", [(r["id"],) for r in ids_to_delete])


# --------------------------------------------------------------------------
# Per-session working state (carried-forward filters, last-shown cars,
# in-progress lead) -- this is what lets a follow-up like "is there a
# warranty on it?" resolve against "that first Honda" without the user
# repeating themselves, and lets "under $20k" survive into the next turn.
# --------------------------------------------------------------------------

_DEFAULT_STATE = {"filters": {}, "last_shown_car_ids": [], "lead_in_progress": {}}


def get_session_state(session_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT filters_json, last_shown_car_ids_json, lead_in_progress_json FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return dict(_DEFAULT_STATE)
    return {
        "filters": json.loads(row["filters_json"]),
        "last_shown_car_ids": json.loads(row["last_shown_car_ids_json"]),
        "lead_in_progress": json.loads(row["lead_in_progress_json"]),
    }


def set_session_state(session_id: str, user_id: str, state: dict) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO session_state
                   (session_id, user_id, filters_json, last_shown_car_ids_json, lead_in_progress_json, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                   filters_json = excluded.filters_json,
                   last_shown_car_ids_json = excluded.last_shown_car_ids_json,
                   lead_in_progress_json = excluded.lead_in_progress_json,
                   updated_at = excluded.updated_at""",
            (
                session_id, user_id,
                json.dumps(state.get("filters", {})),
                json.dumps(state.get("last_shown_car_ids", [])),
                json.dumps(state.get("lead_in_progress", {})),
                _now(),
            ),
        )


# --------------------------------------------------------------------------
# Preferences & interactions (long-term)
# --------------------------------------------------------------------------

def save_preference(user_id: str, key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?,?,?,?)
               ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
            (user_id, key, value, _now()),
        )


def get_preferences(user_id: str) -> dict:
    with _conn() as conn:
        rows = conn.execute("SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)).fetchall()
    return {r["key"]: r["value"] for r in rows}


def record_car_interaction(user_id: str, car_id: int, interaction_type: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO car_interactions (user_id, car_id, interaction_type, ts) VALUES (?,?,?,?)",
            (user_id, car_id, interaction_type, _now()),
        )


def get_recent_car_interactions(user_id: str, limit: int = 5) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT car_id, interaction_type, ts FROM car_interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Leads (also mirrored to a CSV per the assessment brief)
# --------------------------------------------------------------------------

def save_lead(user_id: str, price_range: str | None, needs: str | None, car_id: int | None) -> dict:
    created_at = _now()
    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO leads (user_id, price_range, needs, car_id, created_at) VALUES (?,?,?,?,?)",
            (user_id, price_range, needs, car_id, created_at),
        )
        lead_id = cursor.lastrowid

    _append_lead_csv(
        {
            "lead_id": lead_id,
            "user_id": user_id,
            "price_range": price_range or "",
            "needs": needs or "",
            "car_id": car_id if car_id is not None else "",
            "created_at": created_at,
        }
    )
    return {"lead_id": lead_id, "user_id": user_id, "price_range": price_range, "needs": needs, "car_id": car_id, "created_at": created_at}


def _append_lead_csv(row: dict) -> None:
    import csv

    is_new = not LEADS_CSV_PATH.exists()
    with open(LEADS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_leads(user_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM leads WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Bookings
# --------------------------------------------------------------------------

def save_booking(user_id: str, session_id: str, car_id: int, day: str, time_slot: str) -> dict:
    created_at = _now()
    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO bookings (user_id, session_id, car_id, day, time, created_at) VALUES (?,?,?,?,?,?)",
            (user_id, session_id, car_id, day, time_slot, created_at),
        )
        booking_id = cursor.lastrowid
    return {"booking_id": booking_id, "user_id": user_id, "car_id": car_id, "day": day, "time": time_slot, "created_at": created_at}


def get_bookings(user_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM bookings WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Long-term context summary for the system prompt
# --------------------------------------------------------------------------

def build_user_context(user_id: str, is_returning: bool) -> str | None:
    """A short natural-language blurb to inject into the system prompt for
    a returning user -- deliberately a summary, not raw preference/lead
    rows, to keep prompt tokens bounded."""
    if not is_returning:
        return None

    prefs = get_preferences(user_id)
    interactions = get_recent_car_interactions(user_id, limit=3)
    leads = get_leads(user_id)

    if not prefs and not interactions and not leads:
        return None

    from core.retrieval import get_car  # local import: avoid a memory<->retrieval import cycle

    bits = []
    if prefs:
        pref_str = "; ".join(f"{k}: {v}" for k, v in prefs.items())
        bits.append(f"Stated preferences -- {pref_str}.")

    if interactions:
        car_bits = []
        for interaction in interactions:
            car = get_car(interaction["car_id"])
            if car:
                car_bits.append(
                    f"{car.get('year')} {car.get('make')} {car.get('model')} ({interaction['interaction_type']})"
                )
        if car_bits:
            bits.append("Recently " + "; ".join(car_bits) + ".")

    if leads:
        latest = leads[0]
        bits.append(
            f"Last lead recorded: budget {latest['price_range'] or 'unspecified'}, needs: {latest['needs'] or 'unspecified'}."
        )

    return " ".join(bits) if bits else None
