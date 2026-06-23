# config/memory_store.py
"""
PostgreSQL-backed persistent memory stores.

MemoryStore      — researcher topic findings (cross-session research cache)
UserMemoryStore  — per-user interaction history (who asked what, enables
                   the agent to answer "who am I?" from conversation history)

Tables
  researcher_memory : session_id, topic, key_findings, summary, created_at
  user_memory       : user_id, session_id, user_query, created_at
"""

import asyncio
import json
import logging
from typing import Optional

import psycopg2
import psycopg2.extras

from config.settings import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS researcher_memory (
    id           SERIAL PRIMARY KEY,
    session_id   TEXT        NOT NULL,
    topic        TEXT        NOT NULL,
    key_findings JSONB       NOT NULL DEFAULT '[]',
    summary      TEXT,
    created_at   TIMESTAMP   NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rm_topic ON researcher_memory (topic);
"""


class MemoryStore:
    def __init__(self) -> None:
        self._dsn = settings.DATABASE_URL
        self._available = self._init_table()

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────
    def _connect(self):
        return psycopg2.connect(self._dsn)

    def _init_table(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            logger.info("[MemoryStore] PostgreSQL connected — researcher_memory table ready")
            return True
        except Exception as e:
            logger.warning(f"[MemoryStore] PostgreSQL unavailable — memory disabled: {e}")
            return False

    def _do_save(self, session_id: str, topic: str, key_findings: list, summary: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO researcher_memory (session_id, topic, key_findings, summary)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, topic, json.dumps(key_findings[:5]), summary[:1000]),
            )

    def _do_retrieve(self, topic: str, limit: int) -> list[dict]:
        keywords = [w for w in topic.split() if len(w) > 3][:3]
        conditions = " OR ".join(["topic ILIKE %s"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords] + [limit]

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT topic, key_findings, summary, created_at
                    FROM researcher_memory
                    WHERE {conditions}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    params,
                )
                return [dict(row) for row in cur.fetchall()]

    # ──────────────────────────────────────────────────────────────
    # Public async API  (wraps sync psycopg2 in a thread so the
    # event loop is never blocked)
    # ──────────────────────────────────────────────────────────────
    async def save(
        self,
        session_id: str,
        topic: str,
        key_findings: list,
        summary: str,
    ) -> None:
        if not self._available:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._do_save, session_id, topic, key_findings, summary
            )
            logger.info(f"[MemoryStore] Saved memory | topic='{topic}' | session={session_id}")
        except Exception as e:
            logger.warning(f"[MemoryStore] save() failed: {e}")

    async def retrieve(self, topic: str, limit: int = 3) -> list[dict]:
        if not self._available:
            return []
        try:
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, self._do_retrieve, topic, limit)
            logger.info(f"[MemoryStore] Retrieved {len(rows)} memories for: '{topic}'")
            return rows
        except Exception as e:
            logger.warning(f"[MemoryStore] retrieve() failed: {e}")
            return []


memory_store = MemoryStore()


# ─────────────────────────────────────────────────────────────────────────────
# User Memory Store — tracks what each user has asked across sessions
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_USER_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS user_memory (
    id         SERIAL    PRIMARY KEY,
    user_id    TEXT      NOT NULL,
    session_id TEXT      NOT NULL,
    user_query TEXT      NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_um_user_id ON user_memory (user_id);
"""


class UserMemoryStore:
    """
    Stores every query a user makes, keyed by user_id.
    Lets the orchestrator answer "who am I?" by replaying what the user
    has asked about across all past sessions.
    """

    def __init__(self) -> None:
        self._dsn = settings.DATABASE_URL
        self._available = self._init_table()

    def _connect(self):
        return psycopg2.connect(self._dsn)

    def _init_table(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(_CREATE_USER_MEMORY_TABLE)
            logger.info("[UserMemoryStore] user_memory table ready")
            return True
        except Exception as e:
            logger.warning(f"[UserMemoryStore] PostgreSQL unavailable — user memory disabled: {e}")
            return False

    def _do_record(self, user_id: str, session_id: str, user_query: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_memory (user_id, session_id, user_query) VALUES (%s, %s, %s)",
                (user_id, session_id, user_query),
            )

    def _do_get_history(self, user_id: str, limit: int) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT user_query, session_id, created_at
                    FROM user_memory
                    WHERE user_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    async def record(self, user_id: str, session_id: str, user_query: str) -> None:
        """Persist a user query (called at the start of every pipeline run)."""
        if not self._available:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._do_record, user_id, session_id, user_query)
            logger.info(f"[UserMemoryStore] Recorded query | user={user_id}")
        except Exception as e:
            logger.warning(f"[UserMemoryStore] record() failed: {e}")

    async def get_history(self, user_id: str, limit: int = 20) -> list[dict]:
        """Return the user's past queries in chronological order."""
        if not self._available:
            return []
        try:
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, self._do_get_history, user_id, limit)
            logger.info(f"[UserMemoryStore] Retrieved {len(rows)} history entries for user={user_id}")
            return rows
        except Exception as e:
            logger.warning(f"[UserMemoryStore] get_history() failed: {e}")
            return []

    def _do_clear(self, user_id: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM user_memory WHERE user_id = %s", (user_id,))
            return cur.rowcount

    async def clear(self, user_id: str) -> int:
        """Delete all history for a user. Returns the number of rows deleted."""
        if not self._available:
            return 0
        try:
            loop = asyncio.get_event_loop()
            deleted = await loop.run_in_executor(None, self._do_clear, user_id)
            logger.info(f"[UserMemoryStore] Cleared {deleted} rows for user={user_id}")
            return deleted
        except Exception as e:
            logger.warning(f"[UserMemoryStore] clear() failed: {e}")
            return 0


user_memory_store = UserMemoryStore()
