"""
memory/database.py

Async SQLite database for persisting detection events.

Schema
──────
detections
  id               INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp        TEXT    NOT NULL  (ISO-8601)
  disease          TEXT    NOT NULL
  confidence       REAL    NOT NULL
  severity         TEXT    NOT NULL
  image_path       TEXT              (relative path inside reports/)
  treatment_json   TEXT              (JSON blob from OpenRouter)
  location_tag     TEXT              (optional GPS / grid label)
  created_at       TEXT    DEFAULT (datetime('now'))

sessions
  id               INTEGER PRIMARY KEY AUTOINCREMENT
  start_time       TEXT    NOT NULL
  end_time         TEXT
  total_detections INTEGER DEFAULT 0
  summary          TEXT
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from config import DB_PATH

logger = logging.getLogger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    disease         TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    severity        TEXT    NOT NULL,
    image_path      TEXT,
    treatment_json  TEXT,
    location_tag    TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time       TEXT NOT NULL,
    end_time         TEXT,
    total_detections INTEGER DEFAULT 0,
    summary          TEXT
);

CREATE INDEX IF NOT EXISTS idx_det_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_det_disease   ON detections(disease);
"""


class RoverDatabase:
    """
    Async SQLite wrapper for rover data.

    Usage::

        db = RoverDatabase()
        await db.init()
        det_id = await db.save_detection(...)
        await db.close()
    """

    def __init__(self) -> None:
        self._db: Optional[aiosqlite.Connection] = None
        self._current_session_id: Optional[int]  = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Open connection and ensure schema exists."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(DB_PATH))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_DDL)
        await self._db.commit()
        logger.info("Database ready: %s", DB_PATH)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            logger.info("Database closed.")

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def start_session(self) -> int:
        """Create a new inspection session record and return its id."""
        ts = datetime.utcnow().isoformat()
        async with self._db.execute(
            "INSERT INTO sessions (start_time) VALUES (?)", (ts,)
        ) as cur:
            self._current_session_id = cur.lastrowid
        await self._db.commit()
        logger.info("Session %d started.", self._current_session_id)
        return self._current_session_id  # type: ignore[return-value]

    async def end_session(self, summary: str = "") -> None:
        if not self._current_session_id:
            return
        ts    = datetime.utcnow().isoformat()
        count = await self._count_session_detections(self._current_session_id)
        await self._db.execute(
            """UPDATE sessions
               SET end_time=?, total_detections=?, summary=?
               WHERE id=?""",
            (ts, count, summary, self._current_session_id),
        )
        await self._db.commit()
        logger.info("Session %d ended (%d detections).",
                    self._current_session_id, count)

    async def _count_session_detections(self, session_id: int) -> int:
        # Simple approximation – count detections after session start
        async with self._db.execute(
            "SELECT start_time FROM sessions WHERE id=?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return 0
        async with self._db.execute(
            "SELECT COUNT(*) FROM detections WHERE timestamp >= ?",
            (row["start_time"],),
        ) as cur:
            result = await cur.fetchone()
        return result[0] if result else 0

    # ── Detections ────────────────────────────────────────────────────────────

    async def save_detection(
        self,
        disease:        str,
        confidence:     float,
        severity:       str,
        image_path:     str        = "",
        treatment:      Dict[str, Any] = None,   # type: ignore[assignment]
        location_tag:   str        = "",
    ) -> int:
        """
        Persist a disease detection event.

        Returns the row id.
        """
        ts             = datetime.utcnow().isoformat()
        treatment_json = json.dumps(treatment) if treatment else None

        async with self._db.execute(
            """INSERT INTO detections
               (timestamp, disease, confidence, severity,
                image_path, treatment_json, location_tag)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, disease, confidence, severity,
             image_path, treatment_json, location_tag),
        ) as cur:
            row_id = cur.lastrowid

        await self._db.commit()
        logger.debug("Detection saved: id=%d  disease=%s  conf=%.2f",
                     row_id, disease, confidence)
        return row_id  # type: ignore[return-value]

    async def get_recent_detections(
        self, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return the most recent detections as list of dicts."""
        async with self._db.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_session_detections(self) -> List[Dict[str, Any]]:
        """Return detections from the current session."""
        if not self._current_session_id:
            return []
        async with self._db.execute(
            "SELECT start_time FROM sessions WHERE id=?",
            (self._current_session_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return []
        async with self._db.execute(
            "SELECT * FROM detections WHERE timestamp >= ? ORDER BY id",
            (row["start_time"],),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_disease_summary(self) -> Dict[str, int]:
        """Return {disease_name: count} for all recorded detections."""
        async with self._db.execute(
            "SELECT disease, COUNT(*) as cnt FROM detections GROUP BY disease"
        ) as cur:
            rows = await cur.fetchall()
        return {r["disease"]: r["cnt"] for r in rows}
