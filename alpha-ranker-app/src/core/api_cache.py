"""
SQLite cache for API request results.

Stores responses by (source, cache_key) with TTL. Used by data.py and model FMP
to avoid re-fetching when data is still fresh.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "db" / "api_cache.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            source TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            data TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (source, cache_key)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_api_cache_fetched ON api_cache(fetched_at)")
    c.commit()
    return c


def get(source: str, cache_key: str, max_age_hours: float = 24):
    """
    Return cached data if present and younger than max_age_hours.
    Returns None on miss or expiry. data is JSON-decoded (dict/list).
    """
    try:
        c = _conn()
        row = c.execute(
            "SELECT data, fetched_at FROM api_cache WHERE source = ? AND cache_key = ?",
            (source, cache_key),
        ).fetchone()
        c.close()
        if not row:
            return None
        data_json, fetched_at = row
        t = datetime.fromisoformat(fetched_at)
        if (datetime.now() - t).total_seconds() > max_age_hours * 3600:
            return None
        return json.loads(data_json) if data_json else None
    except Exception:
        return None


def set(source: str, cache_key: str, data, fetched_at=None):
    """Store data (must be JSON-serializable) for (source, cache_key)."""
    if fetched_at is None:
        fetched_at = datetime.now().isoformat()
    try:
        data_json = json.dumps(data, default=str)
        c = _conn()
        c.execute(
            """INSERT OR REPLACE INTO api_cache (source, cache_key, data, fetched_at)
               VALUES (?, ?, ?, ?)""",
            (source, cache_key, data_json, fetched_at),
        )
        c.commit()
        c.close()
    except Exception:
        pass


def clear_older_than_days(days: int = 30):
    """Remove cache entries older than `days` to limit DB size."""
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c = _conn()
        c.execute("DELETE FROM api_cache WHERE fetched_at < ?", (cutoff,))
        c.commit()
        c.close()
    except Exception:
        pass
