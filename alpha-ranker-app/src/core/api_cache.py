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


def get_cache_path():
    """Chemin absolu du fichier SQLite du cache (pour suivre l'état)."""
    return str(DB_PATH.resolve())


def get_cached_ticker_list():
    """Liste des tickers connus (cache_key pour source=yahoo_info), triée."""
    try:
        c = _conn()
        rows = c.execute(
            "SELECT cache_key FROM api_cache WHERE source = ? ORDER BY cache_key",
            ("yahoo_info",),
        ).fetchall()
        c.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def get_tickers_by_market_cap():
    """(ticker, market_cap) triés par market_cap décroissant (pour plus grosses cap)."""
    try:
        c = _conn()
        rows = c.execute(
            "SELECT cache_key, data FROM api_cache WHERE source = ?",
            ("yahoo_info",),
        ).fetchall()
        c.close()
        out = []
        for cache_key, data_json in rows:
            try:
                d = json.loads(data_json) if data_json else {}
                mc = d.get("marketCap")
                if mc is not None and isinstance(mc, (int, float)) and mc > 0:
                    out.append((cache_key, float(mc)))
            except Exception:
                pass
        out.sort(key=lambda x: -x[1])
        return out
    except Exception:
        return []


def get_cache_status():
    """
    État du cache: nb entrées par source, années disponibles, nombre de tickers/ISIN.
    Utile pour suivre sans ouvrir la DB à la main.
    """
    out = {"path": get_cache_path(), "total": 0, "by_source": {}, "prices_yearly_years": [], "n_tickers": None}
    try:
        c = _conn()
        rows = c.execute(
            "SELECT source, cache_key, fetched_at FROM api_cache ORDER BY source, cache_key"
        ).fetchall()
        c.close()
        out["total"] = len(rows)
        for source, cache_key, fetched_at in rows:
            out["by_source"][source] = out["by_source"].get(source, 0) + 1
            if source == "prices_yearly":
                out["prices_yearly_years"].append({"year": cache_key, "fetched_at": fetched_at})
        out["prices_yearly_years"].sort(key=lambda x: x["year"])
        # Nombre de tickers: depuis universe_meta si présent, sinon nb de yahoo_info (proxy)
        meta = get("universe_meta", "count", max_age_hours=24 * 365 * 20)
        if isinstance(meta, dict) and "n_tickers" in meta:
            out["n_tickers"] = int(meta["n_tickers"])
        elif out["by_source"].get("yahoo_info"):
            out["n_tickers"] = out["by_source"]["yahoo_info"]
    except Exception:
        pass
    return out


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


def get_stored_universe_list():
    """
    Liste d'univers stockée en DB. Reste disponible même si l'API ne renvoie plus un titre (délisté, etc.).
    Retourne une liste de tickers (ou list de dict avec isin/ticker si on passe en ISIN plus tard).
    """
    try:
        raw = get("universe_list", "securities", max_age_hours=24 * 365 * 50)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and "tickers" in raw:
            return raw["tickers"]
        return []
    except Exception:
        return []


def set_stored_universe_list(tickers):
    """Enregistre la liste d'univers en DB (persistante, pas effacée par clear_older_than_days)."""
    set("universe_list", "securities", {"tickers": list(tickers), "updated": datetime.now().isoformat()})


def get_isin_map():
    """Dictionnaire ticker -> ISIN (affichage). Vide si pas encore enrichi."""
    try:
        raw = get("universe_list", "isin_map", max_age_hours=24 * 365 * 50)
        if isinstance(raw, dict):
            return raw
        return {}
    except Exception:
        return {}


def set_isin_map(ticker_to_isin):
    """Enregistre ou fusionne le mapping ticker -> ISIN (persistant)."""
    try:
        existing = get_isin_map()
        existing.update(ticker_to_isin)
        set("universe_list", "isin_map", existing)
    except Exception:
        pass


def get_display_id(ticker, isin_map=None):
    """Pour l'affichage: ISIN si connu, sinon ticker."""
    if isin_map is None:
        isin_map = get_isin_map()
    return isin_map.get(ticker) or ticker


def clear_older_than_days(days: int = 30, exclude_sources=None):
    """Remove cache entries older than `days`. Never touch exclude_sources (e.g. prices_yearly, universe_list)."""
    if exclude_sources is None:
        exclude_sources = ("prices_yearly", "universe_list")
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        c = _conn()
        placeholders = ",".join("?" * len(exclude_sources))
        c.execute(
            f"DELETE FROM api_cache WHERE fetched_at < ? AND source NOT IN ({placeholders})",
            [cutoff] + list(exclude_sources),
        )
        c.commit()
        c.close()
    except Exception:
        pass
