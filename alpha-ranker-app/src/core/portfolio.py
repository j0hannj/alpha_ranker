"""Portfolio manager with SQLite persistence.

Tracks positions with strategy type (LONG_TERM / MEDIUM_TERM / SHORT_TERM),
entry/current prices, model metrics (confidence, alpha_score, expected_return),
target_price, stop_loss, holding_horizon_days, transaction_cost, and status (OPEN/SOLD).
"""
import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent.parent.parent / "db" / "portfolio.db"


def get_ranking_db_path():
    """Return absolute path to the portfolio DB (for ranking_history). Use this when passing db_path to add_ranking_insights."""
    return str(DB_PATH.resolve())


# New columns for decision-engine tracking (added via migration)
_HOLDINGS_EXTRA_COLUMNS = [
    ("strategy_type", "TEXT DEFAULT 'LONG_TERM'"),
    ("asset_type", "TEXT"),
    ("entry_price", "REAL"),
    ("entry_date", "TEXT"),
    ("confidence", "REAL"),
    ("alpha_score", "REAL"),
    ("expected_return", "REAL"),
    ("target_price", "REAL"),
    ("stop_loss", "REAL"),
    ("holding_horizon_days", "INTEGER"),
    ("transaction_cost", "REAL"),
    ("status", "TEXT DEFAULT 'OPEN'"),
    ("review_date", "TEXT"),
    ("price_timestamp", "TEXT"),
]

def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS holdings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        isin TEXT,
        name TEXT,
        type TEXT DEFAULT 'stock',
        units REAL NOT NULL,
        avg_price REAL NOT NULL,
        current_price REAL,
        currency TEXT DEFAULT 'USD',
        sector TEXT,
        sectors_json TEXT,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    # Migration: add decision-engine columns if missing
    info = {row[1] for row in c.execute("PRAGMA table_info(holdings)").fetchall()}
    for col, spec in _HOLDINGS_EXTRA_COLUMNS:
        if col not in info:
            c.execute(f"ALTER TABLE holdings ADD COLUMN {col} {spec}")
    c.execute("""CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, price REAL, date TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS system_config (
        domain TEXT PRIMARY KEY,
        value TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS ranking_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        alpha_score REAL,
        rank_position INTEGER NOT NULL,
        confidence REAL,
        timestamp TEXT NOT NULL,
        run_id TEXT
    )""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_ranking_history_ticker_ts ON ranking_history(ticker, timestamp)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_ranking_history_run ON ranking_history(run_id)""")
    # Migration: add columns for stability and model metrics
    rh_info = {row[1] for row in c.execute("PRAGMA table_info(ranking_history)").fetchall()}
    for col, spec in [("run_date", "TEXT"), ("alpha_score_raw", "REAL"), ("predicted_return_pct", "REAL"), ("model_agreement", "REAL")]:
        if col not in rh_info:
            c.execute(f"ALTER TABLE ranking_history ADD COLUMN {col} {spec}")
    c.execute("""CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        holding_id INTEGER,
        ticker TEXT NOT NULL,
        action TEXT NOT NULL,
        units REAL NOT NULL,
        price REAL NOT NULL,
        total_amount REAL,
        pnl_realized REAL,
        pnl_pct REAL,
        currency TEXT DEFAULT 'EUR',
        reason TEXT,
        signal_data TEXT,
        executed_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS universe (
        ticker TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT,
        market_cap REAL,
        country TEXT,
        exchange TEXT,
        discovered_at TEXT,
        last_seen_at TEXT,
        is_active INTEGER DEFAULT 1
    )""")
    c.commit()
    return c


def get_universe():
    """Load known universe from DB. Returns dict ticker -> {shortName, sector, industry, marketCap, country, exchange, discovered_at, last_seen_at}."""
    c = _conn()
    try:
        rows = c.execute("SELECT ticker, name, sector, industry, market_cap, country, exchange, discovered_at, last_seen_at FROM universe WHERE is_active = 1").fetchall()
        c.close()
        return {
            r["ticker"]: {
                "shortName": r["name"] or r["ticker"],
                "sector": r["sector"],
                "industry": r["industry"],
                "marketCap": r["market_cap"],
                "country": r["country"],
                "exchange": r["exchange"],
                "discovered_at": r["discovered_at"],
                "last_seen_at": r["last_seen_at"],
            }
            for r in rows
        }
    except Exception as e:
        logger.warning("get_universe failed: %s", e)
        try:
            c.close()
        except Exception:
            pass
        return {}


def save_universe(stocks_dict):
    """Save/update universe in DB. New tickers get discovered_at=now; existing get last_seen_at=now."""
    if not stocks_dict:
        return
    c = _conn()
    now = datetime.now().isoformat()
    try:
        for ticker, info in stocks_dict.items():
            name = info.get("name") or info.get("shortName") or info.get("companyName") or ticker
            sector = info.get("sector")
            industry = info.get("industry")
            market_cap = info.get("marketCap") or info.get("market_cap")
            country = info.get("country")
            exchange = info.get("exchange") or info.get("exchangeShortName")
            existing = c.execute("SELECT 1 FROM universe WHERE ticker = ?", (ticker,)).fetchone()
            if existing:
                c.execute("""UPDATE universe SET name=?, sector=?, industry=?, market_cap=?,
                            country=?, exchange=?, last_seen_at=?, is_active=1 WHERE ticker=?""",
                         (name, sector, industry, market_cap, country, exchange, now, ticker))
            else:
                c.execute("""INSERT INTO universe (ticker, name, sector, industry, market_cap,
                            country, exchange, discovered_at, last_seen_at, is_active)
                            VALUES (?,?,?,?,?,?,?,?,?,1)""",
                         (ticker, name, sector, industry, market_cap, country, exchange, now, now))
        c.commit()
    except Exception as e:
        logger.warning("save_universe failed: %s", e)
    finally:
        c.close()

def get_all(include_sold=False):
    """Return holdings. By default only OPEN (exclude SOLD). Use include_sold=True for history."""
    c = _conn()
    if include_sold:
        rows = c.execute("SELECT * FROM holdings ORDER BY type, ticker").fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM holdings WHERE COALESCE(status,'OPEN') = 'OPEN' ORDER BY type, ticker"
        ).fetchall()
    c.close()
    return [dict(r) for r in rows]

def add(ticker, name, typ, units, avg_price, currency, sector=None, sectors_json=None, isin=None,
        strategy_type="LONG_TERM", entry_date=None, confidence=None, alpha_score=None, expected_return=None,
        target_price=None, stop_loss=None, holding_horizon_days=None, transaction_cost=None, review_date=None):
    c = _conn()
    entry_date = entry_date or datetime.now().strftime("%Y-%m-%d")
    c.execute("""INSERT INTO holdings (ticker,isin,name,type,units,avg_price,current_price,currency,sector,sectors_json,
        strategy_type,asset_type,entry_price,entry_date,confidence,alpha_score,expected_return,target_price,stop_loss,
        holding_horizon_days,transaction_cost,status,review_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker, isin, name, typ, units, avg_price, avg_price, currency, sector, sectors_json,
         strategy_type, typ, avg_price, entry_date, confidence, alpha_score, expected_return, target_price, stop_loss,
         holding_horizon_days, transaction_cost, "OPEN", review_date))
    c.commit(); c.close()

def update(hid, **kwargs):
    c = _conn()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [hid]
    c.execute(f"UPDATE holdings SET {sets}, updated_at=CURRENT_TIMESTAMP WHERE id=?", vals)
    c.commit(); c.close()

def delete(hid):
    c = _conn()
    c.execute("DELETE FROM holdings WHERE id=?", (hid,))
    c.commit(); c.close()

def update_price(ticker, price, price_timestamp=None):
    """Update current price for a ticker. price_timestamp: when the price was retrieved (e.g. YYYY-MM-DD HH:MM)."""
    ts = price_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    c = _conn()
    c.execute("UPDATE holdings SET current_price=?, updated_at=CURRENT_TIMESTAMP, price_timestamp=? WHERE ticker=?", (price, ts, ticker))
    c.execute("INSERT INTO price_history (ticker, price, date) VALUES (?, ?, ?)", (ticker, price, ts))
    c.commit(); c.close()

def price_with_timestamp(ticker, price, price_timestamp=None):
    """Combined display: ticker | price | updated timestamp. E.g. AAPL | 185.21 | updated 2026-03-09 18:45"""
    ts = price_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    p = f"{price:,.2f}" if price is not None and price == price else "—"
    return f"{ticker} | {p} | updated {ts}"

def get_setting(key, default=None):
    c = _conn()
    r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    c.close()
    return r["value"] if r else default

def set_setting(key, value):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    c.commit(); c.close()

def get_system_config(domain):
    """Load JSON config for a domain (model_settings, portfolio_settings, feature_settings, etc.)."""
    c = _conn()
    r = c.execute("SELECT value FROM system_config WHERE domain = ?", (domain,)).fetchone()
    c.close()
    if not r or not r[0]:
        return None
    try:
        return json.loads(r[0])
    except Exception:
        return None

def set_system_config(domain, value):
    """Store JSON config for a domain. value: dict (will be JSON-serialized)."""
    c = _conn()
    c.execute("INSERT OR REPLACE INTO system_config (domain, value) VALUES (?, ?)",
             (domain, json.dumps(value, default=str)))
    c.commit(); c.close()

def compute_pnl(holdings_list, fx_rate=1.08, gbp_rate=1.16, base="EUR"):
    total_val = 0; total_cost = 0; sectors = {}
    results = []
    for h in holdings_list:
        cp = h.get("current_price") or h["avg_price"]
        val = h["units"] * cp
        cost = h["units"] * h["avg_price"]
        if h["currency"] == "USD" and base == "EUR":
            val /= fx_rate; cost /= fx_rate
        elif h["currency"] == "GBP" and base == "EUR":
            val *= gbp_rate; cost *= gbp_rate
        elif h["currency"] == "EUR" and base == "USD":
            val *= fx_rate; cost *= fx_rate
        total_val += val; total_cost += cost
        pnl = val - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        if h.get("sectors_json"):
            try:
                secs = json.loads(h["sectors_json"])
                for s, w in secs.items():
                    sectors[s] = sectors.get(s, 0) + val * w
            except Exception as e:
                logger.warning("portfolio_with_pnl: sectors_json parse for holding %s failed: %s", h.get("ticker"), e)
        elif h.get("sector"):
            sectors[h["sector"]] = sectors.get(h["sector"], 0) + val
        results.append({**h, "value": round(val, 2), "cost": round(cost, 2),
                       "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2)})
    sec_data = [{"sector": s, "value": round(v, 2), "pct": round(v / total_val * 100, 1) if total_val > 0 else 0}
                for s, v in sorted(sectors.items(), key=lambda x: -x[1])]
    return {"holdings": results, "total_value": round(total_val, 2), "total_cost": round(total_cost, 2),
            "total_pnl": round(total_val - total_cost, 2),
            "total_pnl_pct": round((total_val - total_cost) / total_cost * 100, 2) if total_cost > 0 else 0,
            "sectors": sec_data}

def init_default_portfolio():
    """Insert real portfolio using ISINs as primary identifiers.
    Tickers are resolved automatically via the ISIN resolver."""
    if get_all():
        return

    # Format: (isin, name, type, units, avg_price, currency, sector, sectors_json)
    defaults = [
        # ETFs
        ("LU1681038243", "Amundi Nasdaq-100 Swap ETF", "etf", 31, 503.53, "EUR", None,
         json.dumps({"Technology":0.52,"Communication Services":0.17,"Consumer Discretionary":0.14,"Health Care":0.07,"Industrials":0.03})),
        ("IE00B4L5Y983", "iShares Core MSCI World UCITS", "etf", 173, 108.33, "EUR", None,
         json.dumps({"Technology":0.24,"Financials":0.16,"Health Care":0.12,"Industrials":0.10,"Energy":0.05,"Communication Services":0.08,"Consumer Staples":0.07})),
        ("IE00BK5BQT80", "Vanguard FTSE All-World USD", "etf", 24, 127.98, "EUR", None,
         json.dumps({"Technology":0.23,"Financials":0.17,"Health Care":0.11,"Industrials":0.11,"Energy":0.05,"Communication Services":0.07,"Consumer Staples":0.07})),
        # Stocks in EUR
        ("US02079K1079", "Alphabet Inc -C-", "stock", 1, 264.33, "EUR", "Communication Services", None),
        ("US0231351067", "Amazon.com", "stock", 1, 194.07, "EUR", "Consumer Discretionary", None),
        ("US0378331005", "Apple Inc", "stock", 1, 229.80, "EUR", "Technology", None),
        ("US5949181045", "Microsoft Corp", "stock", 1, 411.88, "EUR", "Technology", None),
        ("US67066G1040", "NVIDIA Corp", "stock", 1, 155.59, "EUR", "Technology", None),
        # Stocks in USD
        ("US0846707026", "Berkshire Hathaway -B-", "stock", 2, 493.12, "USD", "Financials", None),
        ("US11135F1012", "Broadcom Inc", "stock", 1, 343.47, "USD", "Technology", None),
        ("US46266C1053", "IonQ Inc", "stock", 3, 44.47, "USD", "Technology", None),
        ("US69608A1088", "Palantir Technologies -A-", "stock", 1, 182.51, "USD", "Technology", None),
    ]

    # Resolve ISINs to tickers
    try:
        from .isin import resolve_isin
        for d in defaults:
            isin = d[0]
            result = resolve_isin(isin)
            ticker = result["ticker"] if result else isin  # Fallback to ISIN if unresolved
            add(ticker, d[1], d[2], d[3], d[4], d[5], d[6], d[7])
            # Store ISIN in the DB
            holdings = get_all()
            for h in holdings:
                if h["ticker"] == ticker and not h.get("isin"):
                    update(h["id"], isin=isin)
                    break
    except Exception as e:
        # Fallback: use hardcoded tickers if resolver fails
        fallback_tickers = {
            "LU1681038243":"6AQQ.DE", "IE00B4L5Y983":"IWDA.AS", "IE00BK5BQT80":"VWCE.DE",
            "US02079K1079":"GOOG", "US0231351067":"AMZN", "US0378331005":"AAPL",
            "US5949181045":"MSFT", "US67066G1040":"NVDA", "US0846707026":"BRK-B",
            "US11135F1012":"AVGO", "US46266C1053":"IONQ", "US69608A1088":"PLTR",
        }
        for d in defaults:
            ticker = fallback_tickers.get(d[0], d[0])
            add(ticker, d[1], d[2], d[3], d[4], d[5], d[6], d[7])


# ══════════════════════════════════════════════════════════════
# RANKING HISTORY (for delta analysis and stability)
# ══════════════════════════════════════════════════════════════
def save_ranking_snapshot(results_df, run_id=None):
    """Store current ranking for each asset. Persists run for stability comparison on next run."""
    if results_df is None or results_df.empty:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_id = run_id or datetime.now().isoformat()
    c = None
    try:
        c = _conn()
        rank_col = "rank" if "rank" in results_df.columns else "alpha_rank"
        for _, row in results_df.iterrows():
            ticker = row.get("ticker")
            if not ticker:
                continue
            rp = row.get(rank_col) if rank_col in results_df.columns else None
            try:
                rank_pos = int(rp) if rp is not None and (rp == rp if isinstance(rp, float) else True) else 0
            except (TypeError, ValueError):
                rank_pos = 0
            alpha = row.get("alpha_score")
            alpha = float(alpha) if alpha is not None and alpha == alpha else None
            conf = row.get("confidence")
            conf = float(conf) if conf is not None and conf == conf else None
            raw = row.get("alpha_score_raw")
            raw = float(raw) if raw is not None and raw == raw else None
            pred = row.get("predicted_return_pct")
            pred = float(pred) if pred is not None and pred == pred else None
            agr = row.get("model_agreement_score")
            agr = float(agr) if agr is not None and agr == agr else None
            try:
                c.execute(
                    """INSERT INTO ranking_history (ticker, alpha_score, rank_position, confidence, timestamp, run_id, run_date, alpha_score_raw, predicted_return_pct, model_agreement)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (ticker, alpha, rank_pos, conf, ts, run_id, ts, raw, pred, agr),
                )
            except sqlite3.OperationalError:
                c.execute(
                    """INSERT INTO ranking_history (ticker, alpha_score, rank_position, confidence, timestamp, run_id)
                       VALUES (?,?,?,?,?,?)""",
                    (ticker, alpha, rank_pos, conf, ts, run_id),
                )
        c.commit()
        cleanup_old_history(keep_runs=20)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("save_ranking_snapshot failed: %s", e)
    finally:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


def cleanup_old_history(keep_runs=20):
    """Keep only the last keep_runs runs (by distinct timestamp)."""
    c = _conn()
    runs = c.execute(
        "SELECT DISTINCT timestamp FROM ranking_history ORDER BY timestamp DESC"
    ).fetchall()
    if len(runs) > keep_runs:
        old_ts = [r[0] for r in runs[keep_runs:]]
        placeholders = ",".join("?" * len(old_ts))
        c.execute(f"DELETE FROM ranking_history WHERE timestamp IN ({placeholders})", old_ts)
        c.commit()
    c.close()


def get_ranking_history(ticker, last_n=20):
    """Return list of dicts with alpha_score, rank_position, confidence, timestamp for the ticker (most recent first)."""
    c = _conn()
    rows = c.execute(
        "SELECT alpha_score, rank_position, confidence, timestamp FROM ranking_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
        (ticker, last_n),
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_latest_snapshot_before(timestamp=None):
    """Return the most recent full snapshot (all tickers) before given timestamp, as dict ticker -> {rank_position, alpha_score, confidence, timestamp}."""
    c = _conn()
    if timestamp:
        run_ts = c.execute(
            "SELECT DISTINCT timestamp FROM ranking_history WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1",
            (timestamp,),
        ).fetchone()
        run_ts = run_ts[0] if run_ts else None
    else:
        run_ts = c.execute("SELECT DISTINCT timestamp FROM ranking_history ORDER BY timestamp DESC LIMIT 1 OFFSET 1").fetchone()
        run_ts = run_ts[0] if run_ts else None
    if not run_ts:
        c.close()
        return None
    rows = c.execute(
        "SELECT ticker, alpha_score, rank_position, confidence, timestamp FROM ranking_history WHERE timestamp = ?",
        (run_ts,),
    ).fetchall()
    c.close()
    return {r["ticker"]: dict(r) for r in rows}


def get_current_run_timestamp():
    """Return the timestamp of the most recent ranking run (current snapshot)."""
    c = _conn()
    row = c.execute("SELECT timestamp FROM ranking_history ORDER BY timestamp DESC LIMIT 1").fetchone()
    c.close()
    return row[0] if row else None


# ══════════════════════════════════════════════════════════════
# TRADE HISTORY (sell execution audit)
# ══════════════════════════════════════════════════════════════
def log_sell_transaction(holding_id, ticker, units, price, avg_price, pnl_realized, pnl_pct, currency,
                         reason=None, signal_data=None):
    """Record a SELL in trade_history. Called after _process_sell."""
    c = _conn()
    total = round(units * price, 2)
    c.execute(
        """INSERT INTO trade_history
           (holding_id, ticker, action, units, price, total_amount, pnl_realized, pnl_pct, currency, reason, signal_data)
           VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?)""",
        (holding_id, ticker, units, price, total, round(pnl_realized, 2), round(pnl_pct, 2), currency,
         reason, json.dumps(signal_data, default=str) if signal_data else None),
    )
    c.commit()
    c.close()


def get_trade_history(limit=100):
    """Return recent trades (SELLs) for the History UI."""
    c = _conn()
    rows = c.execute(
        "SELECT * FROM trade_history ORDER BY executed_at DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]
