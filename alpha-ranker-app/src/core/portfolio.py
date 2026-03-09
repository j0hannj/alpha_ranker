"""Portfolio manager with SQLite persistence."""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent.parent / "db" / "portfolio.db"

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
    c.execute("""CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, price REAL, date TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    c.commit()
    return c

def get_all():
    c = _conn()
    rows = c.execute("SELECT * FROM holdings ORDER BY type, ticker").fetchall()
    c.close()
    return [dict(r) for r in rows]

def add(ticker, name, typ, units, avg_price, currency, sector=None, sectors_json=None, isin=None):
    c = _conn()
    c.execute("INSERT INTO holdings (ticker,isin,name,type,units,avg_price,currency,sector,sectors_json) VALUES (?,?,?,?,?,?,?,?,?)",
              (ticker, isin, name, typ, units, avg_price, currency, sector, sectors_json))
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

def update_price(ticker, price):
    c = _conn()
    c.execute("UPDATE holdings SET current_price=?, updated_at=CURRENT_TIMESTAMP WHERE ticker=?", (price, ticker))
    c.execute("INSERT INTO price_history (ticker, price) VALUES (?, ?)", (ticker, price))
    c.commit(); c.close()

def get_setting(key, default=None):
    c = _conn()
    r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    c.close()
    return r["value"] if r else default

def set_setting(key, value):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
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
            except: pass
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
