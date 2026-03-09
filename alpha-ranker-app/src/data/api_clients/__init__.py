"""
API client registry for external data vendors.

Each client implements a lightweight, stateless interface:
- fetch_prices(tickers, start, end) -> price DataFrame or dict
- fetch_fundamentals(tickers) -> dict keyed by identifier
- fetch_macro() -> dict of macro features
- fetch_sentiment(tickers) -> dict of sentiment scores (optional)

The `DataPipeline` composes multiple clients and tries them in sequence
when data is missing or incomplete.
"""

from .fmp_client import FMPClient
from .yahoo_client import YahooClient
from .alpha_vantage_client import AlphaVantageClient
from .polygon_client import PolygonClient
from .tiingo_client import TiingoClient

__all__ = [
    "FMPClient",
    "YahooClient",
    "AlphaVantageClient",
    "PolygonClient",
    "TiingoClient",
]

