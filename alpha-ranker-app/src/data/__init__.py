"""
Institutional data access layer.

This package provides:
- Pluggable API clients for multiple market data providers
- ISIN-aware universe construction
- A unified data pipeline that merges prices, fundamentals, macro and sentiment

Existing higher-level code (e.g. `core.model.run_full_pipeline`) can either:
- Keep using `core.data.fetch_all_data` (backward compatible), or
- Gradually migrate to this package for more flexible multi-source ingestion.
"""

from .data_pipeline import DataPipeline
from .universe_builder import UniverseBuilder

__all__ = ["DataPipeline", "UniverseBuilder"]

