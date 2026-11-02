"""StrataDB: an embedded, graph-native vector database."""

from .collection import Collection, CollectionConfig
from .database import Database
from .filters import FilterError, compile_filter
from .hnsw import HNSWIndex

__version__ = "0.5.0"

__all__ = ["Collection", "CollectionConfig", "Database", "FilterError", "HNSWIndex", "compile_filter"]
