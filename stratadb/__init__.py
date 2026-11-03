"""StrataDB: an embedded, graph-native vector database."""

from .collection import Collection, CollectionConfig
from .database import Database
from .filters import FilterError, compile_filter
from .hnsw import HNSWIndex, NativeHNSWIndex, PyHNSWIndex

__version__ = "1.0.0"

__all__ = ["Collection", "CollectionConfig", "Database", "FilterError", "HNSWIndex", "NativeHNSWIndex", "PyHNSWIndex", "compile_filter"]
