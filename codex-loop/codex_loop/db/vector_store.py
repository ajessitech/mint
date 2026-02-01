"""
ChromaDB vector store wrapper for signal and session embeddings.

Provides collections for:
- signal_embeddings: Individual friction/delight moments
- session_embeddings: Full session summaries
- cluster_centroids: Cluster center vectors
"""

from pathlib import Path
from typing import Optional, Any
import uuid

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaVectorStore:
    """
    ChromaDB wrapper for the closed loop system.
    
    Handles embedding storage and retrieval for signals, sessions, and clusters.
    """

    SIGNAL_COLLECTION = "signal_embeddings"
    SESSION_COLLECTION = "session_embeddings"
    CLUSTER_COLLECTION = "cluster_centroids"

    def __init__(self, path: Path | str):
        """
        Initialize the vector store.
        
        Args:
            path: Directory path for persistent storage
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Initialize collections
        self._signal_collection = self.client.get_or_create_collection(
            name=self.SIGNAL_COLLECTION,
            metadata={"description": "Signal embeddings for clustering"},
        )
        self._session_collection = self.client.get_or_create_collection(
            name=self.SESSION_COLLECTION,
            metadata={"description": "Session summary embeddings"},
        )
        self._cluster_collection = self.client.get_or_create_collection(
            name=self.CLUSTER_COLLECTION,
            metadata={"description": "Cluster centroid embeddings"},
        )

    def add_signal_embedding(
        self,
        embedding: np.ndarray | list[float],
        signal_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add a signal embedding to the store.
        
        Args:
            embedding: The embedding vector
            signal_id: Unique signal identifier
            metadata: Optional metadata (signal_type, category, etc.)
            
        Returns:
            The embedding ID
        """
        embedding_id = f"sig_{signal_id}"
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        self._signal_collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        
        return embedding_id

    def add_session_embedding(
        self,
        embedding: np.ndarray | list[float],
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a session embedding to the store."""
        embedding_id = f"sess_{session_id}"
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        self._session_collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        
        return embedding_id

    def add_cluster_centroid(
        self,
        embedding: np.ndarray | list[float],
        cluster_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a cluster centroid embedding."""
        embedding_id = f"clust_{cluster_id}"
        
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        self._cluster_collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        
        return embedding_id

    def get_all_signal_embeddings(
        self,
        where: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> tuple[list[str], np.ndarray, list[dict]]:
        """
        Get all signal embeddings, optionally filtered.
        
        Args:
            where: Optional filter (e.g., {"signal_type": "friction"})
            limit: Maximum number of results
            
        Returns:
            Tuple of (ids, embeddings array, metadatas)
        """
        kwargs = {"include": ["embeddings", "metadatas"]}
        if where:
            kwargs["where"] = where
        if limit:
            kwargs["limit"] = limit
        
        results = self._signal_collection.get(**kwargs)
        
        ids = results["ids"]
        embeddings = np.array(results["embeddings"]) if results["embeddings"] else np.array([])
        metadatas = results["metadatas"] or []
        
        return ids, embeddings, metadatas

    def query_similar_signals(
        self,
        embedding: np.ndarray | list[float],
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[float], list[dict]]:
        """
        Find similar signals by embedding.
        
        Returns:
            Tuple of (ids, distances, metadatas)
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where
        
        results = self._signal_collection.query(**kwargs)
        
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        return ids, distances, metadatas

    def get_signal_count(self, where: Optional[dict[str, Any]] = None) -> int:
        """Get the number of signals in the store."""
        return self._signal_collection.count()

    def delete_signal_embedding(self, signal_id: str) -> None:
        """Delete a signal embedding."""
        embedding_id = f"sig_{signal_id}"
        self._signal_collection.delete(ids=[embedding_id])

    def clear_all(self) -> None:
        """Clear all collections (use with caution)."""
        self.client.delete_collection(self.SIGNAL_COLLECTION)
        self.client.delete_collection(self.SESSION_COLLECTION)
        self.client.delete_collection(self.CLUSTER_COLLECTION)
        
        # Recreate empty collections
        self._signal_collection = self.client.create_collection(self.SIGNAL_COLLECTION)
        self._session_collection = self.client.create_collection(self.SESSION_COLLECTION)
        self._cluster_collection = self.client.create_collection(self.CLUSTER_COLLECTION)

    # Convenience method for backward compatibility
    def add(
        self,
        embedding: np.ndarray | list[float],
        id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add an embedding (defaults to signal collection)."""
        return self.add_signal_embedding(embedding, id, metadata)
