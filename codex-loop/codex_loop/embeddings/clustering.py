"""
Signal clustering: Group similar signals using HDBSCAN.

Enables unsupervised discovery of new friction/delight patterns
per brainstorm-updated.md lines 124-133 (hybrid discovery).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import uuid

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from codex_loop.db.schema import Cluster, Signal
from codex_loop.db.vector_store import ChromaVectorStore


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    clusters: list[Cluster]
    labels: np.ndarray
    noise_count: int
    embedding_ids: list[str]


class SignalClusterer:
    """
    Cluster signals by embedding similarity using HDBSCAN.
    
    HDBSCAN is used because:
    1. It doesn't require specifying k (number of clusters)
    2. It handles noise (unclustered points) naturally
    3. It finds clusters of varying densities
    """
    
    def __init__(self, vector_store: ChromaVectorStore):
        """
        Initialize the clusterer.
        
        Args:
            vector_store: ChromaDB store containing signal embeddings
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan is required for clustering. "
                "Install with: pip install hdbscan"
            )
        
        self.vector_store = vector_store
    
    def cluster_signals(
        self,
        signal_type: str,  # "friction" or "delight"
        min_cluster_size: int = 5,
        min_samples: int = 3,
        existing_clusters: Optional[list[Cluster]] = None,
    ) -> ClusterResult:
        """
        Cluster signals of a given type.
        
        Args:
            signal_type: "friction" or "delight"
            min_cluster_size: Minimum signals to form a cluster
            min_samples: Core point neighborhood size
            existing_clusters: Optional existing clusters to update
            
        Returns:
            ClusterResult with new/updated clusters
        """
        # Get all embeddings of this signal type
        ids, embeddings, metadatas = self.vector_store.get_all_signal_embeddings(
            where={"signal_type": signal_type}
        )
        
        if len(embeddings) < min_cluster_size:
            # Not enough signals to cluster
            return ClusterResult(
                clusters=[],
                labels=np.array([]),
                noise_count=len(embeddings),
                embedding_ids=ids,
            )
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of mass
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Build cluster objects
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        clusters = []
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            cluster_ids = [ids[i] for i, m in enumerate(mask) if m]
            
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0)
            
            # Create or update cluster
            cluster_id = f"cluster_{signal_type}_{label}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            cluster = Cluster(
                id=cluster_id,
                signal_type=signal_type,
                member_count=int(mask.sum()),
                centroid_embedding_id=None,  # Will be set when stored
                created_at=datetime.utcnow(),
            )
            
            # Store centroid in vector store
            centroid_id = self.vector_store.add_cluster_centroid(
                centroid,
                cluster_id,
                metadata={"signal_type": signal_type, "member_count": int(mask.sum())},
            )
            cluster.centroid_embedding_id = centroid_id
            
            clusters.append(cluster)
        
        noise_count = int((labels == -1).sum())
        
        return ClusterResult(
            clusters=clusters,
            labels=labels,
            noise_count=noise_count,
            embedding_ids=ids,
        )
    
    def assign_to_nearest_cluster(
        self,
        embedding: np.ndarray,
        signal_type: str,
        threshold: float = 0.5,
    ) -> Optional[str]:
        """
        Assign a new signal to the nearest existing cluster.
        
        Args:
            embedding: The signal's embedding vector
            signal_type: "friction" or "delight"
            threshold: Maximum distance to assign (larger = more lenient)
            
        Returns:
            Cluster ID if assigned, None if too far from any cluster
        """
        ids, distances, metadatas = self.vector_store.query_similar_signals(
            embedding,
            n_results=1,
            where={"signal_type": signal_type},
        )
        
        if not ids or distances[0] > threshold:
            return None
        
        # Return the cluster ID from metadata
        return metadatas[0].get("cluster_id")
    
    def get_cluster_samples(
        self,
        cluster_id: str,
        n_samples: int = 10,
    ) -> list[str]:
        """
        Get sample signal IDs from a cluster.
        
        Args:
            cluster_id: The cluster to sample from
            n_samples: Number of samples to return
            
        Returns:
            List of signal IDs
        """
        # Query signals near the cluster centroid
        # First get the centroid embedding
        # This would need the centroid embedding - for now return empty
        return []


def compute_cluster_stats(
    signals: list[Signal],
    labels: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """
    Compute statistics for each cluster.
    
    Args:
        signals: List of signals that were clustered
        labels: Cluster labels from HDBSCAN
        
    Returns:
        Dict mapping cluster label to stats
    """
    stats = {}
    
    unique_labels = set(labels)
    for label in unique_labels:
        mask = labels == label
        cluster_signals = [s for s, m in zip(signals, mask) if m]
        
        if not cluster_signals:
            continue
        
        # Compute statistics
        severities = [s.severity for s in cluster_signals]
        categories = [s.category for s in cluster_signals]
        
        from collections import Counter
        category_counts = Counter(categories)
        
        stats[label] = {
            "count": len(cluster_signals),
            "avg_severity": np.mean(severities),
            "max_severity": max(severities),
            "min_severity": min(severities),
            "top_categories": category_counts.most_common(3),
            "is_noise": label == -1,
        }
    
    return stats


def merge_similar_clusters(
    clusters: list[Cluster],
    embeddings: dict[str, np.ndarray],
    threshold: float = 0.3,
) -> list[Cluster]:
    """
    Merge clusters that are very similar.
    
    Args:
        clusters: List of clusters to potentially merge
        embeddings: Dict mapping cluster_id to centroid embedding
        threshold: Maximum distance to merge
        
    Returns:
        Merged cluster list
    """
    if len(clusters) <= 1:
        return clusters
    
    # Build distance matrix
    n = len(clusters)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            ci = clusters[i].id
            cj = clusters[j].id
            if ci in embeddings and cj in embeddings:
                dist = np.linalg.norm(embeddings[ci] - embeddings[cj])
                distances[i, j] = dist
                distances[j, i] = dist
    
    # Find clusters to merge (simple greedy approach)
    merged = []
    used = set()
    
    for i in range(n):
        if i in used:
            continue
        
        cluster = clusters[i]
        
        # Find similar clusters
        for j in range(i + 1, n):
            if j in used:
                continue
            if distances[i, j] < threshold:
                # Merge j into i
                cluster.member_count += clusters[j].member_count
                used.add(j)
        
        merged.append(cluster)
        used.add(i)
    
    return merged
