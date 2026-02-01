"""
Hybrid discovery loop: Combine embeddings and coding for pattern discovery.

This module orchestrates the discovery process:
1. Cluster signals by embedding similarity
2. Identify unlabeled clusters
3. Use LLM to propose new schema entries
4. Validate and integrate new categories
"""

from typing import Optional, Any
import asyncio

from openai import AsyncOpenAI
from sqlalchemy.orm import Session as DBSession

from codex_loop.db.schema import Cluster, Signal, SchemaProposal
from codex_loop.db.vector_store import ChromaVectorStore
from codex_loop.embeddings.clustering import SignalClusterer, ClusterResult
from codex_loop.discovery.schema_evolution import (
    propose_new_schema,
    evaluate_proposal_quality,
    schema_proposal_to_db,
)
from codex_loop.signals.types import SignalCategory


async def run_discovery_loop(
    db: DBSession,
    vector_store: ChromaVectorStore,
    client: Optional[AsyncOpenAI] = None,
    min_cluster_size: int = 5,
    proposal_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Run the full discovery loop for both friction and delight signals.
    
    Args:
        db: Database session
        vector_store: Vector store with embeddings
        client: Optional OpenAI client
        min_cluster_size: Minimum signals to form a cluster
        proposal_threshold: Minimum confidence to accept a proposal
        
    Returns:
        Summary of discovery results
    """
    if client is None:
        client = AsyncOpenAI()
    
    clusterer = SignalClusterer(vector_store)
    
    results = {
        "friction_clusters": 0,
        "delight_clusters": 0,
        "new_proposals": 0,
        "accepted_proposals": [],
        "clusters": [],
    }
    
    # Process friction signals
    friction_result = await _process_signal_type(
        "friction",
        db,
        clusterer,
        client,
        min_cluster_size,
        proposal_threshold,
    )
    results["friction_clusters"] = len(friction_result["clusters"])
    results["new_proposals"] += friction_result["new_proposals"]
    results["accepted_proposals"].extend(friction_result["accepted_proposals"])
    results["clusters"].extend(friction_result["clusters"])
    
    # Process delight signals
    delight_result = await _process_signal_type(
        "delight",
        db,
        clusterer,
        client,
        min_cluster_size,
        proposal_threshold,
    )
    results["delight_clusters"] = len(delight_result["clusters"])
    results["new_proposals"] += delight_result["new_proposals"]
    results["accepted_proposals"].extend(delight_result["accepted_proposals"])
    results["clusters"].extend(delight_result["clusters"])
    
    return results


async def _process_signal_type(
    signal_type: str,
    db: DBSession,
    clusterer: SignalClusterer,
    client: AsyncOpenAI,
    min_cluster_size: int,
    proposal_threshold: float,
) -> dict[str, Any]:
    """Process a single signal type (friction or delight)."""
    
    # Run clustering
    cluster_result = clusterer.cluster_signals(
        signal_type=signal_type,
        min_cluster_size=min_cluster_size,
    )
    
    # Get existing categories
    existing_categories = [c.value for c in SignalCategory]
    
    result = {
        "clusters": [],
        "new_proposals": 0,
        "accepted_proposals": [],
    }
    
    for cluster in cluster_result.clusters:
        # Save cluster to database
        db.add(cluster)
        result["clusters"].append(cluster)
        
        # Check if this cluster represents a known category
        # Get sample signals for this cluster
        sample_signals = db.query(Signal).filter(
            Signal.embedding_id.in_([
                eid for i, eid in enumerate(cluster_result.embedding_ids)
                if cluster_result.labels[i] == cluster_result.labels[
                    cluster_result.embedding_ids.index(cluster.centroid_embedding_id.replace("clust_", "sig_"))
                ] if cluster.centroid_embedding_id else False
            ])
        ).limit(10).all()
        
        # If we can't get samples through embedding matching, just get random signals
        if not sample_signals:
            sample_signals = db.query(Signal).filter(
                Signal.signal_type == signal_type,
                Signal.cluster_id == None,
            ).limit(10).all()
        
        if not sample_signals:
            continue
        
        # Check if samples mostly belong to a known category
        categories = [s.category for s in sample_signals]
        most_common = max(set(categories), key=categories.count) if categories else None
        
        # If > 80% are the same known category, assign the cluster
        if most_common and categories.count(most_common) / len(categories) > 0.8:
            cluster.name = most_common
            continue
        
        # Otherwise, propose a new schema
        proposal = await propose_new_schema(
            cluster,
            sample_signals,
            client,
        )
        result["new_proposals"] += 1
        
        # Evaluate proposal quality
        evaluation = await evaluate_proposal_quality(
            proposal,
            existing_categories,
            client,
        )
        
        if evaluation.get("should_add") and proposal.confidence >= proposal_threshold:
            # Save proposal
            db_proposal = schema_proposal_to_db(proposal, cluster.id)
            db.add(db_proposal)
            cluster.schema_proposal_id = db_proposal.id
            cluster.name = proposal.name
            cluster.description = proposal.description
            result["accepted_proposals"].append(proposal.name)
    
    db.commit()
    return result


def get_unlabeled_clusters(db: DBSession) -> list[Cluster]:
    """Get clusters that don't have a name/schema assigned."""
    return db.query(Cluster).filter(
        Cluster.name == None,
        Cluster.promoted_to_schema == False,
    ).all()


def promote_cluster_to_schema(
    cluster: Cluster,
    db: DBSession,
) -> bool:
    """
    Promote a cluster to an official schema category.
    
    This marks the cluster as promoted and could trigger
    adding the category to the official signal types.
    """
    if not cluster.name:
        return False
    
    cluster.promoted_to_schema = True
    db.commit()
    return True
