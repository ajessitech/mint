"""
Embedding encoder: Generate embeddings for sessions and signals.

Uses OpenAI's text-embedding-3-small model for vector representation
of sessions and signals for clustering and similarity search.
"""

from typing import Optional, Any
import numpy as np

from openai import AsyncOpenAI

from codex_loop.db.schema import Session, Signal


class SessionEncoder:
    """
    Encoder for generating session embeddings.
    
    Creates a summary of the session and embeds it for clustering
    similar sessions together.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the encoder.
        
        Args:
            model: OpenAI embedding model name
            client: Optional pre-configured client
        """
        self.model = model
        self.client = client or AsyncOpenAI()
    
    async def encode_session(self, session: Session) -> np.ndarray:
        """
        Generate an embedding for a session.
        
        Args:
            session: The session to embed
            
        Returns:
            Embedding vector as numpy array
        """
        summary = self._build_session_summary(session)
        
        response = await self.client.embeddings.create(
            input=summary,
            model=self.model,
        )
        
        return np.array(response.data[0].embedding)
    
    async def encode_sessions_batch(
        self,
        sessions: list[Session],
        batch_size: int = 100,
    ) -> list[np.ndarray]:
        """
        Generate embeddings for multiple sessions.
        
        Args:
            sessions: List of sessions to embed
            batch_size: Maximum batch size per API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(sessions), batch_size):
            batch = sessions[i:i + batch_size]
            summaries = [self._build_session_summary(s) for s in batch]
            
            response = await self.client.embeddings.create(
                input=summaries,
                model=self.model,
            )
            
            for data in response.data:
                embeddings.append(np.array(data.embedding))
        
        return embeddings
    
    def _build_session_summary(self, session: Session) -> str:
        """
        Build a text summary of a session for embedding.
        
        Includes key facets and characteristics.
        """
        parts = []
        
        # Basic info
        parts.append(f"Session source: {session.source}")
        if session.model:
            parts.append(f"Model: {session.model}")
        
        # Activity metrics
        parts.append(f"Turns: {session.turn_count or 0}")
        parts.append(f"Tool calls: {session.tool_call_count or 0}")
        parts.append(f"Compactions: {session.compaction_count or 0}")
        
        # Facets
        facets = session.facets or {}
        if facets.get("denial_count"):
            parts.append(f"Denials: {facets['denial_count']}")
        if facets.get("error_count"):
            parts.append(f"Errors: {facets['error_count']}")
        if facets.get("exec_count"):
            parts.append(f"Exec calls: {facets['exec_count']}")
        if facets.get("patch_count"):
            parts.append(f"Patch calls: {facets['patch_count']}")
        
        # User messages (first turn if available)
        if session.turns:
            first_turn = session.turns[0]
            if first_turn.user_message:
                # Truncate for embedding
                msg = first_turn.user_message[:500]
                parts.append(f"Initial request: {msg}")
        
        return " | ".join(parts)


class SignalEncoder:
    """
    Encoder for generating signal embeddings.
    
    Creates contextual embeddings of friction/delight signals
    for clustering similar patterns together.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the encoder.
        
        Args:
            model: OpenAI embedding model name
            client: Optional pre-configured client
        """
        self.model = model
        self.client = client or AsyncOpenAI()
    
    async def encode_signal(self, signal: Signal) -> np.ndarray:
        """
        Generate an embedding for a signal.
        
        Args:
            signal: The signal to embed
            
        Returns:
            Embedding vector as numpy array
        """
        context = self._build_signal_context(signal)
        
        response = await self.client.embeddings.create(
            input=context,
            model=self.model,
        )
        
        return np.array(response.data[0].embedding)
    
    async def encode_signals_batch(
        self,
        signals: list[Signal],
        batch_size: int = 100,
    ) -> list[np.ndarray]:
        """
        Generate embeddings for multiple signals.
        
        Args:
            signals: List of signals to embed
            batch_size: Maximum batch size per API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i + batch_size]
            contexts = [self._build_signal_context(s) for s in batch]
            
            response = await self.client.embeddings.create(
                input=contexts,
                model=self.model,
            )
            
            for data in response.data:
                embeddings.append(np.array(data.embedding))
        
        return embeddings
    
    def _build_signal_context(self, signal: Signal) -> str:
        """
        Build a text context for a signal for embedding.
        
        Includes signal type, category, severity, and evidence.
        """
        parts = []
        
        # Signal classification
        parts.append(f"Signal type: {signal.signal_type}")
        parts.append(f"Category: {signal.category}")
        parts.append(f"Severity: {signal.severity:.2f}")
        
        # Detection method
        if signal.detector:
            parts.append(f"Detected by: {signal.detector}")
        
        # Evidence details
        evidence = signal.evidence or {}
        if evidence.get("description"):
            parts.append(f"Description: {evidence['description']}")
        if evidence.get("raw"):
            raw = evidence["raw"]
            # Extract key evidence fields
            if isinstance(raw, dict):
                for key in ["message", "exit_code", "denial_count", "compaction_count"]:
                    if key in raw:
                        parts.append(f"{key}: {raw[key]}")
        
        # Session context
        if signal.session:
            parts.append(f"Session source: {signal.session.source}")
            if signal.session.model:
                parts.append(f"Model: {signal.session.model}")
        
        return " | ".join(parts)


# Alias for backward compatibility
def encode_text(text: str, client: AsyncOpenAI, model: str = "text-embedding-3-small"):
    """Encode arbitrary text to embedding vector."""
    import asyncio
    
    async def _encode():
        response = await client.embeddings.create(
            input=text,
            model=model,
        )
        return np.array(response.data[0].embedding)
    
    return asyncio.run(_encode())
