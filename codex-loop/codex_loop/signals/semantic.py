"""
Semantic signal detection: LLM-based friction and delight detection.

Implements patterns from brainstorm-updated.md lines 111-113 that require
natural language understanding:
- Repeated rephrasing (line 111)
- Escalation tone (line 112)
- Platform confusion (line 113)
"""

from typing import Optional, Any
import json

from openai import AsyncOpenAI

from codex_loop.db.schema import Session
from codex_loop.ingest.spine_reader import (
    TraceRecord,
    parse_spine,
    get_submissions_by_op,
)
from codex_loop.signals.types import (
    SignalType,
    SignalCategory,
    SignalResult,
    get_default_severity,
)
from pathlib import Path


# Prompts for semantic signal detection
SEMANTIC_PROMPTS = {
    "repeated_rephrasing": """Analyze these consecutive user messages from a coding assistant session.
Determine if the user is rephrasing the same request 3 or more times, indicating frustration or confusion.

User messages:
{messages}

Return JSON:
{{
    "detected": true/false,
    "count": <number of rephrasing attempts>,
    "severity": <0.0-1.0, higher if more rephrasing>,
    "description": "<brief explanation>"
}}""",

    "escalation_tone": """Analyze these user messages for escalation in tone.
Look for signs of frustration like: "broken", "why isn't", "frustrating", "doesn't work", 
"still not working", "I already told you", etc.

User messages:
{messages}

Return JSON:
{{
    "detected": true/false,
    "severity": <0.0-1.0, based on frustration level>,
    "keywords_found": [<list of frustration indicators found>],
    "description": "<brief explanation>"
}}""",

    "platform_confusion": """Analyze these user messages for confusion about the coding assistant's capabilities.
Look for questions like: "can you do X?", "why can't you...", "is it possible to...", 
"I thought you could...", or misunderstandings about what the assistant can access or do.

User messages:
{messages}

Return JSON:
{{
    "detected": true/false,
    "severity": <0.0-1.0, based on confusion level>,
    "confusion_type": "<capability/access/feature/other>",
    "description": "<brief explanation>"
}}""",

    "abandoned_flow": """Analyze this coding session to determine if a multi-step workflow was started but never completed.
Look for patterns like: starting to implement something, then switching topics, 
or explicit abandonment ("nevermind", "forget it", "let's do something else").

User messages:
{messages}

Session context: {context}

Return JSON:
{{
    "detected": true/false,
    "severity": <0.0-1.0>,
    "abandoned_task": "<description of what was abandoned>",
    "description": "<brief explanation>"
}}""",

    "positive_feedback": """Analyze these user messages for positive feedback and delight indicators.
Look for: "thanks", "great", "perfect", "exactly what I needed", "this is amazing", etc.

User messages:
{messages}

Return JSON:
{{
    "detected": true/false,
    "severity": <0.0-1.0, higher for stronger positive feedback>,
    "keywords_found": [<list of positive indicators found>],
    "description": "<brief explanation>"
}}""",
}


async def detect_semantic_signals(
    session: Session,
    records: Optional[list[TraceRecord]] = None,
    client: Optional[AsyncOpenAI] = None,
    model: str = "gpt-5.2",
) -> list[SignalResult]:
    """
    Detect semantic friction and delight signals using LLM analysis.
    
    Args:
        session: The canonicalized session
        records: Optional pre-parsed records
        client: Optional OpenAI client (creates one if not provided)
        model: Model to use for analysis
        
    Returns:
        List of detected signals
    """
    # Load records if not provided
    if records is None:
        if not session.spine_path:
            return []
        spine_path = Path(session.spine_path)
        if not spine_path.exists():
            return []
        records = parse_spine(spine_path)
    
    if not records:
        return []
    
    # Extract user messages
    user_messages = _extract_user_messages(records)
    if not user_messages:
        return []
    
    # Create client if not provided
    if client is None:
        try:
            client = AsyncOpenAI()
        except Exception:
            # No API key available
            return []
    
    signals = []
    
    # Run detection for each semantic pattern
    messages_text = "\n".join([f"- {msg}" for msg in user_messages])
    
    # Repeated rephrasing
    result = await _run_detection(
        client, model,
        SEMANTIC_PROMPTS["repeated_rephrasing"].format(messages=messages_text),
    )
    if result and result.get("detected"):
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.REPEATED_REPHRASING,
            severity=result.get("severity", 0.7),
            session_id=session.id,
            detector="semantic",
            description=result.get("description", "User rephrased request multiple times"),
            raw_evidence=result,
        ))
    
    # Escalation tone
    result = await _run_detection(
        client, model,
        SEMANTIC_PROMPTS["escalation_tone"].format(messages=messages_text),
    )
    if result and result.get("detected"):
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.ESCALATION_TONE,
            severity=result.get("severity", 0.8),
            session_id=session.id,
            detector="semantic",
            description=result.get("description", "User showed signs of frustration"),
            raw_evidence=result,
        ))
    
    # Platform confusion
    result = await _run_detection(
        client, model,
        SEMANTIC_PROMPTS["platform_confusion"].format(messages=messages_text),
    )
    if result and result.get("detected"):
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.PLATFORM_CONFUSION,
            severity=result.get("severity", 0.6),
            session_id=session.id,
            detector="semantic",
            description=result.get("description", "User confused about capabilities"),
            raw_evidence=result,
        ))
    
    # Abandoned flow
    context = f"Turn count: {session.turn_count}, Tool calls: {session.tool_call_count}"
    result = await _run_detection(
        client, model,
        SEMANTIC_PROMPTS["abandoned_flow"].format(messages=messages_text, context=context),
    )
    if result and result.get("detected"):
        signals.append(SignalResult(
            signal_type=SignalType.FRICTION,
            category=SignalCategory.ABANDONED_TOOL_FLOW,
            severity=result.get("severity", 0.7),
            session_id=session.id,
            detector="semantic",
            description=result.get("description", "User abandoned a workflow"),
            raw_evidence=result,
        ))
    
    # Positive feedback (delight)
    result = await _run_detection(
        client, model,
        SEMANTIC_PROMPTS["positive_feedback"].format(messages=messages_text),
    )
    if result and result.get("detected"):
        signals.append(SignalResult(
            signal_type=SignalType.DELIGHT,
            category=SignalCategory.POSITIVE_FEEDBACK,
            severity=result.get("severity", 0.8),
            session_id=session.id,
            detector="semantic",
            description=result.get("description", "User expressed positive feedback"),
            raw_evidence=result,
        ))
    
    return signals


async def _run_detection(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
) -> Optional[dict[str, Any]]:
    """Run a single detection prompt and parse the result using Responses API."""
    try:
        system_prompt = ("You are analyzing user behavior in a coding assistant session. "
                        "Return only valid JSON with no markdown formatting.")
        
        response = await client.responses.create(
            model=model,
            instructions=system_prompt,
            input=prompt,
        )
        
        content = response.output_text
        if not content:
            return None
        
        # Parse JSON response
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        return json.loads(content)
    
    except Exception as e:
        # Log but don't fail
        print(f"Warning: Semantic detection failed: {e}")
        return None


def _extract_user_messages(records: list[TraceRecord]) -> list[str]:
    """Extract user messages from trace records."""
    messages = []
    
    user_turns = get_submissions_by_op(records, "user_turn")
    for r in user_turns:
        op = r.payload.get("op", {})
        items = op.get("items", [])
        
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    messages.append(text)
    
    # Also check for legacy user_input
    user_inputs = get_submissions_by_op(records, "user_input")
    for r in user_inputs:
        op = r.payload.get("op", {})
        items = op.get("items", [])
        
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    messages.append(text)
    
    return messages


# Export for reference
SEMANTIC_FRICTION_SIGNALS = [
    "repeated_rephrasing",
    "escalation_tone", 
    "platform_confusion",
    "abandoned_flow",
]

SEMANTIC_DELIGHT_SIGNALS = [
    "positive_feedback",
]
