#!/usr/bin/env python3
"""
Generate synthetic trace spine data for demo purposes.

Creates 500 realistic Codex sessions with various friction and delight patterns:
- Success sessions (fast completion, zero denial)
- Failure sessions (exec failures, patch failures)
- Denial sessions (repeated denials, abandoned flows)
- Compaction sessions (long sessions with context churn)
- Mixed sessions (combination of patterns)

Usage:
    python scripts/generate_demo_data.py --output ~/.codex/trace_spine --count 500
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


# Session source types (weighted)
SOURCES = ["CLI", "CLI", "CLI", "VSCode", "VSCode", "exec", "SubAgent"]

# Models (weighted)
MODELS = ["gpt-4", "gpt-4", "gpt-4o", "gpt-4o", "gpt-4o-mini", "o1", "o3"]

# Programming tasks (for realistic user messages)
USER_REQUESTS = [
    "Fix the bug in the authentication module",
    "Add unit tests for the API endpoints",
    "Refactor the database layer to use async",
    "Implement user registration with email verification",
    "Create a REST API for the product catalog",
    "Add pagination to the search results",
    "Fix the memory leak in the websocket handler",
    "Implement rate limiting for the API",
    "Add logging throughout the application",
    "Create a CLI tool for database migrations",
    "Fix the failing CI tests",
    "Add Docker support to the project",
    "Implement caching for expensive queries",
    "Create a health check endpoint",
    "Fix the race condition in the worker queue",
    "Add input validation to all forms",
    "Implement file upload functionality",
    "Create a dashboard with real-time metrics",
    "Fix the CORS issues in the API",
    "Add support for multiple languages",
    "Implement OAuth2 authentication",
    "Create a notification system",
    "Fix the date parsing bug",
    "Add export to CSV functionality",
    "Implement search with fuzzy matching",
    "Run npm install",
    "Run the tests",
    "Build the project",
    "Deploy to staging",
    "Check git status",
    "Create a new branch for the feature",
    "Merge the pull request",
    "Review the code changes",
    "Update the dependencies",
    "Fix the linting errors",
]

# Commands that might fail
FAIL_COMMANDS = [
    ["npm", "test"],
    ["pytest", "-v"],
    ["cargo", "build"],
    ["make", "test"],
    ["./run_tests.sh"],
    ["python", "setup.py", "test"],
    ["go", "test", "./..."],
]

# Commands that succeed
SUCCESS_COMMANDS = [
    ["echo", "hello"],
    ["ls", "-la"],
    ["pwd"],
    ["cat", "README.md"],
    ["git", "status"],
    ["date"],
]


def generate_uuid() -> str:
    """Generate a UUID in the format Codex uses."""
    return str(uuid.uuid4())


def generate_timestamp(base_time: datetime, offset_seconds: int = 0) -> str:
    """Generate an ISO timestamp."""
    t = base_time + timedelta(seconds=offset_seconds)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def create_session_meta(thread_id: str, timestamp: str, cwd: str, source: str, model: str) -> dict:
    """Create a session_meta record."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": 1,
        "timestamp": timestamp,
        "type": "session_meta",
        "payload": {
            "meta": {
                "id": thread_id,
                "timestamp": timestamp,
                "cwd": cwd,
                "originator": "codex_cli",
                "cli_version": "0.1.0",
                "source": source,
                "model_provider": "openai",
            },
            "git": {
                "repository_url": f"https://github.com/user/project-{random.randint(1, 100)}",
                "commit_hash": uuid.uuid4().hex[:40],
                "branch": random.choice(["main", "develop", "feature/new-feature", "fix/bug-fix"]),
            },
        },
    }


def create_turn_context(thread_id: str, seq: int, turn_id: str, timestamp: str, model: str) -> dict:
    """Create a turn_context record."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "turn_context",
        "payload": {
            "turn_id": turn_id,
            "approval_policy": random.choice(["on-failure", "always", "never"]),
            "sandbox_policy": "workspace-write",
            "model": model,
        },
    }


def create_user_turn(thread_id: str, seq: int, turn_id: str, timestamp: str, message: str) -> dict:
    """Create a user turn submission."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "submission",
        "payload": {
            "id": turn_id,
            "op": {
                "type": "user_turn",
                "items": [{"type": "text", "text": message}],
            },
        },
    }


def create_exec_begin(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, command: list) -> dict:
    """Create an exec command begin event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "exec_command_begin",
                "call_id": call_id,
                "command": command,
                "cwd": "/workspace",
            },
        },
    }


def create_exec_end(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, exit_code: int, stdout: str = "", stderr: str = "") -> dict:
    """Create an exec command end event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "exec_command_end",
                "call_id": call_id,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
        },
    }


def create_exec_approval_request(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, command: list) -> dict:
    """Create an exec approval request event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "exec_approval_request",
                "call_id": call_id,
                "command": command,
            },
        },
    }


def create_exec_approval(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, approved: bool) -> dict:
    """Create an exec approval submission."""
    decision = {"type": "approved"} if approved else {"type": "denied"}
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "submission",
        "payload": {
            "id": turn_id,
            "op": {
                "type": "exec_approval",
                "id": call_id,
                "decision": decision,
            },
        },
    }


def create_patch_begin(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, files: list) -> dict:
    """Create a patch apply begin event."""
    changes = {f: {"type": "update"} for f in files}
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "patch_apply_begin",
                "call_id": call_id,
                "changes": changes,
            },
        },
    }


def create_patch_end(thread_id: str, seq: int, turn_id: str, timestamp: str, call_id: str, success: bool) -> dict:
    """Create a patch apply end event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "patch_apply_end",
                "call_id": call_id,
                "success": success,
            },
        },
    }


def create_compaction_boundary(thread_id: str, seq: int, turn_id: str, timestamp: str) -> dict:
    """Create a compaction boundary record."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "compaction_boundary",
        "payload": {
            "turn_id": turn_id,
            "reason": "context_limit",
            "pre_tokens": random.randint(50000, 100000),
            "post_tokens": random.randint(10000, 30000),
        },
    }


def create_error_event(thread_id: str, seq: int, turn_id: str, timestamp: str, message: str) -> dict:
    """Create an error event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "error",
                "payload": {
                    "message": message,
                },
            },
        },
    }


def create_warning_event(thread_id: str, seq: int, turn_id: str, timestamp: str, message: str) -> dict:
    """Create a warning event."""
    return {
        "schema_version": "codex_trace_spine_v1",
        "thread_id": thread_id,
        "seq": seq,
        "timestamp": timestamp,
        "type": "event",
        "payload": {
            "id": turn_id,
            "msg": {
                "type": "warning",
                "payload": {
                    "message": message,
                },
            },
        },
    }


def generate_success_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a successful session with fast completion and zero denials."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    # Session meta
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    # 1-2 turns with successful execution
    num_turns = random.randint(1, 2)
    for turn_num in range(num_turns):
        turn_id = f"turn-{turn_num}"
        offset = turn_num * 10
        
        records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, offset), model))
        seq += 1
        
        records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 1), random.choice(USER_REQUESTS)))
        seq += 1
        
        # Successful command or patch
        if random.random() < 0.5:
            call_id = f"call-{turn_num}"
            cmd = random.choice(SUCCESS_COMMANDS)
            records.append(create_exec_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 2), call_id, cmd))
            seq += 1
            records.append(create_exec_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, 0, "Success"))
            seq += 1
        else:
            call_id = f"patch-{turn_num}"
            files = [f"src/file{i}.py" for i in range(random.randint(1, 3))]
            records.append(create_patch_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 2), call_id, files))
            seq += 1
            records.append(create_patch_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, True))
            seq += 1
    
    return records


def generate_failure_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a session with exec failures."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    # 2-4 turns with failures
    num_turns = random.randint(2, 4)
    for turn_num in range(num_turns):
        turn_id = f"turn-{turn_num}"
        offset = turn_num * 15
        
        records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, offset), model))
        seq += 1
        
        records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 1), random.choice(USER_REQUESTS)))
        seq += 1
        
        # Failed command
        call_id = f"call-{turn_num}"
        cmd = random.choice(FAIL_COMMANDS)
        records.append(create_exec_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 2), call_id, cmd))
        seq += 1
        
        exit_code = random.choice([1, 1, 1, 2, 127, 255])
        stderr = random.choice([
            "Error: Command failed",
            "FAILED: 3 tests failed",
            "error[E0382]: borrow of moved value",
            "ModuleNotFoundError: No module named 'foo'",
            "npm ERR! code ENOENT",
        ])
        records.append(create_exec_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, exit_code, "", stderr))
        seq += 1
    
    return records


def generate_denial_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a session with repeated denials."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    turn_id = "turn-0"
    records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, 1), model))
    seq += 1
    
    records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, 2), random.choice(USER_REQUESTS)))
    seq += 1
    
    # 3-5 denial cycles
    num_denials = random.randint(3, 5)
    for i in range(num_denials):
        call_id = f"call-{i}"
        offset = 5 + i * 5
        cmd = random.choice([["rm", "-rf", "/"], ["sudo", "rm", "-rf", "/*"], ["curl", "http://malicious.com"]])
        
        records.append(create_exec_approval_request(thread_id, seq, turn_id, generate_timestamp(base_time, offset), call_id, cmd))
        seq += 1
        
        records.append(create_exec_approval(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 2), call_id, False))
        seq += 1
    
    # User gives up
    records.append(create_user_turn(thread_id, seq, "turn-1", generate_timestamp(base_time, 50), "Fine, I'll do it manually"))
    seq += 1
    
    return records


def generate_compaction_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a long session with multiple compactions."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    # Many turns with compactions interspersed
    num_turns = random.randint(8, 15)
    compaction_points = sorted(random.sample(range(3, num_turns), random.randint(2, 4)))
    
    for turn_num in range(num_turns):
        turn_id = f"turn-{turn_num}"
        offset = turn_num * 20
        
        # Check for compaction
        if turn_num in compaction_points:
            records.append(create_compaction_boundary(thread_id, seq, turn_id, generate_timestamp(base_time, offset)))
            seq += 1
            records.append(create_warning_event(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 1), "Context truncated due to length"))
            seq += 1
        
        records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 2), model))
        seq += 1
        
        records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), random.choice(USER_REQUESTS)))
        seq += 1
        
        # Some activity
        if random.random() < 0.7:
            call_id = f"call-{turn_num}"
            cmd = random.choice(SUCCESS_COMMANDS + FAIL_COMMANDS)
            records.append(create_exec_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, cmd))
            seq += 1
            exit_code = 0 if cmd in SUCCESS_COMMANDS else random.choice([0, 0, 1])
            records.append(create_exec_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 8), call_id, exit_code))
            seq += 1
    
    return records


def generate_error_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a session with errors."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    turn_id = "turn-0"
    records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, 1), model))
    seq += 1
    
    records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, 2), random.choice(USER_REQUESTS)))
    seq += 1
    
    # Error event
    error_messages = [
        "Request timeout after 30 seconds",
        "Rate limit exceeded",
        "Model overloaded, please retry",
        "Connection reset by peer",
        "Internal server error",
    ]
    records.append(create_error_event(thread_id, seq, turn_id, generate_timestamp(base_time, 5), random.choice(error_messages)))
    seq += 1
    
    return records


def generate_mixed_session(thread_id: str, base_time: datetime, model: str, source: str) -> list[dict]:
    """Generate a session with mixed patterns."""
    records = []
    seq = 1
    
    cwd = f"/Users/dev/projects/project-{random.randint(1, 100)}"
    
    records.append(create_session_meta(thread_id, generate_timestamp(base_time), cwd, source, model))
    seq += 1
    
    num_turns = random.randint(3, 6)
    for turn_num in range(num_turns):
        turn_id = f"turn-{turn_num}"
        offset = turn_num * 20
        
        records.append(create_turn_context(thread_id, seq, turn_id, generate_timestamp(base_time, offset), model))
        seq += 1
        
        records.append(create_user_turn(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 1), random.choice(USER_REQUESTS)))
        seq += 1
        
        # Random activity
        activity = random.choice(["exec_success", "exec_fail", "patch_success", "patch_fail", "denial"])
        call_id = f"call-{turn_num}"
        
        if activity == "exec_success":
            cmd = random.choice(SUCCESS_COMMANDS)
            records.append(create_exec_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, cmd))
            seq += 1
            records.append(create_exec_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, 0))
            seq += 1
        elif activity == "exec_fail":
            cmd = random.choice(FAIL_COMMANDS)
            records.append(create_exec_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, cmd))
            seq += 1
            records.append(create_exec_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, 1))
            seq += 1
        elif activity == "patch_success":
            files = [f"src/file{i}.py" for i in range(random.randint(1, 3))]
            records.append(create_patch_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, files))
            seq += 1
            records.append(create_patch_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, True))
            seq += 1
        elif activity == "patch_fail":
            files = [f"src/file{i}.py" for i in range(random.randint(1, 3))]
            records.append(create_patch_begin(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, files))
            seq += 1
            records.append(create_patch_end(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, False))
            seq += 1
        elif activity == "denial":
            cmd = ["rm", "-rf", "/tmp/test"]
            records.append(create_exec_approval_request(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 3), call_id, cmd))
            seq += 1
            records.append(create_exec_approval(thread_id, seq, turn_id, generate_timestamp(base_time, offset + 5), call_id, random.choice([True, False])))
            seq += 1
    
    return records


def write_session(output_dir: Path, thread_id: str, records: list[dict]) -> None:
    """Write a session's records to disk."""
    session_dir = output_dir / thread_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Write records to segment file
    segment_file = session_dir / "segment-000.jsonl"
    with open(segment_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    # Write metadata
    meta_file = session_dir / "segment-000.meta.json"
    meta = {
        "segment_id": 0,
        "thread_id": thread_id,
        "record_count": len(records),
        "created_at": records[0]["timestamp"] if records else datetime.utcnow().isoformat(),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    # Create artifacts directory
    (session_dir / "artifacts").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trace spine data for demo")
    parser.add_argument("--output", type=Path, default=Path.home() / ".codex" / "trace_spine",
                        help="Output directory for trace spines")
    parser.add_argument("--count", type=int, default=500, help="Number of sessions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Session type distribution (weighted)
    session_types = [
        ("success", generate_success_session, 0.35),      # 35% success
        ("failure", generate_failure_session, 0.20),      # 20% failures
        ("denial", generate_denial_session, 0.15),        # 15% denials
        ("compaction", generate_compaction_session, 0.10),# 10% long/compacted
        ("error", generate_error_session, 0.05),          # 5% errors
        ("mixed", generate_mixed_session, 0.15),          # 15% mixed
    ]
    
    # Build weighted list
    weighted_types = []
    for name, generator, weight in session_types:
        weighted_types.extend([(name, generator)] * int(weight * 100))
    
    # Generate sessions spread over the last 30 days
    base_date = datetime.utcnow() - timedelta(days=30)
    
    print(f"Generating {args.count} synthetic sessions...")
    print(f"Output directory: {args.output}")
    
    stats = {name: 0 for name, _, _ in session_types}
    
    for i in range(args.count):
        # Random timestamp within last 30 days
        time_offset = random.uniform(0, 30 * 24 * 3600)
        session_time = base_date + timedelta(seconds=time_offset)
        
        # Generate UUID
        thread_id = generate_uuid()
        
        # Pick session type
        session_name, generator = random.choice(weighted_types)
        stats[session_name] += 1
        
        # Pick model and source
        model = random.choice(MODELS)
        source = random.choice(SOURCES)
        
        # Generate session
        records = generator(thread_id, session_time, model, source)
        
        # Write to disk
        write_session(args.output, thread_id, records)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{args.count} sessions...")
    
    print(f"\nDone! Generated {args.count} sessions:")
    for name, count in stats.items():
        print(f"  {name}: {count}")
    print(f"\nTotal records written to: {args.output}")


if __name__ == "__main__":
    main()
