# Codex-Native Closed Loop: Automatic Task Mining -> Signals -> Harbor Evals -> Measured Improvement

This document describes an end-to-end system that is built directly into Codex, across every surface where Codex runs. When any user starts a Codex session (VS Code, CLI, or other protocol-backed clients), Codex captures a faithful trace of what happened, ships it to a central system, interprets it using a hybrid embeddings-and-coding architecture, distills high-value segments into Harbor eval tasks, and uses those evals to measure whether changes to Codex actually improved the experience. Over time, the system discovers new facets and new friction/delight categories, promotes them into stable schemas when they recur, and continuously increases the quantity and quality of eval coverage with minimal human effort.

The core principle is that normal daily usage of Codex becomes the primary source of evaluation tasks, and evaluation becomes the primary way Codex improves safely and measurably.

The overarching goal is recursive self-improvement: Codex analyzes its own behavior, mines tasks from real usage, turns those tasks into evals, and evolves autonomously in an eval-driven way.

---

## System overview

This is a single closed loop with seven stages:
1. Capture a complete local trace spine from rollouts and ops.
2. Ship segments to a central raw lake with deterministic ordering.
3. Build a canonical spine and indexes for analytics and embeddings.
4. Compute facets and experience signals; discover new schema via clustering.
5. Distill trace segments into Harbor tasks and datasets.
6. Run evals to measure changes and detect regressions.
7. Create work artifacts, implement fixes, and re-evaluate.

---

## Design invariants

These are non-negotiable properties of the system:
- The trace spine is faithful and includes both rollouts and explicit ops decisions.
- Compaction is preserved as a reconstructible boundary with pre- and post-compaction history.
- Provenance is preserved so sub-agent activity does not contaminate attribution.
- The spine retains enough raw operational detail to synthesize Harbor tasks without guesswork.
- Capture is append-only, local-first, and ships asynchronously without user-facing latency.

---

## Capture and trace spine

A Codex session produces a rich timeline of structured items: session metadata, per-turn context, model responses, tool calls, tool outputs, diffs and plan updates, warnings and errors, compactions, sub-agent activity, and collaboration events. Separately, Codex receives explicit user decisions as protocol operations, most importantly approvals and denials for command execution and patch application. Those decisions are not just events; they are the user's intent made concrete. This timeline is the raw material for task mining, so every meaningful real-world session should be capturable as a potential Harbor task.

Because this system is built into Codex everywhere, the capture path must be faithful, universal, and low friction. It must work across session sources such as CLI and VS Code, and it must treat sub-agent sessions (review, compaction, thread spawn) as first-class provenance. It must capture explicit approval decisions as protocol operations, not merely infer them from the item stream, and it must capture the approval request metadata (call id, turn id, command, cwd, patch changes, and any proposed policy amendments) so decisions can be joined reliably.

The rollout persistence policy must include the events we depend on for analysis and distillation: warnings and errors, plan updates, diffs, tool execution begin/end, patch apply begin/end, approval requests, and collaboration events. If these are filtered out at capture time, the trace spine is incomplete and we cannot reliably build Harbor tasks or measure friction.

The trace spine is the merged, ordered union of rollouts and ops. Rollouts capture session metadata, per-turn context, response items, events, and compaction markers. Per-turn context is especially important because it includes policies and constraints such as approval policy, sandbox policy, model identity, truncation policy, and optional per-turn output schema. Ops capture user intent as decisions, including session-scoped approvals and policy amendments. The trace spine must retain enough raw operational detail to enable Harbor distillation: exec command begin/end, patch apply begin/end, tool outputs, and plan/diff artifacts. When we later synthesize a Harbor task, we need to reconstruct the instruction, environment setup, and verifier expectations without ambiguity.

The trace spine is stored locally as an append-only record while the session runs, and it is shipped as segments to the central server. Segmenting matters because sessions can be long and because we want resilience to process crashes and connectivity failures. The spine is treated as immutable history; downstream logic is derived from it and can be recomputed.

---

## Compaction as a first-class boundary

Codex compaction replaces conversation history with a summary so the session can continue within context limits. This creates an explicit tension between interactive usability and trace fidelity, especially if we want ATIF-grade or training-grade traces later.

If we do nothing beyond recording "a compaction happened," then the trace may no longer contain what was actually removed, and we lose the ability to reconstruct what the model previously saw. That breaks faithful replay, reliable distillation, and clean ATIF export.

The trace spine must treat compaction as a boundary with preservation. At the moment compaction occurs, before history is replaced, the system must persist the pre-compaction segment that is being removed, or it must persist stable references that allow reconstruction of those raw items later. The simplest robust rule is write-ahead preservation: store the raw history segment that will be replaced, store the compaction summary, and store a boundary record that links them. That boundary record becomes a deterministic segmentation point for later exports and analysis.

We should preserve both pre-compaction history and the post-compaction summary so we can render both views. Storing only the summary or only the new history is insufficient for faithful replay and for reliable Harbor distillation.

### Harbor trajectory export note (ATIF v1.4)

Harbor’s trajectory tooling uses the Agent Trajectory Interchange Format (ATIF), a standardized JSON format for logging the complete interaction history of autonomous LLM agents (messages, tool calls and observations, per-step metrics, and multi-agent delegation). For this system:

- **Schema version**: emit `schema_version = "ATIF-v1.4"` in exported trajectories.
- **Compaction / context resets**: ATIF exports must not pretend that post-compaction work is a continuation of the pre-compaction chat history. Represent the reset explicitly so the true LLM context can be reconstructed later. A practical approach (used by Harbor-integrated agents) is to split trajectories at each reset into separate linear files (e.g. a pre-compaction `trajectory.json` plus `trajectory.cont-1.json`, `trajectory.cont-2.json`, …) rather than emitting a single file with an ambiguous mid-stream “handoff”.
- **Validation**: validate exported files against the ATIF schema before using them for analysis, SFT, or RL:

```bash
python -m harbor.utils.trajectory_validator trajectory.json
```

---

## Provenance and multi-agent lineage

Codex supports multiple session sources, including sub-agents used for review, compaction, and thread spawns. If we do not preserve this provenance, we will misattribute friction and delight. For example, an apparent tone change or thread reset might actually be a compaction sub-agent workflow rather than a user-driven change. Similarly, a review sub-agent might generate content that should not be interpreted as the user's confusion or the main agent's failure.

Every item in the trace spine is tagged with provenance. The provenance includes the session source (CLI, VS Code, Exec, MCP, or SubAgent), whether the content came from the main agent, a sub-agent and which type (review, compact, thread spawn), the user, the UI, or a tool/environment observation, and the thread lineage if a sub-thread was spawned. This provenance is preserved through ETL and becomes a stable facet axis later.

---

## Shipping and central ingestion

The built-in system writes trace spine segments locally first, then ships them asynchronously to a central server. Shipping is resumable and idempotent. A session can be uploaded in multiple parts, and the central server can reassemble it deterministically based on session IDs and ordering keys. Ordering is preserved through a combination of monotonic sequence keys and timestamps. The shipper never blocks user interaction, and failures to ship do not prevent the session from continuing.

On the central server, ingestion stores raw segments as immutable artifacts. This includes the rollout JSONL, the ops decisions, and the compaction-preserved raw segments. Nothing in the raw lake is rewritten. If analysis changes, we re-derive.

From the raw lake we build derived canonical representations. The key artifact is a canonical session spine that is queryable, indexed, and joins rollouts with ops. It normalizes items into a small number of types: session metadata, turn contexts, responses, tool calls and tool results, diffs and plan updates when available, approvals and denials, warnings and errors, and compaction boundaries that link pre- and post-compaction history.

Because Codex local indexing is optional, the central system provides the indexing that matters for longitudinal analysis. It indexes by time, release, session source, model configuration, sandbox and approval policy, tool names, error classes, compaction counts, and any other stable facets we rely on downstream. It also supports vector indexing for embeddings.

---

## Analysis: facets and experience signals

Facets exist so we can ask stable questions across thousands or millions of sessions, like "which sandbox policy correlates with the most denials," or "did a new release increase compactions for CLI sessions." There are spine facets and derived facets.

Spine facets come directly from session metadata and per-turn context and therefore are high-trust and stable. They include session source, sub-agent provenance, model identity/provider, approval policy, sandbox policy and parameters, truncation policy, presence of structured output schema, session length measures, tool usage measures, compaction count, and thread lineage depth.

Derived facets capture semantic meaning that is not directly encoded as a field. They include intent categories, programming languages, frameworks, workflow patterns like branch switching, counts of tool approvals and denials, and higher-level archetypes like success vs abandonment. These are computed by the hybrid analysis layer and are versioned because their definitions evolve.

The system computes experience signals from the canonical trace spine. A signal is not a vibe; it is a structured event with a type, a severity, a scope, and evidence pointers back to exact trace items.

Friction signals are grounded in objective mechanics wherever possible. Approvals and denials are central because they express user trust and interruption directly. Tool failures, stream errors, sandbox errors, timeouts, repeated compactions, truncation warnings, repeated patch rejections, and oscillating diffs are strong mechanical indicators. Plan instability and repeated backtracking can be inferred from plan updates and diffs when those are available. Because Codex exposes rich policies and context per turn, we can attribute friction to conditions, not just outcomes; for example, friction under a read-only sandbox policy looks different than friction under full access.

The friction analyzer also scans for user struggle patterns that require semantic interpretation: repeated rephrasing, escalation in tone, explicit frustration, tool call rejections clustered in a short window, or repeated attempts at the same action. Each friction moment gets a severity rating and abstracted citations that describe what happened. Those citations must preserve enough non-redacted context to synthesize Harbor tasks (instruction, environment, and verifier expectations), or we risk losing the ability to create evals from the very signals we detected.

Traditional product analytics cannot reliably capture these patterns because they live in natural language, multi-step tool flows, and context-dependent behaviors rather than in a small set of predefined events. The self-discovery system uses LLM judgment over the trace spine to recognize these nuanced, cross-turn patterns, making them observable and actionable at scale.

Example friction patterns we should detect:
- Error events: model errors, tool failures, timeouts.
- Repeated rephrasing: >=3 consecutive restating messages.
- Escalation tone: "broken", "why isn't", "frustrating".
- Platform confusion: questions about Codex features or capabilities.
- Abandoned tool flow: tool calls rejected or cancelled.
- Backtracking: "undo", "revert", deleting code.
- Context churn: add/remove the same file repeatedly.

Delight signals are computed symmetrically. Fast convergence, first-attempt completion under strict policies, clean monotonic plan completion, low retry rates, low denial rates, positive exclamations, explicit mentions of time saved, and rapid approval flows followed by appreciation all indicate delight. Delight is treated as a first-class outcome because it tells us what to preserve, not just what to fix.

Signals are computed at two granularities. They are computed at the moment level, where a localized event is attached to a small span of turns. They are also computed at the session level, where a session can be labeled with an overall experience profile such as "success with high friction" or "success with high delight."

---

## Hybrid discovery and schema evolution

A universal always-on system must both discover new patterns and track stable patterns over time. Embeddings provide unsupervised discovery. They let us group similar sessions and similar friction moments without knowing ahead of time what categories will matter. This is how we discover new facets and new signal types that the schema does not yet contain, and how we detect emergent workflows after new features ship.

Coding provides stability and action. Coding means we define named facets and named signal types with inclusion criteria, severity rules, and recommended remediation routes. This makes the system measurable and governable. You can set thresholds, alert on changes, and build dashboards that do not drift.

The hybrid loop works like this. We continuously embed and cluster sessions and localized signal moments using representations built from the canonical spine and from concise derived summaries. We then look for clusters that are coherent, recurring, and poorly explained by the current schema. When such a cluster appears, the system proposes a new dimension: it describes the invariants that unify the cluster and offers candidate names and definitions for a new facet or a new signal type. Humans do not need to read raw sessions to evaluate this; they can review abstracted invariants and evidence pointers. When a proposal is accepted, it becomes part of the stable schema. From then on, coding rules classify future sessions into that schema, and trend lines become meaningful.

When embedding clusters reveal recurring patterns that do not fit the current schema, the system proposes new candidate facets and friction/delight types (open coding). It then consolidates and formalizes them into stable, named dimensions with crisp definitions, relationships, and thresholds used for analytics and action routing (axial/selective coding).

---

## Thresholding and decisioning

Signals and facets become useful when they turn into decisions. The decision layer determines when to act and what action to take.

Patterns become actionable when they cross thresholds defined in terms of frequency, severity, and impact. Frequency means the pattern appears across enough sessions to matter. Severity means the pattern is strongly negative or strongly positive. Impact means it correlates with outcomes like abandonment, repeated denials, session failure, or a sharp drop in delight. There is also a novelty component, because new clusters that appear suddenly after a release deserve attention even before they reach long-term frequency thresholds.

When thresholds are crossed, the system chooses an action type. Sometimes the right action is to create or expand eval coverage. Sometimes it is to file a product bug or a policy adjustment. Sometimes it is to change agent behavior. The important principle is that actions are tied back to measurable eval sets so that "fixes" become testable hypotheses.

---

## Harbor distillation and evals

Harbor is the execution and measurement backbone. Distillation is how we turn organic real-world Codex usage into Harbor tasks at scale, with minimal human effort.

### Harbor primer (terminology + CLI surface this system relies on)

Harbor is a framework for evaluating agents and models in container environments.

- **Task**: an instruction, a container environment, and a verifier/test script (the trial produces a reward).
- **Dataset**: a collection of tasks (either a local directory or a registry entry that points to versioned tasks in git).
- **Trial**: one agent attempt at one task; essentially a rollout that produces a reward.
- **Job**: a collection of trials, potentially spanning tasks, datasets, agents, and models.

Harbor entry points used by the closed loop:

```bash
harbor datasets list
harbor run -d "dataset@version" -a "<agent>" -m "<model>"
harbor run -p "<path/to/dataset>" -a "<agent>" -m "<model>"
harbor run -p "<path/to/task>" -a "<agent>" -m "<model>"
harbor run -c "<path/to/job.yaml>"  # optional explicit job config
```

For horizontal scaling (useful for large batched eval runs), Harbor can run trials on cloud sandbox providers (e.g. Daytona) and parallelize aggressively:

```bash
harbor run -d "dataset@version" -a "<agent>" -m "<model>" -e daytona -n 32
```

Most cloud sandbox providers are single-container only. Multi-container tasks (docker compose) generally require the local Docker runtime.

If Harbor isn’t already installed, a common setup is:

```bash
uv tool install harbor
```

Distillation starts from a trace segment, not from an entire session. A segment is a bounded window of turns that contains a clear intent and a clear success or failure. Segments are selected primarily from clusters of recurring friction and from high-value delight patterns we want to protect. Selecting segments rather than sessions makes tasks minimal, reproducible, and less noisy.

The distiller uses trace artifacts that are already structured. Diffs and plan updates reveal what changed and what the agent was trying to do. Tool calls and tool outputs reveal the operational path. Approvals and denials reveal what was allowed. The per-turn context reveals constraints. Using these, the distiller can synthesize a Harbor task that captures the essence of the scenario without carrying along irrelevant baggage.

A Harbor task consists of an instruction, an environment, and a verifier. Concretely, a Harbor task is a directory containing `instruction.md`, `task.toml` (metadata and config), `environment/` (typically a Dockerfile), and `tests/test.sh` that writes a reward file to `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`. An optional `solution/solve.sh` can be included for oracle verification. The instruction is derived from the intent of the segment and is written as a minimal definition of done. The environment recreates the starting state and dependencies. The verifier is the test script that checks whether the instruction was satisfied and emits the reward. The task metadata records the provenance: which cluster and which signal type this task represents, and under which facet conditions it arises.

### Harbor task format details (distillation output contract)

Harbor tasks are filesystem directories. Harbor can scaffold a new task directory with:

```bash
harbor tasks init "<task-name>"
```

A canonical task directory looks like:

```text
<task-name>/
  instruction.md
  task.toml
  environment/
    Dockerfile                  # or docker-compose.yaml for local docker multi-container
  tests/
    test.sh
  solution/                      # optional, used by oracle sanity checks
    solve.sh
```

#### `task.toml`: config + metadata

`task.toml` carries both configuration and arbitrary metadata. Distilled tasks should use metadata to preserve provenance (e.g., cluster identifiers, signal type, facet predicates, and source trace pointers), and use config to enforce bounded evaluation.

Example structure:

```toml
version = "1.0"

[metadata]
author_name = "..."
author_email = "..."
cluster_id = "..."
signal_type = "..."
tags = ["..."]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
docker_image = "some-org/some-name:some-tag"  # optional; can be null if building from Dockerfile
cpus = 1
memory_mb = 2048
storage_mb = 10240
```

#### `environment/`: container definition + special paths

Harbor does not require a specific file name inside `environment/` in the abstract, but individual environment runtimes do. For example, a Docker runtime typically expects either `environment/Dockerfile` or `environment/docker-compose.yaml`. Most cloud sandbox providers support Dockerfile-defined environments and not docker compose.

Harbor reserves a few special paths inside the container filesystem:

- **`/logs/verifier/`**: verifier output, including reward files (downloaded to host after the run)
- **`/logs/agent/`**: an agent-writable log directory
- **`/tests/`**: the `tests/` folder copied in by the Harbor harness and executed from the working directory
- **`/solution/`**: the `solution/` folder copied in by the Oracle agent (if present)

#### `tests/test.sh`: verifier contract + reward emission

`tests/test.sh` should install any test dependencies and verify the instruction was satisfied. It must produce a reward file under `/logs/verifier/`:

- **`/logs/verifier/reward.txt`**: plain text single integer/float (typically `1` for success, `0` for failure)
- **`/logs/verifier/reward.json`**: JSON object of float/int metrics (Harbor reads `reward.txt` by default and falls back to `reward.json`)

We recommend using absolute paths in verifiers to avoid working-directory ambiguity.

Distillation includes quality gates. The environment must build. The verifier must be deterministic. The reward must be meaningful. The task must be minimal enough to be run repeatedly. When possible, an oracle solve script can be generated or supplied to sanity-check solvability, but the system is designed so that tasks can still be valuable without oracle scripts as long as the verifier is correct.

Practical Harbor checks that fit these quality gates:

```bash
# If solution/solve.sh exists, use the oracle agent to sanity-check solvability.
harbor run -p "<path/to/task>" -a oracle

# For quick interactive debugging of the container environment.
harbor tasks start-env -p "<path/to/task>" -e docker -a -i
```

Trace segments should also be exportable to ATIF v1.4 trajectories so Harbor's trajectory tooling and validation can be used for analysis, RL, and debugging.

### Harbor traces export (optional: turning trials into SFT-ready datasets)

Harbor can export conversational traces from trial directories into SFT-friendly datasets, but this requires ATIF trajectories (i.e., agents that write `trajectory.json` in ATIF format).

```bash
harbor traces export \
  --path "<path/to/trials-or-jobs-dir>" \
  --recursive \
  --episodes last \
  --filter success \
  --sharegpt
```

Harbor can also export traces automatically after `harbor run` finishes (use `--export-traces` and related `--export-*` flags) so the eval loop can directly feed training/debug pipelines when desired.

Once tasks are generated, they are grouped into datasets that represent specific patterns. A dataset can represent a recurring friction cluster, a specific regression pattern, or a category of delight. In Harbor terms, a trial is a rollout that produces a reward, and a job is a collection of trials across tasks, datasets, agents, and models. These datasets become the eval levers we use to measure improvement.

### Harbor datasets + registries (how distilled tasks become runnable eval sets)

Harbor supports:

- **Local datasets**: directories containing task directories.
- **Registry datasets**: versioned dataset definitions (JSON) that point to tasks stored in git repositories at specific commits.

Run a local dataset:

```bash
harbor run -p "<path/to/dataset>" -a "<agent>" -m "<model>"
```

Run a dataset from a registry:

```bash
harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>"
```

A dataset registry entry has the rough shape:

```json
{
  "name": "my-dataset",
  "version": "1.0",
  "description": "A description of the dataset",
  "tasks": [
    {
      "name": "task-1",
      "git_url": "https://github.com/my-org/my-dataset.git",
      "git_commit_id": "1234567890",
      "path": "task-1"
    }
  ]
}
```

Because this system will often generate private, rapidly-updated datasets, Harbor supports pointing to a custom registry file:

```bash
harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>" --registry-path "<path/to/registry.json>"
```

Or hosting the registry JSON and referencing it by URL:

```bash
harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>" --registry-url "<url/to/registry.json>"
```

### Harbor job artifacts (where rewards + trajectories land)

Running `harbor run` creates a job directory (by default under `jobs/`) containing both job-level and trial-level artifacts:

```text
jobs/<job-name>/
  config.json
  result.json
  <trial-name>/
    config.json
    result.json
    agent/
      trajectory.json
    verifier/
      reward.txt
      test-stdout.txt
      test-stderr.txt
```

These artifacts are the join points back to the originating clusters/facets and the raw material for debugging regressions (verifier logs), tracking success metrics (reward files), and exporting trajectories (ATIF).

---

## Measurement and regression detection

The measurement loop is straightforward once distillation exists. Whenever Codex changes, we run Harbor evals. We do not only run a global benchmark; we run targeted datasets that represent the patterns we care about. If we are fixing a "patch denial churn" issue, we run the dataset that contains tasks distilled from that pattern under the relevant approval and sandbox policies. We also run a neighborhood set of tasks from adjacent clusters because fixes often shift failures elsewhere. We also run a stable baseline set to detect broad regressions.

The output of eval runs is used to update trend lines for both task-level metrics and experience-level signals. This is where the system becomes real: we stop arguing about whether a change "should" help and instead measure whether it does. The reward files emitted by verifiers become the canonical quantitative signals for task success, and those results are linked back to the originating clusters and facets.

---

## Daily batch pipeline: scalable discovery and reporting

The system runs as a daily batch process designed for scale and cost efficiency. Sessions from the past twenty-four hours are fetched from BigQuery, filtered to those with at least thirty agentic steps to ensure meaningful interactions, and sent to OpenAI's batch API. The batch size is dynamically adjusted to a token budget so we can analyze thousands of sessions per day at lower cost. Results flow back to BigQuery for longitudinal analysis and to Slack for daily reports. BigQuery tables let us correlate friction and delight patterns with releases and feature changes; Slack reports give the team a daily pulse on how users experienced Codex.

This is the stage where we capture and create robust Harbor evals using the richest trace data available.

---

## Closing the loop: tickets, PRs, and re-evaluation

Once thresholds are crossed, the system can create work artifacts that are natively linked back to evidence and measurement. A ticket is not just "users are unhappy." A ticket contains the cluster description, the dominant facet conditions, evidence pointers into representative trace spines, and the Harbor dataset that will be used to validate the fix.

Implementation then happens through normal engineering workflows. Codex can generate a plan, propose code changes, or modify agent behavior. Codex can also review its own PRs. What matters is that after implementation, the system reruns the associated Harbor eval sets and produces a clear before/after measurement. If the targeted metrics improved and no unacceptable regressions occurred, the system can mark the issue resolved with evidence. If not, the system has learned something important: the fix did not address the true failure mode, and the cluster definition or remediation hypothesis needs adjustment.

This is recursive self-improvement grounded in evals rather than vibes.

---

## Designing for an optional real-time future

The first version of this system runs in batch, because discovery and distillation do not need to be real-time to create huge value. However, the design intentionally keeps hooks that enable online use later.

Codex already supports per-turn structured output schemas. That makes it possible, later, to ask Codex to emit a small structured self-assessment each turn, or to emit structured features that can be evaluated online. Codex also has explicit approval operations and sandbox policies, which are immediate high-confidence friction indicators. In a future online extension, Codex could detect early signs of frustration (repeated denials, repeated tool failures, compaction churn) and intervene conservatively, for example by prompting a clarifying question, suggesting a safer command, or recommending a fresh thread when compaction begins to degrade accuracy.

The critical point is that online interventions should be built on the same trace spine and the same stable schema, not on a separate ad-hoc system, so that online behavior is measurable against the same Harbor eval sets.

In that future, detected clusters should also be correlated with observability systems (backend error rates, latency spikes, service incidents) and change logs (deploys, feature flags, policy updates). This correlation helps surface root causes faster and provides a clean measurement loop for improvement tracking across product, model, and infrastructure changes.

---

## Conclusion

This is the complete end-to-end architecture for a Codex-native, universal, self-improving system that mines tasks from rollouts, converts them into Harbor evals, and uses those evals to drive measurable product and agent improvement.

---

## Codex internals: implementation documentation for the closed loop (trace spine → signals → Harbor tasks)

This chapter is the Codex-side implementation documentation for the system described above. It specifies exactly what Codex must capture locally, how it should be ordered and segmented, what must be shipped to the central raw lake, and how downstream systems can deterministically reconstruct Harbor tasks and ATIF trajectories.

It is intentionally concrete: it enumerates the data streams Codex produces, the join keys that make them stitchable, the minimum record model we should persist locally, and the `codex/codex-rs` hook points where capture must occur.

### How this chapter maps to the system overview stages

This section is organized to directly support the loop described above:

- **Stage 1 (Capture)**: build a faithful local trace spine by capturing **SQ + EQ + rollouts + reproduction artifacts**, with deterministic ordering and compaction boundaries.
- **Stage 2 (Ship)**: upload immutable, ordered spine segments asynchronously (resumable + idempotent).
- **Stage 3 (Canonicalize)**: merge segments into a canonical session spine and normalized indexes (e.g., BigQuery tables).
- **Stage 4 (Analyze)**: compute facets + experience signals grounded in objective spine evidence.
- **Stage 5 (Distill)**: convert a spine segment window into a runnable Harbor task (`instruction.md`, `environment/`, `tests/test.sh`, `task.toml`).
- **Stage 6 (Measure)**: Harbor rewards/trajectories become the quantitative outcomes, linked back to the originating spine evidence.
- **Stage 7 (Close loop)**: tickets/PRs cite spine evidence + Harbor datasets and are re-evaluated automatically after fixes.

---

### Stage 1 — Capture: the Codex-native trace spine (local-first, append-only)

#### 1) Sources of truth inside Codex (SQ, EQ, rollouts, artifacts)

Codex produces four conceptually different sources of truth during a session. The trace spine must merge all of them:

- **Submission Queue (SQ)**: inbound operations from the client/UI.
  - **Type**: `codex_protocol::protocol::Submission { id, op: Op }` (`codex/codex-rs/protocol/src/protocol.rs`)
  - **Processor**: `submission_loop(...)` in `codex/codex-rs/core/src/codex.rs` (reads `rx_sub`)
  - **Why it matters**: SQ is where durable user intent lives: approvals/denials, interrupts, undo/rollback, compaction, review requests, MCP elicitation resolution, etc. Rollouts do *not* persist SQ today.

- **Event Queue (EQ)**: outbound structured events emitted for clients.
  - **Type**: `codex_protocol::protocol::Event { id, msg: EventMsg }` (`codex/codex-rs/protocol/src/protocol.rs`)
  - **Primary send path**: `Session::send_event_raw(...)` / `Session::send_event_raw_flushed(...)` (`codex/codex-rs/core/src/codex.rs`)
  - **Direct emitters**: some subsystems publish directly onto the event channel (notably exec output deltas) and bypass `send_event`.
  - **Why it matters**: EQ contains operational ground truth: exec begin/end + streaming output, patch apply begin/end, unified diffs, approval prompts, tool call begin/end, warnings/errors, stream errors, collaboration events, etc.

- **Rollout persistence**: local append-only JSONL primarily for replay/resume.
  - **Writer**: `RolloutRecorder` (`codex/codex-rs/core/src/rollout/recorder.rs`)
  - **Filter**: `codex/codex-rs/core/src/rollout/policy.rs`
  - **Why it matters**: rollouts capture model-visible conversation items and some events, but are optimized for resume and user replay rather than eval-grade fidelity.

- **Reproduction artifacts**: inputs to behavior that aren’t fully represented in messages.
  - **Per-turn prefix messages**: created by `Session::build_initial_context(...)` in `core/src/codex.rs` (policy-derived developer instructions, optional overrides, and `<environment_context>` from `core/src/environment_context.rs`).
  - **Shell snapshot**: `~/.codex/shell_snapshots/<threadId>.sh` (or `.ps1`) created by `core/src/shell_snapshot.rs` (functions/aliases/exports; affects execution determinism).
  - **Git identity + diffs**: `collect_git_info` and `git_diff_to_remote(...)` in `core/src/git_info.rs` (bridge from local state → reproducible container state).
  - **Ghost snapshots (optional)**: `ResponseItem::GhostSnapshot { ghost_commit: GhostCommit }` from `core/src/tasks/ghost_snapshot.rs` + `utils/git/src/lib.rs`.
  - **Optional state DB**: SQLite-backed runtime for local indexing/parity checks (`core/src/state_db.rs`).
  - **TUI session logger precedent**: `tui/src/session_log.rs` already logs inbound ops + outbound app events to JSONL (when enabled).

For the closed loop above, the trace spine must include the **union** of SQ + EQ + rollouts + artifacts. Rollouts alone are insufficient.

##### What “capture SQ” means in practice (Op inventory)

SQ is the durable “user/client intent” stream. Because `Op` is `#[non_exhaustive]`, the safest and simplest rule is: **persist every `Submission { id, op }` exactly as received** (no filtering), then derive higher-level semantics downstream.

The closed loop above depends most directly on these `Op` families (`codex/codex-rs/protocol/src/protocol.rs`):

- **Intent-bearing inputs**
  - `Op::UserTurn { items, cwd, approval_policy, sandbox_policy, model, effort?, summary, final_output_json_schema, collaboration_mode?, personality? }`
  - `Op::UserInput { items, final_output_json_schema? }` (legacy; still must be captured)
  - `Op::OverrideTurnContext { cwd?, approval_policy?, sandbox_policy?, windows_sandbox_level?, model?, effort?, summary?, collaboration_mode?, personality? }`
- **User trust / interruption**
  - `Op::ExecApproval { id, decision }`
  - `Op::PatchApproval { id, decision }`
  - `Op::Interrupt`
- **History and context churn (strong mechanical signals)**
  - `Op::Compact`
  - `Op::Undo`
  - `Op::ThreadRollback { num_turns }`
- **Multi-agent and tool ecosystem**
  - `Op::Review { review_request }` (may spawn a review sub-agent thread)
  - `Op::UserInputAnswer { id, response }` (resolves `request_user_input`)
  - `Op::ResolveElicitation { server_name, request_id, decision }` (MCP elicitation)
  - `Op::DynamicToolResponse { id, response }` (dynamic tools)
  - `Op::ListMcpTools`, `Op::RefreshMcpServers { ... }`, `Op::ListSkills { ... }`, etc.
- **Operational**
  - `Op::RunUserShellCommand { command }` (user-initiated shell “!cmd” path; drives exec events)
  - `Op::Shutdown`

Even when a specific op variant isn’t used directly for Harbor distillation, it often matters for the *analysis* stages (e.g., identifying abandonment, repeated rollbacks, repeated compactions, or multi-agent proxying).

#### 2) Canonical IDs and join keys (stitching the spine deterministically)

The trace spine must preserve the join keys that make SQ/EQ/rollouts and artifacts reconstructible:

- **`thread_id` / `conversation_id`** (`ThreadId`)
  - From `SessionMeta.id` (first rollout line) and many events.
  - Used for provenance (“this task came from thread X”) and for file naming.

- **`turn_id`**
  - SQ envelope id: `Submission.id`.
  - EQ envelope id: `Event.id` (universal join key because it exists on every event).

- **`call_id`** (tool call id)
  - Pairs tool call request ↔ tool call output (`ResponseItem::FunctionCall.call_id` ↔ `ResponseItem::FunctionCallOutput.call_id`).
  - Pairs exec begin/end (`ExecCommandBeginEvent.call_id` ↔ `ExecCommandEndEvent.call_id`).
  - Pairs patch apply begin/end (`PatchApplyBeginEvent.call_id` ↔ `PatchApplyEndEvent.call_id`).

- **`process_id`** (interactive exec)
  - Optional on `ExecCommandBegin/End`, required on `TerminalInteractionEvent`.
  - Groups multiple interactions/output deltas into an interactive transcript.

- **Approval join nuance (non-negotiable for correctness)**
  - Approval ops are “foreign-keyed” by the *op payload*, not the approval submission envelope:
    - `Submission { id: <op_submission_id>, op: Op::ExecApproval { id: <approved_turn_id>, decision } }`
    - `Submission { id: <op_submission_id>, op: Op::PatchApproval { id: <approved_turn_id>, decision } }`
  - Clients often set `<op_submission_id> == <approved_turn_id>`, but the protocol does not require it. Join approvals using `Op::*Approval.id` and record both ids in the spine.

- **Known capture gap today**
  - `RolloutItem::TurnContext(TurnContextItem)` does not include `turn_id`, so later joins require positional heuristics. For eval-grade traces, store a `turn_context` spine record explicitly keyed by `turn_id` (or add `turn_id` into `TurnContextItem`).

#### 3) Rollouts today: what’s persisted and why it’s insufficient

Codex writes rollouts to:

- `~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<threadId>.jsonl`

Rollout lines are `RolloutLine { timestamp, item: RolloutItem }`, where `RolloutItem` is serialized as a `{ "type": "...", "payload": ... }` tagged enum (`#[serde(tag = "type", content = "payload")]`).

The first line for a new session is always `RolloutItem::SessionMeta(SessionMetaLine)`, where:

- `SessionMetaLine.meta` is the session-scoped `SessionMeta` (id, forked_from_id, timestamp, cwd, originator, cli_version, source, model_provider, base_instructions)
- `SessionMetaLine.git` is optional `{ commit_hash, branch, repository_url }`

Each line is a `RolloutLine { timestamp, item: RolloutItem }`. `RolloutItem` has five variants:

- `SessionMeta(SessionMetaLine)`
- `TurnContext(TurnContextItem)`
- `ResponseItem(ResponseItem)`
- `Compacted(CompactedItem)`
- `EventMsg(EventMsg)`

The rollout filter today persists:

- **Response items**: `Message`, `Reasoning`, `FunctionCall`, `FunctionCallOutput`, `CustomToolCall`, `CustomToolCallOutput`, `LocalShellCall`, `WebSearchCall`, `GhostSnapshot`, `Compaction` (drops `Other`).
- **Event messages (subset)**: `UserMessage`, `AgentMessage`, `AgentReasoning`, `AgentReasoningRawContent`, `TokenCount`, `ContextCompacted`, `EnteredReviewMode`, `ExitedReviewMode`, `ThreadRolledBack`, `UndoCompleted`, `TurnAborted`.

This drops exactly the signals and mechanics the system above depends on:

- `ExecCommandBegin/End`, `PatchApplyBegin/End`, `TurnDiff`, `PlanUpdate`, `Warning`, `Error`, `StreamError`, MCP/collab begin/end, approval requests, etc.
- `ExecCommandOutputDelta` is emitted directly from exec onto the event channel, so it is not eligible for rollout persistence at all.
- User decisions are SQ ops (`Op::ExecApproval`, `Op::PatchApproval`, `Op::UserInputAnswer`, etc.) and are not persisted to rollouts.

Therefore, the closed loop requires a separate **trace spine capture of SQ + EQ** (rollouts remain a useful parallel artifact, but not sufficient).

#### 4) Trace spine record model (versioned JSONL, deterministic ordering)

To satisfy “append-only, local-first, deterministic ordering,” persist an explicit trace spine in JSONL, in parallel to rollouts.

Minimum record types (store raw payloads for fidelity):

- `session_meta`: `SessionMetaLine` (+ optional git info)
- `turn_context`: snapshot keyed by `turn_id` (policies + model selection + truncation + tool availability)
- `submission`: raw `Submission { id, op }`
- `event`: raw `Event { id, msg }`
- `artifact_ref`: shell snapshot path/hash, git diff-to-remote patch hash, ghost snapshot ids, repo root, etc.
- `compaction_boundary`: pre/post history snapshots (see below)
- `bridge`: provenance bridge records when one thread proxies another (review delegate; see below)

Suggested per-line envelope (example shape; exact schema is up to the implementation, but should be versioned and stable):

```json
{
  "schema_version": "codex_trace_spine_v1",
  "thread_id": "thread_...",
  "seq": 12345,
  "timestamp": "2026-01-29T12:34:56.789Z",
  "type": "event",
  "payload": { "id": "turn_...", "msg": { "type": "exec_command_end", "payload": { /* ... */ } } }
}
```

Notes:

- **Store raw protocol payloads** (`Submission`, `Event`, `RolloutLine`) inside `payload` for maximal future-proofing; do not “flatten away” fields you don’t currently use.
- **Segment metadata** (needed for Stage 2 shipping) should include `thread_id`, `segment_id`, `min_seq`, `max_seq`, `sha256`, and `created_at`/`closed_at`.

Two non-negotiables:

- **Ordering**: assign a writer-side monotonic `seq` to every record. Do not rely on timestamps alone.
- **Segmentability**: write bounded segments (files) so shipping is resumable and crash-safe.

#### 5) Compaction boundary preservation (write-ahead)

The system above requires compaction to be a reconstructible boundary. Today rollouts store `RolloutItem::Compacted(CompactedItem)` but not the removed history segment.

Implementation detail to account for:

- `CompactedItem` carries `message` (the summary) and may optionally carry `replacement_history: Option<Vec<ResponseItem>>` (remote compaction path).
- Neither form preserves the exact pre-compaction history that was removed from model context.

At compaction time (`core/src/compact.rs` and `core/src/compact_remote.rs`) Codex has the full current history right before replacement. Record a `compaction_boundary` spine record that includes:

- **pre-compaction history snapshot**: items in model context immediately before compaction
- **post-compaction replacement**: items in context immediately after compaction (summary and/or replacement history)
- triggering `turn_id` and compaction reason (manual `Op::Compact`, auto threshold, remote compact, etc.)

This boundary is also the clean segmentation point for ATIF export (split trajectories at each reset).

#### 6) Provenance and multi-agent lineage (sub-agents and collaboration)

Codex multi-agent behavior is implemented as real additional threads with explicit provenance:

- Provenance is stored in `SessionMeta.source` (`SessionSource`) and includes `SubAgent(SubAgentSource)`.
- `SubAgentSource` includes `Review`, `Compact`, `ThreadSpawn { parent_thread_id, depth }`, `Other(String)` (`protocol/src/protocol.rs`).

Two multi-agent surfaces matter for attribution:

- **Review sub-agent (delegate)**:
  - Spawned via `core/src/codex_delegate.rs` with `SessionSource::SubAgent(SubAgentSource::Review)`.
  - Key nuance: approvals and `request_user_input` originating inside the sub-agent are handled by the parent session and then injected into the sub-agent as ops (`Op::ExecApproval`, `Op::PatchApproval`, `Op::UserInputAnswer`).
  - To avoid heuristic joins, record a “bridge” spine record whenever proxying occurs:
    - parent thread/turn/call ids (user-facing prompt)
    - sub-agent thread/turn/call ids (originating request)
    - applied `ReviewDecision` / `RequestUserInputResponse`

- **Collaboration tool thread spawn**:
  - `collab.spawn_agent` (`core/src/tools/handlers/collab.rs`) spawns a child thread tagged as `SessionSource::SubAgent(SubAgentSource::ThreadSpawn { parent_thread_id, depth })`.
  - Parent emits explicit lifecycle events: `CollabAgentSpawnBegin/End`, `CollabAgentInteractionBegin/End`, `CollabWaitingBegin/End`, `CollabCloseBegin/End`.
  - These must be captured in the trace spine for lineage and attribution.

#### 7) End-to-end flows that must be captured (for distillation + signals)

These flows are the “ground truth backbone.” If any link is missing, distillation becomes guesswork.

##### User turn ingestion (SQ → history/rollout → EQ)

- Input arrives as `Op::UserTurn { items, cwd, approval_policy, sandbox_policy, model, effort?, summary, final_output_json_schema, ... }` (or legacy `Op::UserInput { items, final_output_json_schema? }`).
- Core builds the model-visible prefix context via `Session::build_initial_context(...)`.
- Core records the user message into history and writes `RolloutItem::TurnContext(TurnContextItem { cwd, approval_policy, sandbox_policy, model, ... truncation_policy? })`.

Harbor mapping:

- `instruction.md`: derive from user intent, not session scaffolding (exclude `<environment_context>`, etc).
- `task.toml`: record approval/sandbox policy + model configuration as provenance metadata.

##### Shell/exec tool call (tool call → optional approval → exec begin/delta/end → tool output)

1. Model emits a tool call output item (typically `ResponseItem::FunctionCall { name, arguments, call_id }`, and sometimes `ResponseItem::LocalShellCall` depending on client wiring).
2. If approval is required under the active policy, EQ emits `ExecApprovalRequestEvent { call_id, turn_id, command, cwd, reason, proposed_execpolicy_amendment, parsed_cmd }`.
3. User decision arrives via SQ: `Op::ExecApproval { id: <approved_turn_id>, decision: ReviewDecision }` (remember `Submission.id` and the approved id may differ; join using the op payload id).
4. Execution emits:
   - `ExecCommandBeginEvent` (command, cwd, parsed_cmd, source, optional `process_id`)
   - `ExecCommandOutputDeltaEvent` (0..N base64 chunks; emitted directly from exec)
   - `ExecCommandEndEvent` (stdout/stderr/aggregated_output, exit_code, duration, formatted_output, ...)
5. Tool returns `ResponseItem::FunctionCallOutput { call_id, output: { content, success? } }`.

Harbor mapping:

- Use `ExecCommandEnd.stdout/stderr/exit_code` as verifier-facing truth; `FunctionCallOutput.content` is model-facing and may be truncated.

##### apply_patch tool call (tool call → optional approval → patch apply begin/end → turn diff → tool output)

1. Model emits `FunctionCall(name="apply_patch", arguments={ input: "<patch>" }, call_id)`.
2. Patch is parsed into structured file changes, then assessed for safety under sandbox/approval policies.
3. If approval is required, EQ emits `ApplyPatchApprovalRequestEvent { call_id, turn_id, changes, reason?, grant_root? }`.
4. Decision arrives via SQ: `Op::PatchApproval { id: <approved_turn_id>, decision }`.
5. Patch emits `PatchApplyBeginEvent` / `PatchApplyEndEvent` and may emit `TurnDiffEvent { unified_diff }`.
6. Tool returns `FunctionCallOutput`.

Harbor mapping:

- For “patch correctness” tasks, the tuple (raw patch input + structured `changes` + `TurnDiffEvent.unified_diff`) enables deterministic diff-based verifiers.
- `changes` is a `HashMap<PathBuf, FileChange>`; importantly:
  - `FileChange::Add { content }` and `FileChange::Delete { content }` include full file bodies
  - `FileChange::Update { unified_diff, move_path? }` carries a per-file diff and optional rename target
  - `TurnDiffEvent.unified_diff` is the aggregated unified diff across the whole patch application

##### Unified exec / interactive sessions (process_id + terminal interactions)

- Group `TerminalInteractionEvent` + streamed output deltas by `process_id`.
- Distill into Harbor only if the interactive flow can be replaced by deterministic checks; otherwise the task becomes non-reproducible.

#### 8) Capture hook points (exactly where to tap in code)

To produce this spine universally (across CLI, VS Code, app-server), capture at the core chokepoints:

- **SQ**: `submission_loop` in `core/src/codex.rs` immediately after `rx_sub.recv()` and before dispatch.
- **EQ**:
  - `Session::send_event_raw(...)` and `Session::send_event_raw_flushed(...)` in `core/src/codex.rs`
  - plus direct publishes that bypass `send_event` (notably `ExecCommandOutputDeltaEvent` in `core/src/exec.rs`)
- **Compaction boundary**: `core/src/compact.rs` and `core/src/compact_remote.rs` before history replacement.
- **Delegate bridges**: `core/src/codex_delegate.rs` when proxying approvals/user input across threads.
- **Plan updates**: `core/src/tools/handlers/plan.rs` emits `EventMsg::PlanUpdate(UpdatePlanArgs)`.
- **Diffs**: `core/src/turn_diff_tracker.rs` computes unified diffs; capture `EventMsg::TurnDiff(TurnDiffEvent { unified_diff })`.
- **Collab events**: `core/src/tools/handlers/collab.rs`.

#### 9) Fidelity notes (model-facing vs verifier-facing truth)

Codex emits both “model-facing” and “operational” representations. For distillation and debugging:

- Prefer `ExecCommandEnd.stdout/stderr/exit_code` for verifier-facing truth.
- Preserve `formatted_output` and tool outputs for “what the model saw,” but do not treat them as an authoritative verifier signal.
- Be aware of caps/truncation:
  - exec output is capped to avoid runaway output
  - previews may be truncated for telemetry/logging
  - turn truncation policy affects model-visible context (store the policy)

#### 10) Field-by-field capture inventory (reference: what “faithful” means)

This is the minimum set of fields/events we need in the spine so that later stages (signals + Harbor distillation) are deterministic and do not rely on heuristics.

##### Session-level (thread-scoped) fields

Source of truth: `RolloutItem::SessionMeta(SessionMetaLine)` (first rollout line).

- **Identity**
  - `thread_id`: `SessionMeta.id`
  - `forked_from_id`: `SessionMeta.forked_from_id` (thread lineage)
  - `session_source`: `SessionMeta.source` (`Cli`, `VSCode`, `Exec`, `Mcp`, `SubAgent(...)`, `Unknown`)
- **Timing / origin**
  - `session_start_timestamp`: `SessionMeta.timestamp`
  - `originator`: `SessionMeta.originator`
  - `cli_version`: `SessionMeta.cli_version`
- **Model baseline**
  - `model_provider`: `SessionMeta.model_provider`
  - `base_instructions`: `SessionMeta.base_instructions`
- **Workspace anchor**
  - `session_cwd`: `SessionMeta.cwd`
- **Git**
  - `git.commit_hash`, `git.branch`, `git.repository_url` (optional, from `SessionMetaLine.git`)

##### Turn-level fields (turn-scoped, policy-critical)

Source of truth today is split between:

- `TurnContextItem` persisted to rollouts (`RolloutItem::TurnContext(TurnContextItem)`), and
- the in-memory `TurnContext` in `core/src/codex.rs` (contains additional policy/tool fields not persisted).

For the closed loop, store a normalized `turn_context` record keyed by `turn_id` containing at least:

- **Join keys**
  - `turn_id`: `Submission.id` (and `Event.id`)
- **Policies**
  - `approval_policy`: `TurnContextItem.approval_policy`
  - `sandbox_policy`: `TurnContextItem.sandbox_policy`
  - plus in-memory policy fields that materially affect behavior (e.g., `windows_sandbox_level`, `shell_environment_policy`) when present
- **Model selection**
  - `model`: `TurnContextItem.model`
  - `effort`, `summary`: `TurnContextItem.effort`, `TurnContextItem.summary`
  - `personality`, `collaboration_mode`: `TurnContextItem.personality`, `TurnContextItem.collaboration_mode`
- **Prompt modifiers**
  - `user_instructions`, `developer_instructions`: `TurnContextItem.*`
  - `final_output_json_schema`: `TurnContextItem.final_output_json_schema`
- **Truncation**
  - `truncation_policy`: `TurnContextItem.truncation_policy`
- **Tools context (in-memory, must be captured for faithful replay)**
  - available tools + settings (`tools_config`)
  - `dynamic_tools`
  - whether ghost snapshots are enabled (affects whether undo/replay artifacts exist)

##### Tool-call fields (call-scoped, needs `call_id`)

Tool calls appear in two complementary places:

- **Model output items** (rollout/history)
  - `ResponseItem::FunctionCall { name, arguments, call_id }`
  - `ResponseItem::FunctionCallOutput { call_id, output: { content, content_items?, success? } }`
  - `ResponseItem::CustomToolCall` / `CustomToolCallOutput`
  - `ResponseItem::LocalShellCall` (client-specific representation)
  - MCP tools are often encoded as `FunctionCall` with a fully-qualified name; core resolves them via `parse_mcp_tool_name`.
- **Operational events** (EQ; must be captured)
  - `EventMsg::McpToolCallBegin/End`
  - `EventMsg::WebSearchBegin/End`
  - `EventMsg::ViewImageToolCall`

Capture (at minimum):

- `call_id` (join key)
- `tool_name` / function name
- raw `arguments` string
- tool outputs (both model-facing output items and operational end events when available)

##### Exec fields (command-scoped, verifier-ground-truth)

Capture:

- `ExecCommandBeginEvent` (command vector, cwd, parsed_cmd, source, optional `process_id`)
- `ExecCommandOutputDeltaEvent` (raw bytes; base64 on the wire; emitted directly from exec)
- `TerminalInteractionEvent` (stdin sent to interactive sessions)
- `ExecCommandEndEvent` (stdout, stderr, aggregated_output, exit_code, duration, formatted_output)

For Harbor verifiers: treat `ExecCommandEnd.stdout/stderr/exit_code` as authoritative truth; treat `formatted_output` as “what the model saw.”

##### Approval fields (user intent as ops)

Approvals have two sides:

- **Request** (EQ):
  - `ExecApprovalRequestEvent` (includes `call_id`, `turn_id`, command, cwd, reason, proposed_execpolicy_amendment)
  - `ApplyPatchApprovalRequestEvent` (includes `call_id`, `turn_id`, changes, reason, grant_root)
- **Decision** (SQ):
  - `Op::ExecApproval { id, decision: ReviewDecision }`
  - `Op::PatchApproval { id, decision: ReviewDecision }`

Also capture approval caching semantics:

- `ReviewDecision::ApprovedForSession` exists and can suppress future prompts; it is implemented via cached approval logic (`with_cached_approval(...)` in `core/src/tools/sandboxing.rs`).
- For training/eval correctness, it matters whether a run was repeatedly prompted vs implicitly approved from cache.

##### Patch/diff fields (file-scoped, enables deterministic diff verifiers)

Capture:

- raw `apply_patch` input (the model-supplied patch string)
- `PatchApplyBeginEvent` / `PatchApplyEndEvent` (includes structured `changes` and whether it was auto-approved)
- `TurnDiffEvent.unified_diff` (aggregate unified diff computed across the patch)

##### Errors, warnings, plan updates, and stream health (signal-critical)

These are explicitly called out in the system above as mechanical indicators and must be present in the spine (even though the rollout filter currently drops many of them):

- `EventMsg::Warning(WarningEvent { message })`
- `EventMsg::Error(ErrorEvent { message })`
- `EventMsg::StreamError(StreamErrorEvent { message })`
- `EventMsg::PlanUpdate(UpdatePlanArgs)`

##### Compaction / rollback / undo fields (history integrity)

Capture:

- `RolloutItem::Compacted(CompactedItem)` plus the trace-level `compaction_boundary` (pre/post snapshots)
- `EventMsg::ThreadRolledBack(ThreadRolledBackEvent { num_turns })`
- `EventMsg::UndoStarted` / `EventMsg::UndoCompleted`

---

### Stage 2 — Shipping: Codex → central raw lake (async, resumable, idempotent)

Codex should ship trace spine segments in immutable chunks without affecting UX latency:

- **Segment**: bounded JSONL chunk with `min_seq`, `max_seq`, `sha256`, `thread_id`, and a `segment_id`.
- **Uploader**: background task with retries and dedupe-by-hash (idempotent); never blocks the interactive loop.
- **Artifact bundling**: ship or separately store referenced artifacts needed for reproduction:
  - shell snapshot (or at minimum path + hash + capture time)
  - git diff-to-remote patch
  - any ghost snapshot material required for reproduction (if used)

Existing telemetry surfaces help but do not replace spine shipping:

- `codex-otel`: aggregate traces/logs/metrics
- `codex-rs/feedback`: Sentry uploads of logs (+ optional rollout attachment)

Neither guarantees full SQ+EQ fidelity.

---

### Stage 3 — Central ingestion: raw lake → canonical spine + indexes

On the central server:

- Store raw segments immutably and reconstruct each session by sorting records by `(thread_id, seq)`.
- Build a canonical, queryable representation (e.g., BigQuery tables) that normalizes:
  - sessions (session meta + git)
  - turns (turn context keyed by turn_id)
  - tool calls (call_id keyed)
  - execs (begin/end + deltas + process_id)
  - patches + diffs
  - approvals (request events + approval ops; include caching semantics)
  - compaction boundaries
  - provenance / lineage edges (thread spawn + delegate bridges)
  - warnings/errors/stream errors
  - plan updates

This canonical spine is the substrate for the “daily batch pipeline” described above (BigQuery filtering → batch inference → Slack reporting).

---

### Stage 4 — Analysis: facets and experience signals grounded in spine evidence

With SQ+EQ captured, the “mechanical” signals described above become direct and high-trust:

- **High-trust facets**:
  - `session_source`: `SessionMeta.source`
  - sub-agent lineage: `SubAgentSource::ThreadSpawn.parent_thread_id`, `depth`
  - model + reasoning settings: from turn context
  - approval/sandbox policy: from turn context
  - tool availability + usage: from turn context + tool call counts
  - compaction count: compaction boundary records + `CompactedItem` markers

- **Mechanical friction signals** (examples):
  - denials/aborts: `*ApprovalRequest` (EQ) + `Op::*Approval` (SQ)
  - failures: `ExecCommandEnd.exit_code != 0`, patch failures, tool outputs with `success=false`
  - stream instability: `EventMsg::StreamError`
  - warnings/degradations: `EventMsg::Warning`
  - plan instability: `EventMsg::PlanUpdate` churn + `TurnDiff` oscillation
  - backtracking: `Undo*`, `ThreadRolledBack`

Every derived signal should carry evidence pointers back to the spine:

- `thread_id`, `turn_id`, `call_id`, `seq_range` (+ artifact hashes when relevant)

These evidence pointers are what make “tickets and PRs linked to representative trace spines and Harbor datasets” implementable.

---

### Stage 5 — Distillation: trace segment → Harbor task (deterministic synthesis)

Given a canonical spine and a chosen segment window, distillation produces a runnable Harbor task directory.

#### Step 1: choose the segment window

Boundaries:

- **Start**: an intent-bearing input (usually `Op::UserTurn` / `Op::UserInput`)
- **End**: a completion signal (successful verification command, stable patch end + done, or a clearly labeled failure/abandonment such as `TurnAborted`, repeated denials, fatal errors)

Prefer minimal windows (single turn) unless multi-turn context is required for coherence.

#### Step 2: derive `instruction.md` (user intent, not scaffolding)

- Prefer `Op::UserTurn.items` / `Op::UserInput.items` as the source of the user’s words (including images).
- If you are distilling from a canonicalized spine that no longer has the raw `Op` payloads handy, fall back to the corresponding model-visible `ResponseItem::Message(role="user")` for the same `turn_id`.
- Do not include session prefix scaffolding (e.g., `<environment_context>`, `<turn_aborted>`). These are model-visible scaffolds, not user intent (`core/src/session_prefix.rs`).
- Include constraints only when they change solvability:
  - approval policy + sandbox policy
  - repo root inside container and any “no network” constraints implied by sandbox policy

#### Step 3: derive `environment/` (starting state reproduction)

Codex provides three main levers:

1. **Repo identity** via `SessionMetaLine.git` (`repository_url`, `commit_hash`, `branch`)
2. **Working tree delta** via `git_diff_to_remote(...)` (`core/src/git_info.rs`)
3. **Ghost snapshot** via `ResponseItem::GhostSnapshot` (useful for local undo; for Harbor reproduction only if we also ship snapshot content)

Recommended Harbor environment synthesis:

- Preferred: clone + checkout + apply patch/diff
- Fallback: minimal synthetic repo using captured file bodies where possible (best when changes are additive or preimages are fully captured)
  - `FileChange::Add { content }` and `FileChange::Delete { content }` include full file bodies and can be replayed in a synthetic repo
  - `FileChange::Update { unified_diff, ... }` still requires a base file preimage, so this fallback works best when the segment also captured the relevant base content (or when the update is small and the preimage can be reconstructed)

Use successful observed commands as dependency hints (e.g., `npm install`, `uv sync`, `cargo test`), but avoid baking in machine-specific shell snapshot state.

#### Step 4: derive `tests/test.sh` (verifier)

Preferred verifier sources:

1. reuse an observed successful verification command (`ExecCommandEnd.exit_code == 0`)
2. patch/diff structural verification (expected diff, file content assertions)
3. stdout matching only when inherently required

Ensure Harbor verifier contract is met (`/logs/verifier/reward.txt` or `/logs/verifier/reward.json`).

#### Step 5: derive `task.toml` (provenance + configuration)

Record provenance under `metadata.*`:

- `source_thread_id`, `source_turn_ids`, `source_session_source`
- `source_git.*`
- `source_policies.*` (approval + sandbox)
- signal/cluster ids from analysis
- tool usage summary

Set timeouts/resources based on observed execution envelopes.

---

### Stage 6 — Harbor execution artifacts: mapping back to Codex evidence

When Harbor runs the distilled task:

- verifier logs and reward files become quantitative success signals
- trajectories (ATIF export) and results must link back to the originating spine evidence pointers (`thread_id`, `turn_id` range, `seq_range`)

This is what makes regressions debuggable end-to-end.

---

### Stage 7 — Closing the loop: from signals to work artifacts to measured improvement

Because every signal and every Harbor task carries spine evidence pointers:

- tickets/PRs can embed exact trace citations and the Harbor dataset name used for validation
- re-evaluation can be automated (run the dataset after the fix; compare reward distributions)

---

### Required Codex-side gaps to close (to make this chapter true in production)

To fully satisfy the system invariants above, Codex must add/ensure:

- **SQ capture**: persist every `Submission` op to the trace spine
- **EQ capture**: persist every `Event` (including direct emitters like exec deltas)
- **Compaction write-ahead**: persist pre/post history snapshots as a `compaction_boundary` record
- **Turn context joinability**: include `turn_id` with turn context snapshots
- **Delegate bridge records**: make proxied approvals/user input between parent and sub-agent joinable without heuristics
- **Artifact references/hashes**: shell snapshot + git diff-to-remote patch (and any required ghost snapshot material) must be captured and shippable

---

### Summary (Codex → closed loop → Harbor)

Codex can support the closed-loop system above by treating SQ + EQ as first-class, append-only trace history (with compaction boundaries and provenance), shipping ordered segments to a raw lake, and enabling deterministic Harbor distillation from segment windows using git metadata/diffs plus exec/patch end events as verifier-facing ground truth.
