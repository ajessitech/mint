## Rollout+ → Harbor Spec: Plan (capture, distill, package)

### Purpose
Design (plan-only) the **slice** of the larger Codex closed-loop system that:

- **Runs a Codex “rollout+”** (eval-grade capture; more than today’s rollout JSONL).
- **Captures/mines** that bundle into a **Harbor spec** (task, plus optional dataset/registry entry).

Everything else (central lake, clustering, dashboards, automatic ticketing, etc.) is treated as downstream. This slice must be **deterministic, provenance-rich, and future-proof**.

---

### Scope (and what’s explicitly out of scope for now)
- **In scope**:
  - Define the **rollout+ bundle contract** (what must be captured).
  - Define **deterministic distillation** of a segment window → Harbor task directory.
  - Define packaging into a runnable “Harbor spec” (local dataset immediately; registry JSON once versioned).
- **Out of scope (for this doc)**:
  - Central ingestion/lake/BigQuery schema.
  - Embedding/clustering/signal discovery.
  - Automated work artifact generation (tickets/PRs).
  - Real-time interventions.

---

### Key definitions

#### “Rollout+”
**Rollout+ = an eval-grade session capture bundle** whose goal is:

> From this bundle alone, a distiller can deterministically produce a runnable Harbor task (instruction + environment + verifier + metadata) **without guesswork**.

Rollout+ is **not** just the existing rollout JSONL. It must include:

- **SQ**: every inbound `Submission { id, op }` (approvals/denials/interrupts/compactions/etc).
- **EQ**: every outbound `Event { id, msg }`, including direct emitters (notably exec output deltas).
- **Rollout JSONL**: model-visible items; useful for replay/resume but insufficient alone.
- **Repro artifacts**: git identity + diff-to-remote, shell snapshot reference, etc.
- **Compaction boundaries**: pre/post history snapshots so resets are reconstructible.

#### “Harbor spec”
The minimum Harbor “spec” we should emit is a **Harbor task directory**, optionally grouped into a **dataset** (local folder or registry entry):

- **Task directory (required)**:
  - `instruction.md`
  - `task.toml`
  - `environment/` (usually `Dockerfile`)
  - `tests/test.sh`
  - optional `solution/solve.sh` (oracle sanity-check)
- **Dataset (optional but recommended quickly)**:
  - **Local dataset**: directory containing multiple task directories
  - **Registry dataset**: JSON entry pointing to git URL + pinned commit + task paths

---

### System shape for this slice

#### Inputs
- A **rollout+ bundle** produced by running Codex in a “record everything we need” mode.

#### Outputs
- One or more **Harbor tasks** distilled from selected segment windows.
- A **dataset definition** to run the tasks repeatedly (local first, registry later).

#### Non-negotiable invariants (to support the larger system later)
- **Determinism**: distillation should not rely on heuristics that silently drift.
- **Joinability**: stable IDs (`thread_id`, `turn_id`, `call_id`, `process_id`) + writer-side monotonic `seq`.
- **Provenance**: every task can point back to exact evidence spans in the trace spine.
- **Reproducibility**: environment synthesis is explicit and auditably derived.
- **Compaction awareness**: resets are explicit boundaries; never pretend history was continuous.

---

### Rollout+ bundle contract (artifact layout)
Goal: a single folder (or tarball) that can be moved to another machine and still be mineable.

#### Proposed bundle layout (v0)
- `spine/`
  - `segment-000.jsonl`, `segment-001.jsonl`, …
  - Contains the **trace spine**: SQ + EQ + artifact refs (+ compaction boundaries)
- `rollout/`
  - existing Codex rollout JSONL (model-visible stream)
- `artifacts/`
  - git identity (`repo_url`, `commit_hash`, `branch`) and **diff-to-remote patch**
  - shell snapshot reference (path + hash; ideally content-addressed)
  - other reproduction inputs required for determinism

---

### Trace spine schema (v1) and join keys

#### Per-record envelope (v1)
Each JSONL line is one record with:
- `schema_version`: e.g. `codex_trace_spine_v1`
- `thread_id`
- `seq`: **monotonic** writer-side order key (required; don’t rely on timestamps)
- `timestamp`
- `type`: `submission` | `event` | `turn_context` | `artifact_ref` | `compaction_boundary` | `bridge` | …
- `payload`: **raw** protocol payload (store it as-is for future-proofing)

#### Join keys (first-class)
- `thread_id`: session identity
- `turn_id`: universal join axis (SQ `Submission.id` and EQ `Event.id`)
- `call_id`: tool call pairing + exec/patch begin/end pairing
- `process_id`: interactive exec transcript grouping (when present)

#### Approval join nuance (critical)
Approval ops must join using `Op::*Approval.id` (the “approved turn id”), not the approval submission envelope id (which may differ).

---

### Phase plan

### Phase A — Define the rollout+ contract (before automation)
Goal: write down exactly what is required so mining never becomes guesswork.

- **Capture sources (union)**:
  - SQ: persist every `Submission { id, op }` exactly as received
  - EQ: persist every `Event { id, msg }`, including direct publishers (exec deltas)
  - Rollouts: keep as parallel artifact; do not rely on it for eval-grade fidelity
  - Repro artifacts: git identity + diff-to-remote; shell snapshot ref; other hashes
  - Compaction boundaries: pre/post history snapshots + boundary record

Exit criteria:
- Given a rollout+ bundle, we can assert “everything required to deterministically create a Harbor task is present.”

---

### Phase B — “Run a rollout+” so it yields mineable bundles
Goal: make it easy to repeatedly generate high-quality raw material.

- **Recording mode**:
  - local-first, append-only, crash-safe
  - segments are bounded for resumable shipping later (even if shipping is out of scope)
- **Make runs intentionally mineable**:
  - prefer single-intent turns (clear definition of done)
  - ensure a completion signal exists:
    - success: observed verification command with `exit_code == 0`
    - success: patch apply end + stable diff state
    - failure/abandonment markers for later negative tasks
- **Privacy posture** (planning decision):
  - avoid irreversible redaction at capture time if it breaks distillation
  - keep raw locally; govern derived tasks separately (conservative defaults)

Exit criteria:
- “Run Codex → get a rollout+ bundle that a separate distiller can consume deterministically.”

---

### Phase C — Distill one segment window into one Harbor task (the mining core)
Goal: convert a bounded span of a session into a **minimal runnable Harbor task**.

#### C1. Segment window selection
- **Start**: intent-bearing input (`Op::UserTurn` / `Op::UserInput`)
- **End** (one of):
  - successful observed verification command
  - stable patch apply end + “done”
  - clearly labeled failure/abandonment (`TurnAborted`, repeated denials, fatal errors)

v0: manual selection by `thread_id` + `turn_id` start/end (keeps mining honest).  
v1: rule-based selector (later upgraded to signal-driven selection).

#### C2. Generate `instruction.md` (user intent only)
- prefer `Op::UserTurn.items` as source of user words (including images)
- exclude prefix scaffolding (`<environment_context>`, etc.)
- include constraints only if they affect solvability (sandbox/no-network/approval constraints)

#### C3. Generate `environment/` (starting state reproduction)
This is the hardest part; plan supports multiple strategies:

- **Strategy 1 (preferred)**: clone + checkout + apply diff-to-remote
  - best fidelity; requires repo accessibility (credentials for private repos)
- **Strategy 2**: include repo snapshot artifact (tarball/git-bundle)
  - deterministic + works for private repos; watch repo size/governance
- **Strategy 3**: synthetic minimal repo from captured file bodies
  - smallest footprint; only safe when preimages/deps are fully captured

v0 recommendation:
- Strategy 1 for open/public repos
- Strategy 2 for private/internal + “no network” environments
- Strategy 3 as later optimization

#### C4. Generate `tests/test.sh` (verifier)
Verifier preference order:
1. reuse an observed successful verification command (`exit_code == 0`)
2. diff/content structural verification (good for patch tasks)
3. stdout matching only when inherently required

Verifier must emit:
- `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`

v0 recommendation:
- mine only tasks where (1) exists to maximize determinism and speed-to-value

#### C5. Generate `task.toml` (config + provenance)
- **Config**: timeouts/resources based on observed envelopes
- **Metadata** (critical for later loops):
  - `source_thread_id`, `source_turn_ids`, `seq_range`, `call_ids`
  - git provenance (`repo_url`, `commit_hash`, `diff_hash`)
  - policy provenance (approval + sandbox)
  - (later) `cluster_id`, `signal_type`, facet predicates

#### C6. Optional `solution/solve.sh` (oracle sanity-check)
- include when simple, to gate solvability (`harbor run -p <task> -a oracle`)

Exit criteria:
- task environment builds
- verifier is deterministic
- oracle passes when provided
- a real agent can run the task and produce meaningful reward

---

### Phase D — Package mined tasks into a runnable “Harbor spec”
Goal: make output a repeatable eval lever, not just a one-off directory.

- **D1. Local dataset first**
  - output multiple tasks into one dataset folder; run with `harbor run -p <dataset>`
- **D2. Registry dataset next**
  - put tasks in a git repo; generate registry JSON pointing to pinned commits
  - use custom registry file/URL for private, fast-moving datasets
- **D3. Optional job specs**
  - generate `job.yaml`/`job.json` templates for repeated runs across agents/models

Exit criteria:
- a single command runs the mined set repeatedly (local dataset or registry dataset)

---

### Edge cases and design stances (decide now; implement later)

#### Approvals/denials vs Harbor headless runs
- always capture approvals/denials in rollout+ (essential evidence)
- v0 mining focuses on tasks that do not require interactive approvals
- v1 introduces policy-sensitive tasks by verifying expected agent behavior via trajectory/action logs

#### Compaction/context resets
- record compaction boundary snapshots in rollout+
- avoid mining tasks whose solvability depends on pre-compaction context unless included unambiguously

#### Interactive exec (process_id)
- v0: exclude from mining unless it can be replaced with deterministic checks

#### Model-visible vs verifier-truth mismatch
- prefer operational truth (`ExecCommandEnd.stdout/stderr/exit_code`) over model-facing tool output strings for verifiers

#### Repo accessibility and “no network”
- decide up front whether network is allowed at build time
- use repo snapshot artifacts when needed to keep tasks runnable and deterministic

---

### Milestones (sequence that minimizes risk)

#### Milestone 1: “One mined task runs”
- run one rollout+ designed to end with a clean verifier command
- manually pick the segment window
- distill to one Harbor task and run it successfully

#### Milestone 2: “Ten tasks, one dataset”
- ~10 rollouts+ across 2–3 archetypes (patch-only, test-command, build-command)
- package into a local dataset; confirm repeatability

#### Milestone 3: “Versioned Harbor spec”
- tasks in git + registry dataset JSON pinned to commits
- run via `harbor run -d dataset@version`

#### Milestone 4: “Mining automation v1”
- rule-based segment selection
- automatic environment strategy choice (clone vs snapshot)
- quality gates (build + determinism + oracle if present)

