Motivation
Why we built Harbor

Harbor is a framework for evaluating and optimizing agents and models in container environments.

When we released Terminal-Bench in May, we were surprised to see it used in unexpected ways like building custom evals, optimizing prompts, running RL, generating SFT traces, and CI/CD agent testing.

We also learned that defining and managing containerized tasks at scale is hard. We built Harbor to make it easy.

Harbor provides:

Simple, modular interfaces for environments, agents, and tasks
All popular CLI agents pre-integrated
A registry of popular benchmarks and datasets
Integrations with cloud sandbox providers like Daytona, Modal, and E2B for horizontal scaling
Integrations with frameworks like SkyRL and GEPA for optimizing agents


Getting Started
Installing the package and running your first eval

Harbor is a framework for evals, post-training, and prompt optimization using agentic environments.

Installation
uv
pip

uv tool install harbor
Getting started
Run the following command to see a list of all available commands:


harbor --help
Running an eval
The primary command is harbor run, which is used to run evals or generate rollouts.


harbor run --help
To view registered datasets, run


harbor datasets list
Running a registered dataset
To evaluate an agent and model one of these datasets, you can use the following command:


harbor run -d "<dataset@version>" -m "<model>" -a "<agent>"
Harbor will automatically download registered datasets.

Running a local dataset
Local datasets (directories of tasks) can also be run using


harbor run -p "<path/to/dataset>" -m "<model>" -a "<agent>"
Running a cloud sandbox
To run using a cloud sandbox provider like Daytona, you can use the following command:


harbor run -d "<dataset@version>" -m "<model>" -a "<agent>" --env "daytona" -n 32
If you run a cloud sandbox using an API model, trials become I/O bounded rather than compute bounded, which means you can typically parallelize far above your CPU count (the example command above runs 32 trials concurrently).

Sandboxed agent evaluations are often slow, because they can require many turns to complete and each command requires time to execute. Horizontal scaling becomes the only viable way to accelerate experimentation, so we recommend using a cloud sandbox provider like Daytona.


Core Concepts
Core concepts and terminology in Harbor

Harbor has the following core concepts:

Task
A task is a single instruction, container environment, and test script. Tasks are used to evaluate agents and models. A task is implemented as a directory of files in the Harbor task format.

Dataset
A dataset is a collection of tasks. Datasets are used to evaluate agents and models. Usually, a dataset corresponds to a benchmark (e.g. Terminal-Bench, SWE-Bench Verified, etc.). Datasets can optionally be distributed via the Harbor registry.

Agent
An agent is a program that completes tasks. Agents are defined by implementing the BaseAgent or BaseInstalledAgent interfaces.

Container environment
Environments in Harbor are containers, typically defined as Docker images using a Dockerfile. The BaseEnvironment interface provides a unified interface for interacting with environments. Many cloud container runtimes are already supported out of the box, including Daytona, Modal, and E2B. Other container runtimes can be supported by implementing the BaseEnvironment interface.

Trial
A trial is an agent's attempt at completing a task. Trials can be configured using the TrialConfig class.

Essentially, a trial is a rollout that produces a reward.

Job
A job is a collection of trials. Jobs are used to evaluate agents and models. A job can consist of multiple datasets, agents, tasks, and models. Jobs can be configured using the JobConfig class.

Once you define your job.yaml or job.json file, you can run it using the following command:


harbor run -c "<path/to/job.yaml>"
Alternatively, you can create an adhoc job by configuring the harbor run flags.

Under the hood, a job generates a bunch of TrialConfig objects and runs them in parallel.




Agents
Using popular agents and integrating your own

How to evaluate on existing agents and integrate your own. This is particularly useful for benchmarking your agent, optimizing its prompts, using it as a scaffold for RL, or using it to generate SFT datasets.

Existing agents
Harbor comes with most popular agents pre-integrated. You can run the following command and reference the --agent flag to see a list of all available agents:


harbor run --help
Right now, Harbor includes Terminus-2, Claude Code, Codex CLI, Gemini CLI, OpenHands, Mini-SWE-Agent, and more.

Integrating your own agent
Harbor supports integrating your own agent without having to modify the Harbor source code.

There are two types of agents:

External agents which interface with the environment through the BaseEnvironment interface, typically by executing bash commands via the exec method.
Installed agents which are agents that are installed directly into the container environment and are executed in headless mode. This is how most agents are integrated and comes with the advantage of bringing custom tools.
External agents
To build an external agent, you need to implement the BaseAgent interface which involved defining the following methods:

my_external_agent.py

from harbor.agents.base import BaseAgent
class MyExternalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        """The name of the agent."""
        pass
    def version(self) -> str | None:
        """The version of the agent."""
        pass
    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.
        """
        pass
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Runs the agent in the environment. Be sure to populate the context with the
        results of the agent execution. Ideally, populate the context as the agent
        executes in case of a timeout or other error.
        Args:
            instruction: The task instruction.
            environment: The environment in which to complete the task.
            context: The context to populate with the results of the agent execution.
        """
        pass
Installed agents
To build an installed agent, you need to implement the BaseInstalledAgent interface which involved defining the following methods:

my_installed_agent.py

from harbor.agents.installed.base import BaseInstalledAgent
class ExecInput(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout_sec: int | None = None
class MyInstalledAgent(BaseInstalledAgent):
    @property
    def _install_agent_template_path(self) -> Path:
        """
        Path to the jinja template script for installing the agent in the container.
        """
        pass
    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """
        Create the commands to run the agent in the container. Usually this is a single
        command that passes the instruction to the agent and executes it in headless
        mode.
        """
        pass
    def populate_context_post_run(self, context: AgentContext) -> None:
        """
        Populate the context with the results of the agent execution. Assumes the run()
        method has already been called. Typically involves parsing a trajectory file.
        """
        pass
Running a custom agent
To run a custom agent, you can use the following command:


harbor run -d "<dataset@version>" --agent-import-path path.to.agent:SomeAgent


Terminus-2
Harbor's high-performance reference agent implementation

Overview
Terminus-2 is Harbor's reference agent implementation, designed as a research-preview agent for evaluating language models' capabilities in terminal environments. It operates entirely autonomously within sandboxed environments and serves as a high-performance neutral test bed for understanding language model agent capabilities.

Key Features
Mono-tool Design
Terminus-2 uses a unique single-tool approach - an interactive tmux session - allowing it to:

Send keystrokes and navigate environments flexibly
Scroll through output and use arrow keys to navigate menus
Launch additional shells within the environment
Interact with any terminal-based application naturally
This design philosophy enables the agent to work with virtually any command-line interface without requiring specialized tools for each interaction pattern.

Independent Execution
The agent's logic runs in a separate Python process from the Docker container, enabling:

Remote connection to arbitrary computer environments
Dockerized execution environments for safety and isolation
Flexible deployment across different infrastructure setups
Clean separation between agent logic and task environment
Autonomy-First Approach
Terminus-2 is designed to operate without human intervention:

Will never ask for user input during task execution
Independently attempts to complete tasks end-to-end
Currently recommended only for sandboxed environments due to full autonomy
Makes decisions and recovers from errors without guidance
Using Terminus-2 with Harbor
Basic Usage
Run Terminus-2 on a task using the --agent terminus-2 flag:


harbor run \
  --agent terminus-2 \
  --model openai/gpt-5 \
  --path examples/tasks/ \
  --task-name hello-world
Configuration Options
Terminus-2 supports various configuration options through the agent config:


from harbor.models.trial.config import AgentConfig
from harbor.models.agent_name import AgentName
agent_config = AgentConfig(
    name=AgentName.TERMINUS_2,
    model_name="openai/gpt-5",
    kwargs={
        # Parser configuration
        "parser_name": "json",  # "json" or "xml" (default: "json")
        # API configuration
        "api_base": "https://your-vllm-server.com",  # Custom API endpoint
        "temperature": 0.7,  # Sampling temperature (default: 0.7)
        # Episode/turn limits
        "max_turns": 100,  # Maximum number of episodes (default: 1000000)
        # Summarization configuration
        "enable_summarize": True,  # Enable context summarization (default: True)
        "proactive_summarization_threshold": 8000,  # Free tokens threshold for summarization (default: 8000)
        # RL training configuration (default: False)
        # If enabled, token ids and logprobs are collected in result and persisted in trajectories
        "collect_rollout_details": False,
        # Advanced model configuration
        "reasoning_effort": "medium",  # "none", "minimal", "low", "medium", "high", or "default" (default: None)
        "max_thinking_tokens": 2048,  # For Anthropic extended thinking mode (minimum: 1024, default: None)
        # Optional: Register custom model info with LiteLLM
        # LiteLLM doesn't recognize uncommon models like custom models. For metrics
        # tracking and context summarization to work properly, provide model_info following
        # https://docs.litellm.ai/docs/completion/token_usage#9-register_model
        "model_info": {
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
        },  
        # Session tracking (included in the LLM request body unless LLM provider doesn't support)
        "session_id": "custom-session-id",  # Custom session ID (default: auto-generated UUID)
    }
)
Conversation History Management
Terminus-2 implements intelligent conversation history management to handle long-running tasks efficiently while staying within context window limits.

Standard Summarization Process
Both proactive and passive summarization use a 3-step subagent process to generate high-quality summaries:


┌─────────────────────────────────────────────────────────────────┐
│                   Standard Summarization Flow                    │
└─────────────────────────────────────────────────────────────────┘
  Previous History
        │
        ▼
  ┌─────────────────────┐
  │ 1. Summary Subagent │
  │   Input: Previous   │
  │   Output: Summary   │
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 2. Question Subagent│
  │   Input: Summary    │
  │   Output: Questions │
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │ 3. Answer Subagent  │
  │   Input: Previous + │
  │   Summary + Qs      │
  │   Output: Answers   │
  └─────────────────────┘
        │
        ▼
  ┌─────────────────────┐
  │   Main Agent        │
  │   Context:          │
  │   • System prompt   │
  │   • Task            │
  │   • Summary         │
  │   • Questions       │
  │   • Answers         │
  └─────────────────────┘
Step 1 - Summary Subagent: Receives the full previous conversation history and generates an initial summary.

Step 2 - Question Subagent: Receives only the summary (not the full history) and generates clarifying questions about any missing critical information.

Step 3 - Answer Subagent: Receives the previous history, summary, and questions, then answers the questions to fill in the gaps.

The main agent then continues with a compressed context containing: system prompt, task description, summary, questions, and answers.

Proactive Summarization
When free tokens (max input tokens - current context length) drop below the proactive_summarization_threshold (default: 8000), Terminus-2:

Pauses execution
Runs the standard 3-step summarization process on the conversation history
Replaces the middle portion of the conversation history with the summary + Q&A
Keeps the system prompt and task description intact
Resumes execution with the compressed history
The threshold can be configured via proactive_summarization_threshold in agent config.

Passive Summarization
When a ContextLengthExceededError occurs, Terminus-2 uses a 3-way fallback strategy to recover and continue execution:


┌─────────────────────────────────────────────────────────────────┐
│              Passive Summarization Fallback Flow                 │
└─────────────────────────────────────────────────────────────────┘
              ContextLengthExceededError
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 1. Unwind to Free Tokens     │
            │    Remove recent messages    │
            │    from end until enough     │
            │    space (keeps first msg)   │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ 2. Standard Summarization    │
            │    (3-step subagent process) │
            └──────────────────────────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
              Success            Failure
                  │                 │
                  │                 ▼
                  │    ┌──────────────────────────┐
                  │    │ 3. Fallback Summary      │
                  │    │    Only: System prompt + │
                  │    │    Task + Current state  │
                  │    └──────────────────────────┘
                  │                 │
                  │        ┌────────┴────────┐
                  │        │                 │
                  │    Success            Failure
                  │        │                 │
                  │        │                 ▼
                  │        │    ┌──────────────────────┐
                  │        │    │ 4. Ultimate Fallback │
                  │        │    │    System prompt +   │
                  │        │    │    Task + State only │
                  │        │    │    (Continue without │
                  │        │    │     summarization)   │
                  │        │    └──────────────────────┘
                  │        │                 │
                  └────────┴─────────────────┘
                           │
                           ▼
                   Continue execution with
                   compressed/recovered context
Step 1 - Unwind: Remove recent messages from the end of the conversation (in pairs of user + assistant) until there are enough free tokens for summarization. Always keeps at least the first message.

Step 2 - Standard Summarization: Run the 3-step subagent process. If successful, replace the unwound messages with the summary + Q&A and continue execution.

Step 3 - Fallback: If standard summarization fails, attempt a simpler summary using only system prompt, task description, and current state. If successful, continue with this compressed context.

Step 4 - Ultimate Fallback: If fallback also fails, continue execution with only system prompt, task description, and current state (no summary).

This recovery mechanism allows Terminus-2 to continue executing even when context limits are exceeded. Enable with enable_summarize=True in agent config.

Reinforcement Learning Support
Terminus-2 is designed with RL training in mind and collects detailed rollout information for use in RL pipelines.

Rollout Details Collection
During execution, Terminus-2 can collect and export:

Token Information
Prompt Token IDs: List of token ID sequences, one per turn. Each sequence contains the full prompt including chat history.
Completion Token IDs: List of token ID sequences, one per turn. Each sequence contains the response tokens for that turn.
Logprobs: List of log probability sequences corresponding to each completion.
These are stored as a list of RolloutDetail objects in the agent result metadata:


# First RolloutDetail contains main agent conversation
rollout_detail = trial_result.agent_result.metadata["rollout_details"][0]
# Access turn-by-turn data
prompt_token_ids = rollout_detail["prompt_token_ids"]  # List[List[int]]
completion_token_ids = rollout_detail["completion_token_ids"]  # List[List[int]]
logprobs = rollout_detail["logprobs"]  # List[List[float]]
Rewards
Terminus-2 integrates with Harbor's verifier system to collect rewards:


# Access rewards from trial results
reward = trial_result.verifier_result.rewards.get("reward", 0)
Trajectory Format
Terminus-2 automatically generates trajectories in the Agent Trajectory Interchange Format (ATIF), Harbor's standardized trajectory format. This enables:

SFT dataset generation: Convert successful trajectories to supervised fine-tuning data
RL training: Use complete action sequences and rewards for policy optimization
Debugging: Inspect detailed step-by-step execution logs
Visualization: Replay agent actions in Harbor's trajectory viewer
See the Agent Trajectory Format documentation for details on the ATIF specification.

Trajectory Configuration
Terminus-2 supports a TrajectoryConfig that controls how trajectories are recorded and formatted. This is particularly important when generating SFT datasets or when context summarization occurs.

Configuration Options
raw_content (default: False)

Controls whether to save raw LLM responses or parsed structured data in the trajectory.

raw_content=False (default): Saves parsed, structured data with separate message and tool_calls fields. Best for trajectory analysis and debugging.
raw_content=True: Saves the exact raw LLM response in the message field without parsing. Essential for SFT data export where you need the exact model outputs.
linear_history (default: False)

Controls how trajectories are split when context summarization occurs. The key difference is whether you can recover the true LLM conversation history from the trajectory files.

During agent execution, the actual conversation history sent to the LLM looks like this:


Turn 1:
  {"user": "You are an AI assistant tasked with solving command-line tasks...
           Task Description: Create a file called hello.txt
           Current terminal state: ..."}
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
Turn 2:
  {"user": "You are an AI assistant tasked with solving command-line tasks...
           Task Description: Create a file called hello.txt
           Current terminal state: ..."}
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
  {"user": "New Terminal Output:\nroot@container:/app# ..."}
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
// Context summarization happens here
Turn 3:
  {"user": "You are an AI assistant tasked with solving command-line tasks...
           Task Description: Create a file called hello.txt
           Current terminal state: ..."}
  {"user": "You are picking up work from a previous AI agent on this task:
           **Original Task:** Create a file called hello.txt
           **Summary from Previous Agent:** ...
           **Current Terminal Screen:** ...
           Please begin by asking several questions..."}
  {"assistant": "1. What files have been created so far?\n2. ..."}
  {"user": "Here are the answers the other agent provided.\n\n[answers]\n\nContinue working on this task..."}
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
Notice how in Turn 3, the conversation history was reset and compressed - the system prompt is followed by the question prompt (which includes the task, summary, and terminal screen), then model questions, then the handoff prompt with answers, skipping all the intermediate conversation steps.

When linear_history=False (default):

All main agent steps are stored in a single trajectory.json file in a human-readable format, while summarization subagents are stored in separate files. However, you cannot recover the true conversation history from the main trajectory file - the handoff prompt appears to be a continuation of the previous conversation, but in reality, the LLM context was reset.

File structure:


trajectory.json                              # All main agent steps
trajectory.summarization-1-summary.json      # First summarization: summary subagent
trajectory.summarization-1-questions.json    # First summarization: questions subagent
trajectory.summarization-1-answers.json      # First summarization: answers subagent
If multiple summarizations occur, you'll see:


trajectory.json
trajectory.summarization-1-*.json            # First summarization
trajectory.summarization-2-*.json            # Second summarization
...
When linear_history=True:

The trajectory is split into separate files when summarization occurs. Each file represents a continuous, unambiguous linear history that was actually sent to the LLM.

trajectory.json:


[
  {"user": "You are an AI assistant tasked with solving command-line tasks...\nTask Description: Create a file called hello.txt\nCurrent terminal state: ..."},
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"},
  {"user": "New Terminal Output:\nroot@container:/app# ..."},
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
]
trajectory.cont-1.json:


[
  {"user": "You are an AI assistant tasked with solving command-line tasks...\nTask Description: Create a file called hello.txt\nCurrent terminal state: ..."},
  {"user": "You are picking up work from a previous AI agent on this task:\n**Original Task:** ...\n**Summary from Previous Agent:** ...\n**Current Terminal Screen:** ...\nPlease begin by asking several questions..."},
  {"assistant": "1. What files have been created so far?\n2. ..."},
  {"user": "Here are the answers the other agent provided.\n\n[answers]\n\nContinue working on this task..."},  // No ambiguity!
  {"assistant": "{'analysis': '...', 'plan': '...', 'commands': [...]}"}
]
File structure:


trajectory.json                              # Before first summarization
trajectory.cont-1.json                       # After first summarization
trajectory.cont-2.json                       # After second summarization (if any)
trajectory.summarization-1-*.json            # First summarization subagents
trajectory.summarization-2-*.json            # Second summarization subagents (if any)
Use cases:

linear_history=False: Simpler structure, easier to see the full agent execution in one file. Good for debugging and human analysis.
linear_history=True: Each file represents the exact LLM context. Essential for SFT training where you need unambiguous input/output sequences.
Example Configuration

from harbor.agents.terminus_2 import Terminus2
from harbor.models.agent.trajectory_config import TrajectoryConfig
# For SFT dataset generation
trajectory_config = TrajectoryConfig(
    raw_content=True,      # Preserve exact LLM responses
    linear_history=True    # Split on summarization for clean sequences
)
agent = Terminus2(
    logs_dir=Path("logs"),
    model_name="anthropic/claude-3-5-sonnet-20241022",
    trajectory_config=trajectory_config
)
Common configurations:

For debugging and analysis:


TrajectoryConfig(
    raw_content=False,     # Structured, parsed data
    linear_history=False   # Single file, easier to navigate
)
For SFT data export:


TrajectoryConfig(
    raw_content=True,      # Raw model outputs
    linear_history=True    # Clean input/output sequences
)
For RL training:


TrajectoryConfig(
    raw_content=False,     # Either works (token IDs are always exact), but structured helps debugging
    linear_history=False   # Full episode in one file
)
Related Documentation
Agents Overview - General agent integration guide
Agent Trajectory Format - ATIF specification and usage
RL Training - Using Terminus-2 for reinforcement learning
SFT Datasets - Generating supervised fine-tuning data

Agent Trajectory Format (ATIF)
Understanding and working with the Agent Trajectory Interchange Format

Overview
The Agent Trajectory Interchange Format (ATIF) is a standardized, JSON-based specification for logging the complete interaction history of autonomous LLM agents. ATIF unifies the data requirements of conversational logs, action sequences, and replayable data structures, ensuring collected data is immediately usable across debugging, visualization, Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL) pipelines.

For the complete specification, see the ATIF RFC.

Key Features
ATIF provides a comprehensive format that captures:

Complete interaction history: User messages, agent responses, tool executions, and environment feedback
Multi-turn conversations: Support for both single-turn tasks and extended conversational interactions
LLM metrics: Token usage, costs, logprobs, and other operational metrics
Tool calls and observations: Structured logging of agent actions and their results
Multi-agent systems: Support for subagent delegation and hierarchical architectures
Extensibility: Optional extra fields at all levels for custom metadata
Harbor Support
Harbor provides first-class support for ATIF through:

Pydantic models for type-safe trajectory construction and validation
Trajectory validator for validating trajectory files against the ATIF schema
Automatic trajectory generation by integrated agents
Supported Agents
The following agents in Harbor automatically generate ATIF-compliant trajectories:

Terminus-2 - Harbor's reference agent implementation
OpenHands - Converts OpenHands event logs to ATIF format
Mini-SWE-Agent - Software engineering agent with trajectory support
Gemini CLI - Google's Gemini agent interface
Claude Code - Anthropic's code agent
Codex - OpenAI's Codex agent with trajectory support
OpenHands Example
OpenHands is a great example of how Harbor converts agent-specific formats to ATIF. The OpenHands agent reads event files from the agent's execution and converts them to a standardized ATIF trajectory:


# From harbor/agents/installed/openhands.py
def populate_context_post_run(self, context: AgentContext) -> None:
    """Convert OpenHands events to ATIF trajectory format."""
    # Get the session directory
    session_dir = self._get_session_dir()
    events_dir = session_dir / "events"
    # Convert events to trajectory
    trajectory = self._convert_events_to_trajectory(events_dir)
    # Write trajectory.json file using Pydantic's to_json_dict method
    trajectory_path = self.logs_dir / "trajectory.json"
    with open(trajectory_path, "w") as f:
        json.dump(trajectory.to_json_dict(), f, indent=2)
    # Populate context from trajectory
    if trajectory.final_metrics:
        context.cost_usd = trajectory.final_metrics.total_cost_usd
        context.n_input_tokens = trajectory.final_metrics.total_prompt_tokens
        context.n_output_tokens = trajectory.final_metrics.total_completion_tokens
The conversion process:

Reads OpenHands event files
Maps events to ATIF steps (system/user/agent)
Converts accumulated metrics to per-step deltas
Creates a complete Trajectory object using Pydantic models
Exports to JSON format
Data Classes
Harbor provides Pydantic models for all ATIF schema components in harbor.models.trajectories:

Core Models
Trajectory - Root-level trajectory object


from harbor.models.trajectories import Trajectory, Agent, Step
trajectory = Trajectory(
    schema_version="ATIF-v1.4",
    session_id="session-123",
    agent=Agent(
        name="my-agent",
        version="1.0.0",
        model_name="claude-3-5-sonnet-20241022"
    ),
    steps=[
        # ... steps
    ]
)
Agent - Agent configuration


from harbor.models.trajectories import Agent
agent = Agent(
    name="openhands",
    version="0.9.0",
    model_name="gpt-4",
    extra={"agent_class": "CodeActAgent"}
)
Step - Individual interaction step


from harbor.models.trajectories import Step
# User step
user_step = Step(
    step_id=1,
    timestamp="2025-01-15T10:30:00Z",
    source="user",
    message="Create a file called hello.txt with 'Hello, world!' as the content."
)
# Agent step with tool calls
agent_step = Step(
    step_id=2,
    timestamp="2025-01-15T10:30:02Z",
    source="agent",
    model_name="claude-3-5-sonnet-20241022",
    message="I'll create the file for you.",
    reasoning_content="The user wants a simple text file. I'll use the file_write tool.",
    tool_calls=[
        ToolCall(
            tool_call_id="call_1",
            function_name="file_write",
            arguments={"path": "hello.txt", "content": "Hello, world!"}
        )
    ],
    observation=Observation(
        results=[
            ObservationResult(
                source_call_id="call_1",
                content="File created successfully"
            )
        ]
    ),
    metrics=Metrics(
        prompt_tokens=520,
        completion_tokens=80,
        cached_tokens=200,
        cost_usd=0.00045
    )
)
ToolCall - Tool/function invocation


from harbor.models.trajectories import ToolCall
tool_call = ToolCall(
    tool_call_id="call_price_1",
    function_name="financial_search",
    arguments={"ticker": "GOOGL", "metric": "price"}
)
Observation - Environment feedback


from harbor.models.trajectories import Observation, ObservationResult
observation = Observation(
    results=[
        ObservationResult(
            source_call_id="call_price_1",
            content="GOOGL is currently trading at $185.35"
        )
    ]
)
Metrics - LLM operational metrics


from harbor.models.trajectories import Metrics
metrics = Metrics(
    prompt_tokens=520,
    completion_tokens=80,
    cached_tokens=200,
    cost_usd=0.00045,
    logprobs=[-0.1, -0.05, -0.02],  # Optional
    completion_token_ids=[1722, 310, 5533]  # Optional
)
FinalMetrics - Trajectory-level aggregate metrics


from harbor.models.trajectories import FinalMetrics
final_metrics = FinalMetrics(
    total_prompt_tokens=1120,
    total_completion_tokens=124,
    total_cached_tokens=200,
    total_cost_usd=0.00078,
    total_steps=3
)
Export to JSON
All models provide a to_json_dict() method for clean JSON export:


from harbor.models.trajectories import Trajectory
import json
# Build trajectory using Pydantic models
trajectory = Trajectory(...)
# Export to JSON (excludes None values by default)
trajectory_dict = trajectory.to_json_dict()
# Write to file
with open("trajectory.json", "w") as f:
    json.dump(trajectory_dict, f, indent=2)
# Include None values if needed
trajectory_dict_full = trajectory.to_json_dict(exclude_none=False)
Validation
Harbor provides a trajectory validator for validating ATIF trajectory files:

Command Line

# Validate a trajectory file
python -m harbor.utils.trajectory_validator trajectory.json
Output:


✓ Trajectory is valid: trajectory.json
Or for invalid trajectories:


✗ Trajectory validation failed: trajectory.json
Found 2 error(s):
  - trajectory.steps.0.step_id: expected 1 (sequential from 1), got 0
  - trajectory.agent.name: required field is missing
Programmatic Usage

from harbor.utils.trajectory_validator import TrajectoryValidator
validator = TrajectoryValidator()
# Validate from file path
is_valid = validator.validate("trajectory.json")
# Validate from dict
trajectory_dict = {...}
is_valid = validator.validate(trajectory_dict)
# Validate from JSON string
trajectory_json = '{"schema_version": "ATIF-v1.4", ...}'
is_valid = validator.validate(trajectory_json)
# Check errors
if not is_valid:
    for error in validator.get_errors():
        print(f"Error: {error}")
The validator:

Validates against the complete ATIF schema using Pydantic models
Checks required fields, types, and constraints
Validates sequential step IDs (starting from 1)
Validates tool call references in observations
Validates ISO 8601 timestamps
Ensures agent-only fields are only present on agent steps
Collects all errors before returning (not just the first error)
Building Custom Trajectories
Here's a complete example of building an ATIF trajectory:


from harbor.models.trajectories import (
    Trajectory,
    Agent,
    Step,
    ToolCall,
    Observation,
    ObservationResult,
    Metrics,
    FinalMetrics,
)
import json
# Build the trajectory
trajectory = Trajectory(
    schema_version="ATIF-v1.4",
    session_id="025B810F-B3A2-4C67-93C0-FE7A142A947A",
    agent=Agent(
        name="my-agent",
        version="1.0.0",
        model_name="claude-3-5-sonnet-20241022",
    ),
    steps=[
        # Step 1: User message
        Step(
            step_id=1,
            timestamp="2025-01-15T10:30:00Z",
            source="user",
            message="What is the current trading price of Alphabet (GOOGL)?",
        ),
        # Step 2: Agent action with tool call
        Step(
            step_id=2,
            timestamp="2025-01-15T10:30:02Z",
            source="agent",
            message="I will search for the current trading price for GOOGL.",
            reasoning_content="The request requires the current stock price. I will execute a tool call to retrieve this information.",
            tool_calls=[
                ToolCall(
                    tool_call_id="call_price_1",
                    function_name="financial_search",
                    arguments={"ticker": "GOOGL", "metric": "price"},
                )
            ],
            observation=Observation(
                results=[
                    ObservationResult(
                        source_call_id="call_price_1",
                        content="GOOGL is currently trading at $185.35",
                    )
                ]
            ),
            metrics=Metrics(
                prompt_tokens=520,
                completion_tokens=80,
                cached_tokens=200,
                cost_usd=0.00045,
            ),
        ),
        # Step 3: Agent response
        Step(
            step_id=3,
            timestamp="2025-01-15T10:30:05Z",
            source="agent",
            message="Alphabet (GOOGL) is trading at $185.35.",
            metrics=Metrics(
                prompt_tokens=600,
                completion_tokens=44,
                cost_usd=0.00033,
            ),
        ),
    ],
    final_metrics=FinalMetrics(
        total_prompt_tokens=1120,
        total_completion_tokens=124,
        total_cached_tokens=200,
        total_cost_usd=0.00078,
        total_steps=3,
    ),
)
# Export to JSON
with open("trajectory.json", "w") as f:
    json.dump(trajectory.to_json_dict(), f, indent=2)
# Validate the trajectory
from harbor.utils.trajectory_validator import validate_trajectory
is_valid = validate_trajectory(trajectory.to_json_dict())
print(f"Trajectory is valid: {is_valid}")
Schema Versions
ATIF follows semantic versioning. The current version is v1.4. Supported versions:

ATIF-v1.4 (current) - Added optional prompt_token_ids field for storing prompt token IDs
ATIF-v1.3 - Added optional completion_token_ids field for RL training
ATIF-v1.2 - Extended observation field to support system steps
ATIF-v1.1 - Added optional extra field at root level
ATIF-v1.0 - Initial specification
Related Resources
ATIF RFC Specification
Real-world trajectory examples
Harbor Pydantic models
Trajectory validator source



Task Structure
Creating and running tasks for agentic environments

The Harbor task format is designed to be maximally flexible while still being intuitive to implement. The differences between the Harbor task format and the Terminal-Bench task format are documented here.

Creating a task
To create a task, you can use the following command:


harbor tasks init "<task-name>"
This will create a new task directory with the following structure:

instruction.md
task.toml
environment
Dockerfile
...
solution
solve.sh
...
tests
test.sh
...
You can then populate the files with your task's content.

To evaluate an agent on your task, you can use the following command:


harbor run -p "<path/to/task>" -a "<agent>" -m "<model>"
Task files
Instruction
The instruction is a markdown file that contains the task's instruction.

Configuration & Metadata
The task.toml file contains the task's configuration and metadata. Metadata is arbitrary and can consist of any information a task developer wants. Config params are nested into their respective sections rather than flat.

An example is shown below:


version = "1.0"
[metadata]
author_name = "Steve Jobs"
author_email = "steve@apple.com"
difficulty = "easy"
category = "programming"
tags = ["trivial"]
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
build_timeout_sec = 600.0
docker_image = "some-org/some-name:some-tag"
cpus = 1
memory_mb = 2048
storage_mb = 10240
The configuration parameters are shown below:

Prop

Type


version
string

metadata?
object

verifier.timeout_sec?
number

agent.timeout_sec?
number

environment.build_timeout_sec?
number

environment.docker_image?
string | null

environment.cpus?
integer

environment.memory_mb?
integer

environment.storage_mb?
integer

source?
string | null
Environment
The environment definition is placed in an environment/ folder. Harbor does not require any specific file to exist in that directory. Which file is required depends on the environment type being used for the evaluation. For example, to use --env docker, the DockerEnvironment class checks that an environment/Dockerfile or environment/docker-compose.yaml is present. Different environment types could require other files to be present (e.g. an Apptainer environment could check for an image.def file). Most cloud sandbox providers only support Dockerfile defined environments and not docker compose.

There are a few special paths in the environment's filesystem:

Path	Description
/logs/verifier/	Contains the reward file and other verifier logs.
/logs/agent/	A directory agents can use to store logs from their runs.
/solution/	The solution folder is copied here by the Oracle agent at runtime and executed from the working directory.
/tests/	The tests folder is copied here by the Harbor harness at runtime and executed from the working directory.
The /logs/ directory is downloaded to the host after the agent/verifier run and are often useful for debugging and analysis.

Solution (Optional)
The solution folder must contain a solution/solve.sh script. Other dependencies are allowed. This folder is copied to /solution by the Oracle agent at runtime and executed from the working directory.

If no solution is provided, the Oracle agent cannot be used to sanity check the task.

Tests
The tests folder must contain a tests/test.sh script. The test script should install test dependencies and verify the agent completed the instruction. In Terminal-Bench, this was done by running a pytest command, but this is now left to the task developer.

Other dependencies are allowed in the tests/ folder. This folder is copied to /tests by the Harbor harness at runtime and executed from the working directory. E.g. bash /tests/test.sh is executed from /app in many cases.

We recommend using absolute paths in your test script to avoid relative path issues.

Importantly, the test script must produce a reward file in the /logs/verifier/ directory. This is the file that the verifier will read to determine if the task was successful.

There are two ways to produce a reward file:

Reward File	Format	Description
/logs/verifier/reward.txt	Plain text (e.g. 1)	A plain text file containing a single integer or float value, typically 1 for success or 0 for failure.
/logs/verifier/reward.json	JSON (e.g. { "runtime_sec": 1.23, "accuracy": 0.95, ... })	A JSON file that can define multiple metrics as rewards, but they must be floats or integers.
You may use either reward.txt or reward.json as the output of your test script. Harbor will read reward.txt by default and fall back to reward.json.

Often, a reward can be determined by the exit code or a unit test command.

tests/test.sh

#!/bin/bash
uvx pytest /tests/test.py
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi


Tutorial
In this tutorial, we will walk through creating a simple Harbor task.

Step 0: Install Harbor
Follow our installation instructions to install Harbor, which involves installing the package and its dependencies.

Step 1: Create your task
Now that Harbor is installed, run the following command to create a new task directory with the required files:


harbor tasks init ssh-key-pair
This will generate a task directory with the following structure:


ssh-key-pair/
├── instruction.md         # Task instructions
├── task.toml              # Configuration and metadata
├── environment/
│   └── Dockerfile         # Container definition
├── solution/
│   └── solve.sh           # Solution script
└── tests/
    │── test_outputs.py    # Pytest unit tests
    └── test.sh            # Test verification script
Step 2: Write the task instructions
Open the instruction.md file in your task directory and add the task description:

ssh-key-pair/instruction.md

# SSH Key Pair Generation
Generate an SSH key pair in the files `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`.
Don't make them password protected.
Step 3: Configure task metadata
Open the task.toml file and configure your task metadata:

ssh-key-pair/task.toml

version = "1.0"
[metadata]
author_name = "Your Name"
author_email = "your.email@example.com"
difficulty = "easy"
category = "system-administration"
tags = ["ssh", "cryptography", "linux"]
[verifier]
timeout_sec = 120.0
[agent]
timeout_sec = 120.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory = "2G"
storage = "10G"
Step 4: Create the task environment
Open the Dockerfile in the environment/ directory that was generated:

ssh-key-pair/environment/Dockerfile

FROM ubuntu:24.04
# Create working directory
WORKDIR /app
# Install openssh-client for the task
RUN apt-get update && apt-get install -y openssh-client && rm -rf /var/lib/apt/lists/*
This Dockerfile defines the environment an agent will interact with through the terminal. Add any dependencies your task requires here.

Step 5: Test your solution idea
Before writing the automated solution, you'll want to manually verify your approach works. Build and run the container interactively:


harbor tasks start-env -p ssh-key-pair -e docker -a -i # or use daytona or modal
Inside the container, test that the following command solves the task without requiring interactive input:


ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
Verify the keys were created correctly:


ls -l ~/.ssh/id_rsa*
You should see:


~/.ssh/
├── id_rsa      (-rw-------  600  private key)
└── id_rsa.pub  (-rw-r--r--  644  public key)
Exit the container with exit or Ctrl+D.

Step 6: Write the solution script
Take the command you verified in the previous step and create the solution script. This file will be used by the Oracle agent to ensure the task is solvable.

Update the solution/solve.sh file:

ssh-key-pair/solution/solve.sh

#!/bin/bash
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
Make sure the script is executable:


chmod +x ssh-key-pair/solution/solve.sh
Step 7: Create the test script
The test script verifies whether the agent successfully completed the task. It must produce a reward file in /logs/verifier/.

Update the tests/test.sh file:

ssh-key-pair/tests/test.sh

#!/bin/bash
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
# Run pytest tests
uvx \
  --python 3.12 \
  --with pytest==8.4.1 \
  pytest /tests/test_outputs.py
# Check exit code and write reward
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
Now create the Python test file:

ssh-key-pair/tests/test_outputs.py

import os
from pathlib import Path
def test_key_files_exist() -> None:
    """Test that both private and public key files exist."""
    private_key = Path.home() / ".ssh" / "id_rsa"
    public_key = Path.home() / ".ssh" / "id_rsa.pub"
    assert private_key.exists(), "Private key file does not exist"
    assert public_key.exists(), "Public key file does not exist"
def test_key_file_permissions() -> None:
    """Test that the key files have correct permissions."""
    private_key = Path.home() / ".ssh" / "id_rsa"
    public_key = Path.home() / ".ssh" / "id_rsa.pub"
    private_perms = oct(os.stat(private_key).st_mode)[-3:]
    public_perms = oct(os.stat(public_key).st_mode)[-3:]
    assert private_perms == "600", (
        f"Private key has incorrect permissions: {private_perms}"
    )
    assert public_perms == "644", (
        f"Public key has incorrect permissions: {public_perms}"
    )
def test_key_format() -> None:
    """Test that the public key has the correct RSA format."""
    public_key = Path.home() / ".ssh" / "id_rsa.pub"
    with open(public_key, 'r') as f:
        content = f.read()
    assert content.startswith("ssh-rsa "), "Public key does not start with 'ssh-rsa'"
    assert len(content.split()) >= 2, "Public key format is invalid"
Step 8: Test your task with the Oracle agent
Run the following command to verify your task is solved by the solution script:


harbor run -p ssh-key-pair -a oracle
If successful, you should see output indicating the task was completed and the reward was 1.

Troubleshooting

If the Oracle agent fails, check:

The solution script has execute permissions
The Dockerfile installs all required dependencies
The test script correctly writes to /logs/verifier/reward.txt
The paths in your tests match the paths in your solution
Step 9 (Optional): Test with a real agent
Test your task with an actual AI agent to see if it can solve the task. For example, using Terminus with Claude:


harbor run \
  -p ssh-key-pair \
  -a terminus-2 \
  -m anthropic/claude-haiku-4-5
Step 10: Contribute your task!
Congratulations! You've created your first Harbor task. Your task is now ready to be used for benchmarking AI agents!


Congratulations! You've created your first Harbor task. Your task is now ready to be used for benchmarking AI agents!


Datasets
Running a dataset

A Harbor task is an instruction, container environment, and test script. Datasets are collections of tasks used for evals and training.

There are two types of datasets:

Local datasets which are datasets that are stored on the local machine.
Registry datasets which are datasets that are stored in a git repository and registered in a json file.
Local datasets
A local dataset is a directory that contains a set of tasks. To evaluate on a local dataset, use the following command:


harbor run -p "<path/to/dataset>" -a "<agent>" -m "<model>" 
Harbor registry
Harbor comes with a default registry defined in a registry.json file stored in the repository root.

Simply use the --dataset or -d flag to reference a dataset by name and version:


harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>" 
A dataset has the following structure:


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
        },
        ...
    ]
}
Datasets can contain tasks from multiple repositories.

The Harbor registry is currently only intended to house benchmarks and popular training datasets. Consider submitting a PR to add your dataset to the registry to take advantage of Harbor's distribution.

Custom registry
Sometimes, you may want to create your own registry to store private datasets. You can define your own registry.json file and use the --registry-path flag to point to it (or host it at a URL and use the --registry-url flag).


harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>" --registry-path "<path/to/registry.json>"
# Or to host it at a URL
harbor run -d "my-dataset@1.0" -a "<agent>" -m "<model>" --registry-url "<url/to/registry.json>"



Cloud Deployments
Horizontal scaling using cloud sandboxes

Containerized agentic tasks can be slow when performing rollouts. This is due to container startup and teardown overhead, waiting for LLM API calls, and waiting for command execution. Horizontal scaling becomes the only viable way to accelerate experimentation, so we recommend using a cloud sandbox provider like Daytona.

Using a cloud sandbox provider shifts command execution to the cloud, making trials I/O bounded rather than compute bounded. This means you can typically parallelize far above your CPU count.

Using a cloud sandbox provider
There are many cloud sandbox providers to choose from. We recommend Daytona, because we have found them to be the most flexible. Other good options are Modal and E2B.


harbor run -d "<dataset@version>" \
  -m "<model>" \
  -a "<agent>" \
  -e daytona \
  -n "<n-parallel-trials>"
We run up to 100 trials in parallel on a MacBook Pro with 14 cores.

Limitations
All cloud sandbox providers we have tried do not support multi-container environments. Until one does, or you are willing to implement from scratch using Kubernetes (please submit a PR), you won't be able to run multi-container tasks in the cloud.

However, the Docker environment still supports multi-container tasks. Just make sure to include an environment/docker-compose.yaml file in your task definiton.


Evals
Running a dataset

Harbor is built by the creators of Terminal-Bench with evals as a core use case.

What is a dataset?
In Harbor, a dataset is a collection of tasks in the Harbor task format. Tasks are agentic environments consisting of an instruction, container environment, and test script.

Datasets can be used to evaluate agents and models, to train models, or to optimize prompts and other aspects of an agent.

Viewing registered benchmarks
Harbor comes with a default registry defined in a registry.json file stored in the repository root.

To view all available datasets, you can use the following command:


harbor datasets list
Running a benchmark from the registry
To evaluate on Terminal-Bench, you can use the following command:


harbor run -d terminal-bench@2.0 -m "<model>" -a "<agent>"
Harbor will automatically download the dataset based on the registry definition (which points to version controlled task definitions).

To evaluate on SWE-Bench Verified:


harbor run -d swe-bench-verified@1.0 -m "<model>" -a "<agent>"
If you leave off the version, Harbor will use the latest version of the dataset.

Running a local dataset
If you want to evaluate on a local dataset, you can use the following command:


harbor run -p "<path/to/dataset>" -m "<model>" -a "<agent>"
Analyzing results
Running the harbor run command creates a job which by default is stored in the jobs directory.

The file structure looks something like this:


jobs/job-name
├── config.json               # Job config
├── result.json               # Job result
├── trial-name
│   ├── config.json           # Trial config
│   ├── result.json           # Trial result
│   ├── agent                 # Agent directory, contents depend on agent implementation
│   │   ├── recording.cast
│   │   └── trajectory.json
│   └── verifier              # Verifier directory, contents depend on test.sh implementation
│       ├── ctrf.json
│       ├── reward.txt
│       ├── test-stderr.txt
│       └── test-stdout.txt
└── ...                       # More trials


└── ...                       # More trials



SFT
Generating SFT datasets from Harbor trials

Harbor includes utilities for turning trials (agent task completion attempts) into conversational traces that can be fed into supervised fine-tuning pipelines for agentic LLMs. Export helpers live under harbor.utils.traces_utils and power several CLI entry points.

Requires ATIF trajectory format

The SFT exporter works with any agent that produces trajectories in the ATIF format. This includes Terminus-2, OpenHands, Claude Code, Gemini CLI, and other agents with ATIF support. Agents without ATIF trajectory support cannot be exported.

For best results with SFT data export, configure Terminus-2 with appropriate trajectory settings:


from harbor.models.agent.trajectory_config import TrajectoryConfig
trajectory_config = TrajectoryConfig(
    raw_content=True,      # Preserve exact LLM responses for SFT
    linear_history=True    # Split on summarization for clean sequences
)
Learn more about trajectory configuration options in the Terminus-2 documentation.

Each exported row represents one agent/episode-* directory and captures the input debug.json messages plus the final agent reply from response.json or response.txt.
Rows include metadata such as agent, model, model_provider, task, trial_name, episode, and run_id, letting you merge runs from multiple jobs.
--sharegpt adds a ShareGPT-style column to support instruction-tuning datasets expecting the {"from": "...", "value": "..."} schema.
Success filtering (--filter success|failure) inspects result.json and lets you keep only passing or failing attempts for curriculum-style datasets.
Run harbor traces export on a trial directory (or a parent directory) to build a datasets.Dataset. The command prints the number of rows produced and, when --push is set, uploads directly to the Hugging Face Hub.


harbor traces export \
  --path trials \
  --recursive \
  --episodes last \
  --filter success \
  --sharegpt \
  --push \
  --repo my-org/harbor-terminus2-sft
Key options
Prop

Type


--episodes?
all | last

--sharegpt / --no-sharegpt?
flag

--filter?
success | failure | all

--push?
flag

--verbose?
flag
If you want to persist the dataset locally (e.g., to Parquet), call the Python helper directly:


from harbor.utils.traces_utils import export_traces
dataset = export_traces("trials", episodes="last", success_filter="success")
dataset.to_parquet("harbor-terminus2-success.parquet")
The datasets library is an optional dependency; install it if you plan to export traces.

harbor run can export traces automatically once a job completes. Pass trace flags alongside your job invocation:


harbor run \
  --config examples/configs/job.yaml \
  --agent claude-code \
  --model anthropic/claude-3-sonnet-20240229 \
  --export-traces \
  --export-sharegpt \
  --export-episodes last \
  --export-push \
  --export-repo my-org/harbor-job-run
When --export-traces is set, Harbor exports from the produced job directory using the same machinery as harbor traces export. The --export-* options mirror the standalone CLI flags and default to in-memory exports unless --export-push is provided. Errors during export are surfaced at the end of the job run without interrupting evaluation.

harbor sweeps run can emit split datasets that separate successful and failed trajectories. Supply --push together with one of the repo arguments:


# Push a DatasetDict with "success" and "failure" splits
harbor sweeps run \
  --config examples/configs/job.yaml \
  --max-sweeps 3 \
  --trials-per-task 2 \
  --push \
  --export-repo my-org/harbor-sweeps
You can also push successes and failures to independent repos by combining --push with --export-separate (alias --no-export-splits) plus --export-repo-success and --export-repo-failure. These exports reuse the same trace discovery logic and default to the last episode from each trial.