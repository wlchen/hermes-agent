# Hermes Agent — Architecture Document

> **Last updated:** April 2026

## Table of Contents

- [1. Overview](#1-overview)
- [2. High-Level Architecture](#2-high-level-architecture)
- [3. Entry Points](#3-entry-points)
- [4. Core Agent Loop](#4-core-agent-loop)
- [5. Tool System](#5-tool-system)
- [6. Agent Internals](#6-agent-internals)
- [7. CLI Subsystem](#7-cli-subsystem)
- [8. Gateway & Messaging Platforms](#8-gateway--messaging-platforms)
- [9. State & Persistence](#9-state--persistence)
- [10. Supporting Systems](#10-supporting-systems)
- [11. Configuration](#11-configuration)
- [12. Data Flow — Message Lifecycle](#12-data-flow--message-lifecycle)
- [13. Key Architectural Patterns](#13-key-architectural-patterns)
- [14. Extension Points](#14-extension-points)

---

## 1. Overview

Hermes Agent is a modular, multi-platform AI agent framework built on top of
the OpenAI-compatible chat completions API. It supports tool-calling loops,
multiple LLM providers, persistent memory, interactive CLI, messaging gateway
integration (Telegram, Discord, Slack, etc.), batch processing, RL training
environments, and an extensible plugin/skill system.

The project is written in Python and follows a **self-registration** pattern
for tools, a **callback-driven** architecture for UI integration, and a
**profile-isolated** configuration model for multi-instance deployments.

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        CLI["hermes CLI<br/>(cli.py)"]
        GW["Gateway<br/>(gateway/run.py)"]
        BATCH["Batch Runner<br/>(batch_runner.py)"]
        MCP["MCP Server<br/>(mcp_serve.py)"]
        ACP["ACP Adapter<br/>(acp_adapter/)"]
        RL["RL Environments<br/>(environments/)"]
    end

    subgraph "Core Engine"
        AGENT["AIAgent<br/>(run_agent.py)"]
        MT["Model Tools<br/>(model_tools.py)"]
        TS["Toolsets<br/>(toolsets.py)"]
    end

    subgraph "Agent Internals"
        PB["Prompt Builder"]
        CC["Context Compressor"]
        PC["Prompt Caching"]
        AUX["Auxiliary Client"]
        MM["Model Metadata"]
        DISP["Display / Spinner"]
    end

    subgraph "Tool Registry"
        REG["tools/registry.py"]
        TW["Web Tools"]
        TF["File Tools"]
        TT["Terminal Tool"]
        TB["Browser Tools"]
        TV["Vision / Image"]
        TD["Delegate Tool"]
        TMCP["MCP Tool"]
        TO["Other Tools..."]
    end

    subgraph "Platforms (Gateway)"
        TELE["Telegram"]
        DISC["Discord"]
        SLK["Slack"]
        MTX["Matrix"]
        WA["WhatsApp"]
        SIG["Signal"]
        EML["Email"]
        OTH["DingTalk / Feishu /<br/>WeCom / SMS / Webhook /<br/>Mattermost / Home Assistant"]
    end

    subgraph "Persistence"
        DB["SessionDB<br/>(hermes_state.py)<br/>SQLite + FTS5"]
        CFG["Config<br/>(~/.hermes/config.yaml)"]
        ENV["Env Vars<br/>(~/.hermes/.env)"]
        MEM["Memory<br/>(MEMORY.md)"]
        SKILLS["Skills<br/>(~/.hermes/skills/)"]
    end

    subgraph "Execution Environments"
        LOCAL["Local Shell"]
        DOCKER["Docker"]
        SSH["SSH Remote"]
        MODAL["Modal"]
        DAYTONA["Daytona"]
        SING["Singularity"]
    end

    CLI --> AGENT
    GW --> AGENT
    BATCH --> AGENT
    MCP --> AGENT
    ACP --> AGENT
    RL --> AGENT

    AGENT --> MT
    MT --> REG
    MT --> TS

    AGENT --> PB
    AGENT --> CC
    AGENT --> PC
    AGENT --> AUX
    AGENT --> MM
    CLI --> DISP

    REG --> TW
    REG --> TF
    REG --> TT
    REG --> TB
    REG --> TV
    REG --> TD
    REG --> TMCP
    REG --> TO

    GW --> TELE
    GW --> DISC
    GW --> SLK
    GW --> MTX
    GW --> WA
    GW --> SIG
    GW --> EML
    GW --> OTH

    TT --> LOCAL
    TT --> DOCKER
    TT --> SSH
    TT --> MODAL
    TT --> DAYTONA
    TT --> SING

    AGENT --> DB
    AGENT --> MEM
    CLI --> CFG
    CLI --> ENV
    AGENT --> SKILLS
```

---

## 3. Entry Points

Hermes Agent supports multiple entry points, all converging on the same
`AIAgent` core.

```mermaid
graph LR
    subgraph "User Interfaces"
        A["$ hermes<br/>(Interactive REPL)"]
        B["$ hermes -q 'query'<br/>(Single-shot)"]
        C["$ hermes gateway<br/>(Messaging)"]
        D["$ hermes mcp serve<br/>(MCP Server)"]
        E["$ hermes cron<br/>(Scheduler)"]
        F["batch_runner.py<br/>(Parallel batch)"]
    end

    A --> MAIN["hermes_cli/main.py<br/>CLI Dispatcher"]
    B --> MAIN
    C --> MAIN
    D --> MAIN
    E --> MAIN
    F --> CORE["AIAgent"]

    MAIN -->|"chat"| CLI["cli.py<br/>HermesCLI"]
    MAIN -->|"gateway"| GW["gateway/run.py<br/>GatewayRunner"]
    MAIN -->|"mcp serve"| MCP["mcp_serve.py"]

    CLI --> CORE
    GW --> CORE
    MCP --> CORE
```

| Entry Point | File | Description |
|-------------|------|-------------|
| **Interactive REPL** | `hermes_cli/main.py` → `cli.py` | Rich terminal UI with autocomplete, spinner, skin engine |
| **Single-shot** | `hermes_cli/main.py` → `cli.py` | One query → one response, then exit |
| **Gateway** | `gateway/run.py` | Multi-platform messaging (Telegram, Discord, etc.) |
| **Batch Runner** | `batch_runner.py` | Parallel dataset processing with checkpointing |
| **MCP Server** | `mcp_serve.py` | Expose agent as MCP tool for Claude Desktop / editors |
| **ACP Adapter** | `acp_adapter/` | VS Code / Zed / JetBrains Copilot integration |
| **RL Environments** | `environments/` | Atropos-compatible RL training harness |
| **Cron Scheduler** | `cron/scheduler.py` | Periodic task execution |

**Profile isolation:** Before any imports, `_apply_profile_override()` in
`hermes_cli/main.py` sets `HERMES_HOME` to scope all state to the active
profile directory.

---

## 4. Core Agent Loop

The `AIAgent` class in `run_agent.py` is the heart of the system. It manages
the conversation loop, tool execution, context compression, and session
persistence.

### 4.1 Agent Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant Agent as AIAgent
    participant Prompt as PromptBuilder
    participant LLM as LLM Provider
    participant Tools as ToolRegistry
    participant DB as SessionDB

    User->>Agent: run_conversation(message)
    Agent->>DB: Load history (if session exists)
    Agent->>Prompt: Build system prompt (cached)
    
    loop Tool-Calling Loop (max 90 iterations)
        Agent->>LLM: chat.completions.create(messages, tools)
        LLM-->>Agent: Response (text or tool_calls)
        
        alt Has tool_calls
            loop For each tool call
                Agent->>Tools: handle_function_call(name, args)
                Tools-->>Agent: Result (JSON string)
                Agent->>Agent: Append tool result to messages
            end
            Agent->>Agent: budget.consume() — check remaining
        else Final text response
            Agent->>Agent: Break loop
        end
    end

    Agent->>Agent: Context compression (if needed)
    Agent->>DB: Persist messages + session metadata
    Agent-->>User: Return {response, messages, usage, ...}
```

### 4.2 Key AIAgent Parameters

```python
class AIAgent:
    def __init__(self,
        model="anthropic/claude-opus-4.6",  # Model identifier
        max_iterations=90,                   # Tool-call budget
        enabled_toolsets=None,               # Whitelist toolsets
        disabled_toolsets=None,              # Blacklist toolsets
        session_id=None,                     # Resume session
        platform=None,                       # "cli", "telegram", etc.
        skip_memory=False,                   # Disable MEMORY.md
        skip_context_files=False,            # Disable AGENTS.md etc.
        save_trajectories=False,             # Export ShareGPT format
        # ... 40+ more params for callbacks, routing, credentials
    )
```

### 4.3 Iteration Budget

The `IterationBudget` provides thread-safe budget tracking with warning
thresholds:
- **70% consumed** → caution message injected into tool results
- **90% consumed** → urgent warning injected
- **100% consumed** → loop terminates

---

## 5. Tool System

### 5.1 Registry Architecture

```mermaid
graph TB
    subgraph "Registration (import-time)"
        WT["web_tools.py"] -->|register| REG["ToolRegistry<br/>(tools/registry.py)"]
        FT["file_tools.py"] -->|register| REG
        TT["terminal_tool.py"] -->|register| REG
        BT["browser_tool.py"] -->|register| REG
        VT["vision_tools.py"] -->|register| REG
        DT["delegate_tool.py"] -->|register| REG
        MT["mcp_tool.py"] -->|register| REG
        OT["... 20+ more"] -->|register| REG
    end

    subgraph "Discovery (model_tools.py)"
        DISC["_discover_tools()"] -->|import all| WT
        DISC -->|import all| FT
        DISC -->|import all| TT
        DISC -->|import all| BT
    end

    subgraph "Resolution (toolsets.py)"
        TS["resolve_toolset()"] -->|flatten| SETS["TOOLSETS dict"]
    end

    subgraph "Consumption"
        GD["get_tool_definitions()"] --> REG
        GD --> TS
        HFC["handle_function_call()"] --> REG
    end

    DISC --> GD
    GD -->|OpenAI schemas| AGENT["AIAgent"]
    HFC -->|dispatch| AGENT
```

### 5.2 Tool Registration

Each tool file self-registers at import time — no central manifest required:

```python
# tools/your_tool.py
from tools.registry import registry

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(args),
    check_fn=check_requirements,      # Availability check
    requires_env=["EXAMPLE_API_KEY"],  # Required env vars
)
```

### 5.3 Complete Tool Inventory

| Toolset | Tools | Description |
|---------|-------|-------------|
| **web** | `web_search`, `web_extract` | Web research and content extraction |
| **files** | `read_file`, `write_file`, `patch`, `search_files` | Filesystem operations |
| **terminal** | `terminal`, `process` | Shell execution and process management |
| **browser** | `browser_navigate`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_get_images`, `browser_vision`, `browser_snapshot`, `browser_console` | Browser automation |
| **vision** | `vision_analyze` | Image analysis via auxiliary LLM |
| **image_gen** | `image_generate` | DALL-E 3 image generation |
| **skills** | `skills_list`, `skill_view`, `skill_manage` | Skill management |
| **code_execution** | `execute_code` | Sandboxed code execution |
| **delegation** | `delegate_task` | Sub-agent delegation |
| **mcp** | *(dynamic)* | MCP server integration |
| **memory** | `memory` | Persistent MEMORY.md |
| **todo** | `todo` | Task list management |
| **session** | `session_search` | FTS5 history search |
| **tts** | `text_to_speech` | Text-to-speech synthesis |
| **homeassistant** | `ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service` | Smart home control |
| **messaging** | `send_message` | Cross-platform message sending |
| **cronjob** | `cronjob` | Scheduled task management |
| **moa** | `mixture_of_agents` | Multi-agent voting ensemble |

### 5.4 Toolset Composition

Toolsets can include other toolsets for composition:

```python
TOOLSETS = {
    "web": {"tools": ["web_search", "web_extract"], "includes": []},
    "research": {"tools": [], "includes": ["web", "files", "terminal"]},
    "full_stack": {"tools": [], "includes": ["research", "browser", "code_execution"]},
}
```

`resolve_toolset()` recursively flattens these to a final tool list.

---

## 6. Agent Internals

The `agent/` package contains modules that support the core agent loop.

```mermaid
graph TB
    AGENT["AIAgent<br/>(run_agent.py)"]

    subgraph "agent/ package"
        PB["prompt_builder.py<br/>System prompt assembly"]
        CC["context_compressor.py<br/>LLM-based summarization"]
        PC["prompt_caching.py<br/>Anthropic cache_control"]
        AUX["auxiliary_client.py<br/>Side-task LLM routing"]
        MM["model_metadata.py<br/>Context lengths, probing"]
        MD["models_dev.py<br/>models.dev registry"]
        DISP["display.py<br/>KawaiiSpinner, previews"]
        SC["skill_commands.py<br/>Skill slash commands"]
        TRAJ["trajectory.py<br/>ShareGPT export"]
        MMAN["memory_manager.py<br/>Memory block builder"]
        AA["anthropic_adapter.py<br/>Native Anthropic support"]
        UP["usage_pricing.py<br/>Cost estimation"]
    end

    AGENT --> PB
    AGENT --> CC
    AGENT --> PC
    AGENT --> AUX
    AGENT --> MM
    AGENT --> DISP
    AGENT --> TRAJ
    AGENT --> MMAN
    AGENT --> AA
    AGENT --> UP
```

### 6.1 System Prompt Assembly

`prompt_builder.py` constructs the system prompt from layered components:

```
┌─────────────────────────────────────────┐
│ 1. Identity (SOUL.md or default)        │
├─────────────────────────────────────────┤
│ 2. Tool Guidance                        │
│    • Memory guidance                    │
│    • Session search guidance            │
│    • Skills guidance                    │
├─────────────────────────────────────────┤
│ 3. Tool-Use Enforcement                 │
│    (tell model to call, not describe)   │
├─────────────────────────────────────────┤
│ 4. Model-Specific Guidance              │
│    (Google, OpenAI, Anthropic tweaks)   │
├─────────────────────────────────────────┤
│ 5. Memory Snapshot                      │
│    • MEMORY.md content                  │
│    • USER.md content                    │
├─────────────────────────────────────────┤
│ 6. Skills Index                         │
├─────────────────────────────────────────┤
│ 7. Context Files (threat-scanned)       │
│    • AGENTS.md, .cursorrules            │
├─────────────────────────────────────────┤
│ 8. Timestamp + Model Info               │
├─────────────────────────────────────────┤
│ 9. Platform Hints                       │
└─────────────────────────────────────────┘
```

The system prompt is built once per session and cached for Anthropic prompt
caching (75% cost reduction on multi-turn conversations).

### 6.2 Context Compression

When conversation history exceeds the context window threshold:

1. **Prune old tool results** (cheap, no LLM call)
2. **Protect head** (system prompt + first exchange)
3. **Protect tail** (last ~20K tokens)
4. **Summarize middle** turns with auxiliary LLM
5. **Iteratively update** summary on subsequent compactions

### 6.3 Auxiliary Client Router

Routes side tasks (compression, vision, web extraction) to the best
available LLM. Resolution chain:

```
OpenRouter → Nous Portal → Custom Endpoint → Codex OAuth →
Native Anthropic → Direct API (Kimi/GLM/MiniMax) → None
```

---

## 7. CLI Subsystem

### 7.1 CLI Architecture

```mermaid
graph TB
    subgraph "hermes_cli/"
        MAIN["main.py<br/>Entry point & dispatcher"]
        CFG["config.py<br/>Config loading & migration"]
        CMD["commands.py<br/>Slash command registry"]
        CB["callbacks.py<br/>Terminal callbacks"]
        SETUP["setup.py<br/>Setup wizard"]
        SKIN["skin_engine.py<br/>Theme engine"]
        SC["skills_config.py<br/>Skill management"]
        TC["tools_config.py<br/>Tool management"]
        SH["skills_hub.py<br/>Skill discovery"]
        MOD["models.py<br/>Model catalog"]
        MS["model_switch.py<br/>Model switching"]
        AUTH["auth.py<br/>Credential resolution"]
        BAN["banner.py<br/>Welcome banner"]
        DOC["doctor.py<br/>Health checks"]
    end

    MAIN -->|"chat"| CLI["cli.py<br/>HermesCLI"]
    CLI --> CMD
    CLI --> CB
    CLI --> SKIN
    CLI --> BAN

    MAIN -->|"setup"| SETUP
    MAIN -->|"model"| MS
    MAIN -->|"doctor"| DOC
    MAIN -->|"config"| CFG
    MAIN -->|"skills"| SC
    MAIN -->|"tools"| TC
```

### 7.2 Slash Command Registry

All slash commands are defined centrally in `hermes_cli/commands.py` as
`CommandDef` objects. Downstream consumers derive automatically:

```mermaid
graph LR
    REG["COMMAND_REGISTRY<br/>(commands.py)"]

    REG --> CLI["CLI dispatch<br/>process_command()"]
    REG --> GW["Gateway dispatch"]
    REG --> TG["Telegram bot menu"]
    REG --> SLK["Slack subcommands"]
    REG --> AC["Autocomplete"]
    REG --> HELP["Help text"]
```

### 7.3 Skin Engine

The skin engine (`hermes_cli/skin_engine.py`) provides data-driven CLI
theming — pure data, no code changes needed:

| Element | Skin Key |
|---------|----------|
| Banner border color | `colors.banner_border` |
| Spinner faces | `spinner.thinking_faces` |
| Spinner verbs | `spinner.thinking_verbs` |
| Tool output prefix | `tool_prefix` |
| Agent name | `branding.agent_name` |
| Response box label | `branding.response_label` |

Built-in skins: **default** (gold/kawaii), **ares** (crimson), **mono**
(grayscale), **slate** (blue).

---

## 8. Gateway & Messaging Platforms

### 8.1 Gateway Architecture

```mermaid
graph TB
    subgraph "Gateway Runner"
        GR["GatewayRunner<br/>(gateway/run.py)"]
        SS["SessionStore<br/>(gateway/session.py)"]
        HOOKS["Hook System<br/>(gateway/hooks.py)"]
    end

    subgraph "Platform Adapters"
        BASE["PlatformAdapter<br/>(platforms/base.py)"]
        TELE["Telegram"]
        DISC["Discord"]
        SLK["Slack"]
        MTX["Matrix"]
        WA["WhatsApp"]
        SIG["Signal"]
        EML["Email"]
        MT["Mattermost"]
        DT["DingTalk"]
        FS["Feishu"]
        WC["WeCom"]
        SMS["SMS (Twilio)"]
        WH["Webhook"]
        HA["Home Assistant"]
    end

    GR --> BASE
    BASE --> TELE
    BASE --> DISC
    BASE --> SLK
    BASE --> MTX
    BASE --> WA
    BASE --> SIG
    BASE --> EML
    BASE --> MT
    BASE --> DT
    BASE --> FS
    BASE --> WC
    BASE --> SMS
    BASE --> WH
    BASE --> HA

    GR --> SS
    GR --> HOOKS

    TELE --> AGENT["AIAgent<br/>(fresh per message)"]
    DISC --> AGENT
    SLK --> AGENT
    MTX --> AGENT
    WA --> AGENT
    SIG --> AGENT
    EML --> AGENT
    MT --> AGENT
```

### 8.2 Message Flow (Gateway)

```mermaid
sequenceDiagram
    participant Platform as Platform (Telegram/Discord/...)
    participant Adapter as PlatformAdapter
    participant GW as GatewayRunner
    participant Agent as AIAgent
    participant DB as SessionDB

    Platform->>Adapter: Incoming message
    Adapter->>GW: on_message(user_id, text, platform_meta)
    GW->>DB: Load/create session for user
    GW->>Agent: Create fresh AIAgent(session_id=...)
    Agent->>Agent: run_conversation(message)
    Agent-->>GW: Response text
    GW->>DB: Persist updated session
    GW->>Adapter: send_response(user_id, text)
    Adapter->>Platform: Deliver message
```

### 8.3 Platform Feature Matrix

| Platform | Threads | Reactions | Media | Groups | E2E Encryption |
|----------|---------|-----------|-------|--------|----------------|
| Telegram | — | ✓ | ✓ | ✓ | — |
| Discord | ✓ | ✓ | ✓ | ✓ | — |
| Slack | ✓ | ✓ | ✓ | ✓ | — |
| Matrix | — | — | ✓ | ✓ | ✓ |
| WhatsApp | — | — | ✓ | ✓ | ✓ |
| Signal | — | ✓ | — | ✓ | ✓ |
| Email | — | — | ✓ | — | — |

---

## 9. State & Persistence

### 9.1 Session Database

```mermaid
erDiagram
    sessions {
        text id PK
        text source
        text user_id
        text model
        text system_prompt
        text parent_session_id FK
        text started_at
        text ended_at
        int message_count
        int input_tokens
        int output_tokens
        int cache_read_tokens
        real estimated_cost
    }

    messages {
        text id PK
        text session_id FK
        text role
        text content
        text tool_call_id
        text tool_calls
        text tool_name
        text timestamp
        int token_count
        text reasoning
    }

    messages_fts {
        text content
        text session_id
        text role
    }

    sessions ||--o{ messages : contains
    sessions ||--o| sessions : "parent chain"
    messages ||--|| messages_fts : "indexed by"
```

**Technology:** SQLite3 with WAL mode (concurrent reads + single writer)

**Features:**
- FTS5 full-text search across all messages
- Parent session chains (when compression splits sessions)
- Token counting and cost tracking per session
- Automatic session metadata updates

### 9.2 File-Based State

```
~/.hermes/                     # HERMES_HOME (profile-aware)
├── config.yaml                # All settings
├── .env                       # API keys
├── active_profile             # Sticky profile selection
├── sessions.db                # SQLite session database
├── MEMORY.md                  # Persistent agent memory
├── USER.md                    # User preferences
├── skills/                    # User-installed skills
│   └── SKILL.md               # Skills index
├── skins/                     # Custom YAML skins
├── auth.json                  # OAuth tokens
├── trajectories/              # Exported conversations
├── logs/                      # Application logs
└── profiles/                  # Profile directories
    └── <name>/                # Each profile = full ~/.hermes copy
```

---

## 10. Supporting Systems

### 10.1 Execution Environments

```mermaid
graph LR
    TT["terminal tool"]

    subgraph "Backends (tools/environments/)"
        LOCAL["LocalEnvironment<br/>OS shell"]
        DOCKER["DockerEnvironment<br/>Container"]
        SSH["SSHEnvironment<br/>Remote host"]
        MODAL["ModalEnvironment<br/>Cloud GPU"]
        DAYTONA["DaytonaEnvironment<br/>Dev workspace"]
        SING["SingularityEnvironment<br/>HPC"]
    end

    TT --> LOCAL
    TT --> DOCKER
    TT --> SSH
    TT --> MODAL
    TT --> DAYTONA
    TT --> SING
```

All backends implement a common interface: `execute(command) → (stdout, stderr, exit_code)`.

### 10.2 Skills System

Skills are user-created Python scripts stored in `~/.hermes/skills/` that
extend the agent's capabilities:
- Discovered via `SKILL.md` index
- Managed through `skill_manage` tool
- Injected as user messages (not system prompt) to preserve prompt caching

### 10.3 Plugin System

The `plugins/` directory supports extensibility:
- **Memory Providers** — custom backends (e.g., Honcho) implementing
  `MemoryProvider` interface
- **Hooks** — `pre_llm_call`, `on_session_start`, etc.

### 10.4 Cron Scheduler

`cron/scheduler.py` enables periodic task execution:
- Job definitions in `cron/jobs.py`
- Managed via `cronjob` tool or `hermes cron` CLI

### 10.5 RL Training Environments

The `environments/` package provides Atropos-compatible harnesses for
reinforcement learning:
- `hermes_base_env.py` — Base environment
- `hermes_swe_env/` — Software engineering tasks
- `web_research_env.py` — Web research tasks
- `terminal_test_env/` — Terminal interaction tasks

---

## 11. Configuration

### 11.1 Configuration Flow

```mermaid
graph TB
    subgraph "Sources"
        YAML["~/.hermes/config.yaml"]
        DOTENV["~/.hermes/.env"]
        SYSENV["System env vars"]
        CLI_ARGS["CLI arguments"]
    end

    subgraph "Loaders"
        LC["load_cli_config()<br/>(cli.py)"]
        LCF["load_config()<br/>(hermes_cli/config.py)"]
        GWCFG["Direct YAML load<br/>(gateway/run.py)"]
    end

    subgraph "Consumers"
        CLI["CLI Mode"]
        SETUP["hermes setup"]
        TOOLS["hermes tools"]
        GATEWAY["Gateway"]
    end

    YAML --> LC
    YAML --> LCF
    YAML --> GWCFG
    DOTENV --> SYSENV
    SYSENV --> LC
    CLI_ARGS --> LC

    LC --> CLI
    LCF --> SETUP
    LCF --> TOOLS
    GWCFG --> GATEWAY
```

### 11.2 Key Configuration Sections

```yaml
model:
  model: "anthropic/claude-opus-4.5"
  base_url: "https://openrouter.ai/api/v1"
  provider: "openrouter"

agent:
  max_iterations: 90
  tool_use_enforcement: "auto"
  skip_context_files: false
  reasoning_effort: null

memory:
  memory_enabled: false
  provider: "honcho"

compression:
  enabled: true
  threshold: 0.50

display:
  skin: "default"
  tool_progress: true
  background_process_notifications: "all"

terminal:
  environment: "local"
  docker_image: null
  ssh_host: null
```

### 11.3 Config Migration

`hermes_cli/config.py` includes a `_config_version` (currently 5). When
bumped, existing users' configs are automatically migrated on next launch.

---

## 12. Data Flow — Message Lifecycle

The complete journey of a user message through the system:

```mermaid
flowchart TD
    INPUT["User Input<br/>(CLI / Gateway / MCP / Batch)"]
    PREPROCESS["Preprocessing<br/>• Sanitize surrogates<br/>• PII hashing (gateway)<br/>• Platform normalization"]
    LOAD_HIST["Load History<br/>• SQLite or parameter<br/>• Strip budget warnings<br/>• Hydrate todos"]
    BUILD_SYS["Build System Prompt<br/>(cached per session)"]
    BUILD_MSG["Build Messages<br/>[system, ...history, user]"]
    API_CALL["LLM API Call<br/>• Apply cache control<br/>• Add reasoning config<br/>• Provider preferences<br/>• Retry on errors"]
    PARSE["Parse Response<br/>• Extract tool_calls<br/>• Extract reasoning<br/>• Extract content"]
    
    TOOL_EXEC["Tool Execution<br/>• Validate & dispatch<br/>• Parallel (read-only) or sequential<br/>• Truncate large results<br/>• Persist in storage"]
    
    BUDGET_CHECK{"Budget<br/>remaining?"}
    POST["Post-Processing<br/>• Context compression<br/>• Trajectory export<br/>• SQLite flush<br/>• Memory/skill review<br/>• Cleanup (VMs, browsers)"]
    OUTPUT["Output Routing<br/>• CLI: print + save<br/>• Gateway: platform send<br/>• Batch: aggregate stats<br/>• RL: record trajectory"]

    INPUT --> PREPROCESS
    PREPROCESS --> LOAD_HIST
    LOAD_HIST --> BUILD_SYS
    BUILD_SYS --> BUILD_MSG
    BUILD_MSG --> API_CALL
    API_CALL --> PARSE

    PARSE -->|"tool_calls"| TOOL_EXEC
    TOOL_EXEC --> BUDGET_CHECK
    BUDGET_CHECK -->|"Yes"| BUILD_MSG
    BUDGET_CHECK -->|"No"| POST

    PARSE -->|"text response"| POST
    POST --> OUTPUT
```

---

## 13. Key Architectural Patterns

| Pattern | Implementation |
|---------|---------------|
| **Self-Registration** | Tools register at import time via `registry.register()` — no central manifest |
| **Callback Injection** | UI layer (CLI/gateway) injects callbacks for tool events, streaming, status |
| **Profile Isolation** | `HERMES_HOME` env var scopes all state per profile |
| **Prompt Caching** | System prompt built once, reused across turns (Anthropic `cache_control`) |
| **Graceful Degradation** | Missing API keys disable tools rather than crash |
| **Persistent Event Loops** | Reused async loops prevent "Event loop is closed" errors |
| **Parallel Tool Execution** | Read-only tools dispatched in thread pool (8 threads) |
| **Layered Prompt Assembly** | System prompt built from composable layers |
| **Adapter Pattern** | Gateway platforms implement common `PlatformAdapter` interface |
| **Iteration Budget** | Thread-safe budget with caution/warning/hard-stop thresholds |
| **Context Compression** | LLM-based summarization when history exceeds threshold |
| **Config Migration** | Versioned config with automatic upgrade on launch |

---

## 14. Extension Points

### Adding a New Tool

1. Create `tools/your_tool.py` with `registry.register()` call
2. Add import in `model_tools.py` `_discover_tools()`
3. Add to `_HERMES_CORE_TOOLS` in `toolsets.py`

### Adding a Gateway Platform

1. Create `gateway/platforms/your_platform.py` extending `PlatformAdapter`
2. Implement `start()`, `on_message()`, `send_message()`
3. Register in gateway config

### Adding a Memory Provider Plugin

1. Create `plugins/memory/your_provider/__init__.py`
2. Implement `MemoryProvider` interface
3. Register via plugin system

### Adding a Slash Command

1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `HermesCLI.process_command()` in `cli.py`
3. Optionally add gateway handler in `gateway/run.py`

### Adding a Skin

Add to `_BUILTIN_SKINS` in `skin_engine.py` or drop a YAML file in
`~/.hermes/skins/`.

### Adding an Execution Environment

1. Create `tools/environments/your_env.py` extending base
2. Implement `execute(command)` interface
3. Register in terminal tool configuration
