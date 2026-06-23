# Smart Research Assistant — Multi-Agent AI System

A multi-agent research pipeline built with **Google ADK**, **AWS Bedrock (Claude)**, **LiteLLM**, and **Langfuse**. Three specialized AI agents collaborate to research any topic, analyze findings, and synthesize a structured report — while demonstrating production-grade capabilities: response caching, persistent memory, and content guardrails.

---

## Tech Stack

| Component | Role |
|---|---|
| **Google ADK** | Agent orchestration framework (`LlmAgent`, `InMemoryRunner`, `FunctionTool`) |
| **AWS Bedrock** | LLM provider (Claude Sonnet 4 via APAC endpoint) |
| **LiteLLM Proxy** | AI gateway — routes to Bedrock, caches responses in Redis, logs to Langfuse |
| **Redis** | Response cache backend (exact-match, TTL-based) |
| **PostgreSQL** | Persistent memory store (`researcher_memory`, `user_memory` tables) |
| **Langfuse** | Observability — traces every pipeline run and LLM call |
| **FastAPI** | Local guardrail service (weapons content filter) |

---

## Project Structure

```
research-agent-adk-litellm/
│
├── agents/                         # Agent implementations
│   ├── __init__.py
│   ├── orchestrator.py             # Root agent — decomposes query, calls sub-agents, synthesizes answer
│   ├── researcher.py               # Searches web, fetches content, produces structured research JSON
│   └── analyst.py                  # Analyzes research data, returns insights + confidence score
│
├── tools/                          # ADK FunctionTools for the Researcher agent
│   ├── __init__.py
│   └── research_tools.py           # search_web, fetch_page_content, get_current_datetime
│
├── config/                         # Shared configuration
│   ├── __init__.py
│   ├── settings.py                 # Loads all env vars via Settings class
│   └── memory_store.py             # MemoryStore (researcher_memory) + UserMemoryStore (user_memory)
│
├── main.py                         # CLI entry point (interactive, single query, demo modes)
├── guardrail_service.py            # Standalone FastAPI server — weapons content filter on :8001
├── test_components.py              # Isolated tests for guardrails and memory components
│
├── litellm_config.yaml             # LiteLLM proxy config (models, caching, logging)
├── docker-compose.yml              # Redis + LiteLLM proxy + Langfuse + PostgreSQL
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
```

---

## Prerequisites

### System Requirements
- Python 3.11+
- Docker + Docker Compose
- AWS account with Bedrock access (Claude Sonnet 4 enabled in `ap-southeast-1`)

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Copy the template and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS credentials with Bedrock permissions |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_SESSION_TOKEN` | AWS session token (if using temporary credentials) |
| `AWS_REGION_NAME` | AWS region (default: `us-east-1`) |
| `LITELLM_MASTER_KEY` | LiteLLM proxy auth key (any string, e.g. `sk-local-...`) |
| `LITELLM_PROXY_URL` | LiteLLM proxy URL (default: `http://localhost:4000`) |
| `REDIS_URL` | Redis URL (default: `redis://localhost:6379`) |
| `DATABASE_URL` | PostgreSQL URL (default: `postgresql://litellm:litellm_secret@localhost:5433/litellm`) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_HOST` | Langfuse host (default: `http://localhost:3000`) |
| `ORCHESTRATOR_MODEL` | LiteLLM model alias for orchestrator (default: `orchestrator-model`) |
| `SUBAGENT_MODEL` | LiteLLM model alias for sub-agents (default: `subagent-model`) |

---

## How to Run

### Step 1 — Start infrastructure

```bash
docker compose up -d
```

Starts: Redis, LiteLLM proxy (:4000), Langfuse (:3000), PostgreSQL (:5433).

Wait ~30 seconds for all services to be healthy:

```bash
docker compose ps          # all should show "healthy"
docker logs litellm-proxy  # confirm "Application startup complete"
```

### Step 2 — (Optional) Start guardrail service

In a separate terminal:

```bash
python3 guardrail_service.py
```

Runs a local FastAPI server on `:8001` that filters weapons-related content before requests reach the LLM.

### Step 3 — Run the pipeline

**Interactive mode** (recommended for memory demo):
```bash
python3 main.py --user-id YOUR_NAME
```

**Single query:**
```bash
python3 main.py --query "What are the key trends in agentic AI in 2025?" --user-id YOUR_NAME
```

**Cache demo** (runs the same query twice to show cache hit):
```bash
python3 main.py --demo
```

**User memory demo** (3-query scenario: interests → "Who am I?"):
```bash
python3 main.py --user-memory-demo --user-id YOUR_NAME
```

### Step 4 — Run component tests

```bash
# Test guardrails (requires guardrail_service.py running)
python3 test_components.py --test guardrails

# Test researcher memory (requires docker compose up)
python3 test_components.py --test memory

# Test user memory / "Who am I?" scenario
python3 test_components.py --test user-memory --user-id YOUR_NAME

# Run all tests
python3 test_components.py
```

### Useful management commands

```bash
# Clear Redis cache
docker exec -it research-redis redis-cli FLUSHALL

# ── PostgreSQL Memory — Clear ─────────────────────────────────────────────────

# Clear ALL memory (both tables)
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "TRUNCATE researcher_memory, user_memory;"

# Clear only researcher memory (topic findings)
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "TRUNCATE researcher_memory;"

# Clear only user memory (all users)
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "TRUNCATE user_memory;"

# Clear user memory for a specific user (via psql)
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "DELETE FROM user_memory WHERE user_id = 'YOUR_NAME';"

# Clear user memory for a specific user (via CLI)
python3 main.py --clear-user-memory --user-id YOUR_NAME

# ── PostgreSQL Memory — Inspect ───────────────────────────────────────────────

# Inspect researcher memories
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "SELECT session_id, topic, created_at FROM researcher_memory ORDER BY created_at DESC LIMIT 10;"

# Inspect user memory
docker exec -it litellm-postgres psql -U litellm -d litellm \
  -c "SELECT user_id, user_query, created_at FROM user_memory ORDER BY created_at;"

# ── Other ─────────────────────────────────────────────────────────────────────

# Restart LiteLLM (picks up .env changes)
docker compose up -d litellm

# View observability traces
open http://localhost:3000   # Langfuse UI
```

---

## Agent Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Orchestrator Agent                                     │
│  • Checks guardrail (localhost:8001)                    │
│  • Records query to user_memory (PostgreSQL)            │
│  • Detects personal questions → answers from history    │
│  • Otherwise: calls Researcher → Analyst → synthesizes  │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
   ┌───────────────────┐   ┌──────────────────────┐
   │  Researcher Agent │   │   Analyst Agent      │
   │  • search_web     │   │  • No tools          │
   │  • fetch_page     │   │  • Pure reasoning    │
   │  • get_datetime   │   │  • JSON insights     │
   │  • Saves to       │   │  • Confidence score  │
   │    researcher_    │   └──────────────────────┘
   │    memory (PG)    │
   └───────────────────┘
               │
               ▼
        All LLM calls route through
        LiteLLM Proxy → AWS Bedrock
        (cached in Redis, traced in Langfuse)
```

---

## Capabilities

---

### Memory

The system implements **four types of memory** across the agent pipeline.

#### Memory Types Summary

| Type | Status | Implementation |
|---|---|---|
| **In-context (short-term)** | Full | ADK `InMemoryRunner` — each agent sees its own message history within a pipeline run |
| **Session memory** | Partial | ADK session state held in RAM; reused across queries in interactive mode but lost on restart |
| **Semantic (long-term)** | Partial | `researcher_memory` table — keyword-based retrieval (`ILIKE`), no vector embeddings |
| **Episodic (long-term)** | Full | `user_memory` table — every user query recorded with timestamp and user_id |
| **Procedural / User profile** | Partial | Inferred from episodic history on demand ("Who am I?"), not as an explicit stored profile |

#### How Memory Works in the Multi-Agent System

**1. In-Context Memory (per agent, per run)**

Each agent's `InMemoryRunner` maintains the conversation within a single pipeline run. The orchestrator sees its own tool call history; the researcher sees each tool result as it accumulates search results.

**2. Researcher Memory (cross-session semantic store)**

After every research call, the researcher saves findings to PostgreSQL:

```
researcher_memory table:
  session_id   — which pipeline run produced this
  topic        — the researched topic
  key_findings — JSONB list of key facts
  summary      — 2-3 paragraph summary
  created_at   — timestamp
```

On subsequent runs, findings are retrieved by keyword match and **merged into the result dict after the LLM call** (not injected into the prompt). This preserves the exact-match cache key while still enriching the orchestrator's context.

**3. User Memory (cross-session episodic store)**

Every pipeline run records the user's query to PostgreSQL before any LLM call:

```
user_memory table:
  user_id     — stable identifier passed via --user-id
  session_id  — which pipeline run
  user_query  — the exact question asked
  created_at  — timestamp
```

When the user asks a personal question ("Who am I?", "What is my passion?", "What are my interests?"), the orchestrator detects it, retrieves the user's full question history, and synthesizes a profile — without calling the researcher or analyst. Regular research queries never see the history, preserving caching.

**Demo — User Memory Scenario**

```bash
python3 main.py --user-memory-demo --user-id YOUR_NAME
# Query 1: What is the role of a Data Scientist?
# Query 2: What is MLOps and its key practices?
# Query 3: Who am I? What seems to be my passion?
# → Agent synthesizes a profile from Q1 and Q2 history
```

**What is NOT stored**

| Item | Reason |
|---|---|
| Analyst output | Stateless by design — analysis is deterministic given the same research |
| Orchestrator final answer | Visible in Langfuse traces, not in PostgreSQL |
| LLM responses | Stored in Redis by LiteLLM (separate cache, not application memory) |

---

### Caching

Response caching is handled by **LiteLLM proxy + Redis** at the individual LLM API call level.

#### How It Works

Every LLM call from any agent is routed through the LiteLLM proxy. Before forwarding to AWS Bedrock, the proxy computes a cache key from the request fingerprint and checks Redis:

```
Agent (ADK LiteLlm adapter)
    │  POST /v1/chat/completions
    ▼
LiteLLM Proxy
    │
    ├── compute cache_key = hash(model + messages + temperature + tools + ...)
    │
    ├── GET cache_key from Redis
    │       ├── HIT  → return stored response immediately (no Bedrock call)
    │       └── MISS → call AWS Bedrock → SET cache_key in Redis (TTL: 1 hour) → return
    │
    └── response → Agent
```

#### Cache Key

The key is a hash of all request parameters. The `messages` array is the dominant factor:

```python
cache_key = hash({
    "model":    "orchestrator-model",
    "messages": [{role, content}, ...],   # changes = different key
    "temperature": ...,
    "tools":    [{function schemas}],
    ...
})
```

#### When Cache Hits (Scenario: different sessions, same query)

```
Run 1 — Orchestrator call:
  messages: [system_prompt, {user: "What are trends in agentic AI?"}]
  → MISS → Bedrock → stored as hash H1 in Redis

Run 2 — Orchestrator call:
  messages: [system_prompt, {user: "What are trends in agentic AI?"}]
  → same H1 → HIT → instant response, no Bedrock call
```

Because the cached response contains the same tool call instruction, the researcher and analyst calls that follow also produce identical messages → cascade of cache hits across the full pipeline.

#### When Cache Cannot Hit

| Scenario | Reason |
|---|---|
| Same session, second query | ADK accumulates conversation history in the session — messages array grows → different hash |
| After memory injection in prompt | Adding past memories to the LLM message changes the hash |

#### What We Did to Preserve Caching

| Problem | Fix |
|---|---|
| Session ID embedded in user message | Removed — session ID flows via Python `contextvar`, never touches the LLM message |
| Researcher memory injected into prompt | Moved to post-LLM merge — memories enrich the result dict, not the API call |
| Non-deterministic tool outputs | Hardcoded `search_web`, `fetch_page_content`, `get_current_datetime` return values |
| User history injected into every message | Only injected when query is detected as personal (`who am i`, `my passion`, etc.) |

#### Verify Cache Hits

```bash
# Watch Redis live
docker exec -it research-redis redis-cli MONITOR
# Run 1: SET litellm:<hash> <response>
# Run 2: GET litellm:<hash>  ← cache hit

# Check LiteLLM logs
docker logs litellm-proxy --follow | grep -i cache
```

#### Cache Configuration (`litellm_config.yaml`)

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: redis        # Docker service name
    port: 6379
    ttl: 3600          # 1 hour expiry
    supported_call_types:
      - acompletion    # async (used by ADK)
      - completion
```

---

### Guardrails

A local FastAPI service (`guardrail_service.py`) runs on `:8001` and acts as a pre-call content filter. The orchestrator calls it directly before each pipeline run — blocking weapons-related content before any LLM call is made.

```bash
# Start guardrail service
python3 guardrail_service.py

# Test blocking
python3 test_components.py --test guardrails
```

The guardrail raises a `ValueError` (blocking the pipeline) for harmful content, or logs "allowed" and continues for safe queries.

---

### Observability

All LLM calls are traced in **Langfuse** automatically via LiteLLM's callback integration.

```bash
open http://localhost:3000   # Langfuse UI (self-hosted)
```

Each pipeline run creates a top-level span (`research_pipeline`) containing child spans for every researcher and analyst LLM call, with token usage and latency.
