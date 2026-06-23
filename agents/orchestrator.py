# agents/orchestrator.py
"""
Orchestrator Agent — Google ADK LlmAgent

This is the root agent. It:
1. Receives a user query
2. Decomposes it into sub-tasks
3. Calls Researcher and Analyst as sub-agents (AgentTool pattern)
4. Synthesizes their outputs into a final, human-readable answer

Architecture:
    User → Orchestrator → [Researcher, Analyst] → Orchestrator → Final Answer
"""

import contextvars
import json
import logging
import uuid
import httpx
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types
from langfuse import Langfuse

from agents.researcher import run_researcher
from agents.analyst import run_analyst
from config.settings import settings
from config.memory_store import user_memory_store

logger = logging.getLogger(__name__)

# Carries session_id through the call stack without embedding it in LLM messages
_session_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="default")

langfuse = Langfuse(
    secret_key=settings.LANGFUSE_SECRET_KEY,
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    host=settings.LANGFUSE_HOST,
)

_GUARDRAIL_URL = "http://localhost:8001/guardrail/check"

# Keywords that signal the user is asking about themselves rather than a research topic.
# When matched, user history is injected into the message so the orchestrator can answer
# from memory. For all other queries the plain user_query is sent, preserving cache hits.
_PERSONAL_KEYWORDS = {
    "who am i", "who am i?",
    "what am i", "about me",
    "my passion", "my interest", "my interests",
    "my background", "my career", "my focus",
    "what do i do", "what is my",
    "my expertise", "my skills",
}


def _is_personal_question(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _PERSONAL_KEYWORDS)


async def _check_guardrail(text: str) -> None:
    """
    Call the local guardrail service before sending any query to LiteLLM.
    Raises ValueError if the content is blocked.
    If the guardrail service is not running, logs a warning and continues.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                _GUARDRAIL_URL,
                json={"input": {"messages": [{"role": "user", "content": text}]}},
            )
        if resp.status_code == 400:
            detail = resp.json().get("detail", {})
            raise ValueError(f"Guardrail blocked request: {detail.get('message', 'Content blocked')}")
        logger.info("[Guardrail] Content allowed")
    except httpx.ConnectError:
        logger.warning("[Guardrail] Service not reachable at localhost:8001 — skipping check")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-agent tools — wrapped as plain Python functions so ADK can call them
# The Orchestrator calls these exactly like any other tool
# ─────────────────────────────────────────────────────────────────────────────
async def delegate_to_researcher(topic: str) -> str:
    """
    Delegate a research task to the Researcher Agent.

    Use this tool when you need to gather factual information, web search results,
    or raw data about a specific topic before performing analysis.

    Args:
        topic: The specific topic or question to research.

    Returns:
        A JSON string containing research findings, sources, and a summary.
    """
    research_data = await run_researcher(topic=topic, session_id=_session_id_ctx.get())
    return json.dumps(research_data)


async def delegate_to_analyst(research_json: str) -> str:
    """
    Delegate analysis work to the Analyst Agent.

    Use this tool after you have gathered research data. Provide the raw research
    JSON and the analyst will return structured insights, key themes, risks,
    opportunities, and a confidence score.

    Args:
        research_json: JSON string output from the delegate_to_researcher tool.

    Returns:
        A JSON string with analysis summary, themes, risks, opportunities,
        and a confidence score between 0.0 and 1.0.
    """
    try:
        research_data = json.loads(research_json)
    except json.JSONDecodeError:
        research_data = {"raw_summary": research_json, "topic": "unknown"}

    analysis = await run_analyst(research_data=research_data, session_id=_session_id_ctx.get())
    return json.dumps(analysis)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_PROMPT = """You are a Smart Research Orchestrator.
You coordinate a team of specialized agents to answer user research queries comprehensively.
You also have access to the user's conversation history so you can answer personal questions.

YOUR TEAM:
- delegate_to_researcher: Gathers raw facts, web search results, and source data
- delegate_to_analyst: Produces structured insights, themes, risks, and confidence scores

WORKFLOW:

PERSONAL QUESTIONS (e.g. "who am I?", "what are my interests?", "what is my passion?"):
  → Answer DIRECTLY from the USER HISTORY provided at the end of the message.
  → Do NOT call delegate_to_researcher or delegate_to_analyst.
  → Synthesize a profile of the user based on the topics they have asked about.

ALL OTHER QUERIES — follow this order:
1. DECOMPOSE: Break the user query into a clear research topic
2. RESEARCH: Call delegate_to_researcher with the topic
3. ANALYZE: Pass the research JSON to delegate_to_analyst
4. SYNTHESIZE: Combine both outputs into a clear, well-structured final answer.
   If the research JSON contains a "past_context" field, incorporate those prior
   findings into the Executive Summary as additional supporting evidence.

FINAL ANSWER FORMAT:
# Research Report: <Topic>

## Executive Summary
<1-2 paragraph synthesis of research + analysis>

## Key Findings
<bullet list from research key_findings>

## Insights & Themes
<bullet list from analysis key_themes>

## Opportunities
<from analysis>

## Risks & Challenges  
<from analysis>

## Recommended Next Steps
<from analysis>

## Confidence Level
<confidence_score as percentage> — <confidence_rationale>

## Sources
<list of sources from research>

---
Always be factual, objective, and cite the confidence level clearly."""


# ─────────────────────────────────────────────────────────────────────────────
# ADK Agent definition
# ─────────────────────────────────────────────────────────────────────────────
def create_orchestrator_agent() -> Agent:
    """Build and return the Orchestrator ADK LlmAgent."""
    model = LiteLlm(
        model=f"openai/{settings.ORCHESTRATOR_MODEL}",
        api_base=settings.LITELLM_PROXY_URL,
        api_key=settings.LITELLM_MASTER_KEY,
    )

    agent = Agent(
        name="orchestrator_agent",
        description="Coordinates researcher and analyst sub-agents to answer complex research queries.",
        model=model,
        instruction=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=[
            FunctionTool(func=delegate_to_researcher),
            FunctionTool(func=delegate_to_analyst),
        ],
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — call this from main.py
# ─────────────────────────────────────────────────────────────────────────────
_session_service = InMemorySessionService()
_orchestrator_agent = create_orchestrator_agent()
_runner = InMemoryRunner(agent=_orchestrator_agent, app_name="orchestrator")


async def run_research_pipeline(
    user_query: str,
    session_id: str | None = None,
    user_id: str = "default-user",
) -> str:
    """
    Entry point: run the full multi-agent research pipeline.

    Args:
        user_query: The user's research question
        session_id: Optional session ID (generated if not provided)
        user_id:    Stable identifier for the user across sessions (enables
                    memory recall for "who am I?" style questions)

    Returns:
        The orchestrator's final synthesized answer as a markdown string
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # Make session_id available to delegate tools without embedding it in the LLM message
    _session_id_ctx.set(session_id)

    # ── Guardrail check (application-level, version-independent) ─────────────
    await _check_guardrail(user_query)

    # ── Record this query in user memory ─────────────────────────────────────
    await user_memory_store.record(user_id=user_id, session_id=session_id, user_query=user_query)

    # ── Inject user history only for personal questions ──────────────────────
    # Regular research queries stay as plain text so LiteLLM's exact-match
    # Redis cache can still produce hits. History injection changes the message
    # hash and would break caching for every repeated research query.
    user_history_context = ""
    if _is_personal_question(user_query):
        history = await user_memory_store.get_history(user_id=user_id)
        if len(history) > 1:
            lines = [
                f"  [{i+1}] \"{h['user_query']}\""
                for i, h in enumerate(history[:-1])  # exclude the current query
            ]
            user_history_context = (
                "\n\nUSER HISTORY (all past questions from this user, chronological):\n"
                + "\n".join(lines)
            )

    # Create a Langfuse trace observation for the entire pipeline run (v4 API)
    trace_obs = langfuse.start_observation(
        name="research_pipeline",
        as_type="span",
        input=user_query,
        metadata={
            "pipeline": "researcher → analyst → orchestrator",
            "session_id": session_id,
            "user_id": user_id,
        },
    )

    logger.info(f"[Orchestrator] Pipeline start | query='{user_query}' | session={session_id}")

    #----------------------
    runner_session_service = _runner.session_service
    try:
        await runner_session_service.create_session(
            app_name="orchestrator",
            user_id=user_id,
            session_id=session_id
        )
    except:
        runner_session_service.get_session(
            app_name="orchestrator",
            user_id=user_id,
            session_id=session_id,
        )
    #----------------------

    # Append user history to the message so the orchestrator can answer
    # personal questions ("who am I?") without calling sub-agents
    message_text = user_query + user_history_context

    final_answer = ""
    async for event in _runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message_text)]
        ),
    ):
        if event.is_final_response():
            final_answer = event.content.parts[0].text

    # Close Langfuse trace observation
    trace_obs.update(output=final_answer)
    trace_obs.end()
    langfuse.flush()

    logger.info(f"[Orchestrator] Pipeline complete | session={session_id}")
    return final_answer
