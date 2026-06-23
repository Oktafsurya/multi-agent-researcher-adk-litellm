# agents/researcher.py
"""
Researcher Agent — Google ADK LlmAgent

Responsibilities:
- Receives a research topic from the Orchestrator
- Uses search_web + fetch_page_content tools to gather raw facts
- Returns a structured research report as a string
"""

import json
import logging


def _extract_json(text: str) -> dict | None:
    """
    Extract the first complete JSON object from an LLM response.
    Handles: plain JSON, ```json fences, extra prose before/after.
    Returns None if no valid JSON object is found.
    """
    # Strip markdown code fences first
    stripped = text.strip()
    if stripped.startswith("```"):
        inner = stripped.split("```")
        for block in inner:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                stripped = block
                break

    # Find outermost { ... } by tracking brace depth
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(stripped[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

from tools.research_tools import search_web, fetch_page_content, get_current_datetime
from config.settings import settings
from config.memory_store import memory_store

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
RESEARCHER_SYSTEM_PROMPT = """You are a Researcher Agent.
Your job is to gather accurate, up-to-date information on the topic you receive.

WORKFLOW:
1. Use search_web to find relevant sources for the topic
2. Use fetch_page_content on the most relevant URL(s) to get detailed content
3. Use get_current_datetime to timestamp your research
4. Synthesize all gathered information into a structured report

OUTPUT FORMAT — respond with ONLY a raw JSON object, no markdown fences, no extra text:
{
  "topic": "<the topic you researched>",
  "researched_at": "<ISO datetime>",
  "key_findings": [
    "<finding 1>",
    "<finding 2>",
    "<finding 3>"
  ],
  "sources": [
    {"title": "...", "url": "...", "relevance": "high|medium|low"}
  ],
  "raw_summary": "<2-3 paragraph summary of everything you found>"
}

Be factual. Do not hallucinate. Only include information from your tool results."""


# ─────────────────────────────────────────────────────────────────────────────
# ADK Agent definition
# ─────────────────────────────────────────────────────────────────────────────
def create_researcher_agent() -> Agent:
    """
    Build and return the Researcher ADK LlmAgent.

    The agent uses LiteLlm adapter to route through our local LiteLLM proxy,
    which handles Bedrock routing, caching, guardrails, and Langfuse logging.
    """
    # LiteLlm adapter: points to our proxy URL with the subagent model alias
    model = LiteLlm(
        model=f"openai/{settings.SUBAGENT_MODEL}",   # openai/ prefix = OpenAI-compat endpoint
        api_base=settings.LITELLM_PROXY_URL,
        api_key=settings.LITELLM_MASTER_KEY,
    )

    agent = Agent(
        name="researcher_agent",
        description="Searches the web and gathers factual information on a given topic.",
        model=model,
        instruction=RESEARCHER_SYSTEM_PROMPT,
        tools=[
            FunctionTool(func=search_web),
            FunctionTool(func=fetch_page_content),
            FunctionTool(func=get_current_datetime),
        ],
        # ADK generates JSON schema for tools automatically from type annotations
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner — used by the Orchestrator via run_researcher()
# ─────────────────────────────────────────────────────────────────────────────
_session_service = InMemorySessionService()
_researcher_agent = create_researcher_agent()
_runner = InMemoryRunner(agent=_researcher_agent, app_name="researcher")


async def run_researcher(topic: str, session_id: str) -> dict:
    """
    Run the Researcher agent on a topic and return structured research data.

    Args:
        topic:      The research topic string
        session_id: Session ID for memory continuity

    Returns:
        Parsed dict with research findings (matches RESEARCHER output format above)
    """
    logger.info(f"[Researcher] Starting research on: '{topic}'")

    # ── Retrieve past memories — merged into result AFTER the LLM call ───────
    # NOT injected into the message so the LiteLLM cache key stays stable.
    # The same topic always produces the same cache key regardless of how many
    # past memories have accumulated, enabling both exact-match and semantic hits.
    past_memories = await memory_store.retrieve(topic)
    if past_memories:
        logger.info(f"[Researcher] Found {len(past_memories)} past memories — will merge post-LLM")

    #----------------------
    runner_session_service = _runner.session_service
    try:
        await runner_session_service.create_session(
            app_name="researcher",
            user_id="system",
            session_id=session_id
        )
    except:
        runner_session_service.get_session(
            app_name="researcher",
            user_id="system",
            session_id=session_id,
        )

    # Run the agent — clean message, no memory injected → stable cache key
    final_response_text = ""
    async for event in _runner.run_async(
        user_id="system",
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"Research this topic thoroughly: {topic}")]
        ),
    ):
        if event.is_final_response():
            final_response_text = event.content.parts[0].text

    logger.info(f"[Researcher] Completed research for: '{topic}'")

    # Parse JSON response — robust extraction handles text/fences before or after JSON
    result = _extract_json(final_response_text)
    if result is None:
        logger.warning("[Researcher] Could not parse JSON response, returning raw text")
        result = {
            "topic": topic,
            "key_findings": ["See raw_summary for details"],
            "sources": [],
            "raw_summary": final_response_text[:2000],  # cap so orchestrator LLM isn't overwhelmed
        }

    # ── Merge past memories into result (Python layer, not LLM layer) ────────
    if past_memories:
        result["past_context"] = [
            {"topic": m["topic"], "summary": m["summary"]}
            for m in past_memories
        ]
        logger.info(f"[Researcher] Merged {len(past_memories)} past memories into result")

    # ── Persist findings to memory for future sessions ────────────────────────
    await memory_store.save(
        session_id=session_id,
        topic=result.get("topic", topic),
        key_findings=result.get("key_findings", []),
        summary=result.get("raw_summary", ""),
    )

    return result
