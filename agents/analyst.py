# agents/analyst.py
"""
Analyst Agent — Google ADK LlmAgent

Responsibilities:
- Receives raw research data from the Orchestrator
- Produces structured insights: key themes, risks, opportunities, confidence score
- Does NOT use tools — pure reasoning over input data
"""

import json
import logging


def _extract_json(text: str) -> dict | None:
    """Extract first complete JSON object from LLM response text."""
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
from google.genai import types as genai_types

from config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
ANALYST_SYSTEM_PROMPT = """You are an Analyst Agent. You receive raw research data and 
produce structured analytical insights.

Your analysis should be objective, concise, and actionable. 
Do not search for additional information — work only with the data provided.

OUTPUT FORMAT — respond with ONLY a raw JSON object, no markdown fences, no extra text:
{
  "analysis_summary": "<1 paragraph executive summary>",
  "key_themes": [
    "<theme 1>",
    "<theme 2>",
    "<theme 3>"
  ],
  "opportunities": [
    "<opportunity 1>",
    "<opportunity 2>"
  ],
  "risks_or_challenges": [
    "<risk 1>",
    "<risk 2>"
  ],
  "confidence_score": <float 0.0 to 1.0>,
  "confidence_rationale": "<why you gave that confidence score>",
  "recommended_next_steps": [
    "<step 1>",
    "<step 2>"
  ]
}

Confidence score guidance:
- 0.9+ : Multiple high-quality sources, consistent findings, recent data
- 0.7-0.9: Good sources but some gaps or minor inconsistencies
- 0.5-0.7: Limited sources, older data, or conflicting information
- below 0.5: Very limited data — low confidence, flag for further research"""


# ─────────────────────────────────────────────────────────────────────────────
# ADK Agent definition
# ─────────────────────────────────────────────────────────────────────────────
def create_analyst_agent() -> LlmAgent:
    """Build and return the Analyst ADK LlmAgent."""
    model = LiteLlm(
        model=f"openai/{settings.SUBAGENT_MODEL}",
        api_base=settings.LITELLM_PROXY_URL,
        api_key=settings.LITELLM_MASTER_KEY,
    )

    agent = Agent(
        name="analyst_agent",
        description="Analyzes research data and produces structured insights with confidence scoring.",
        model=model,
        instruction=ANALYST_SYSTEM_PROMPT,
        # No tools — analyst reasons over provided data only
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner — used by the Orchestrator via run_analyst()
# ─────────────────────────────────────────────────────────────────────────────
_session_service = InMemorySessionService()
_analyst_agent = create_analyst_agent()
_runner = InMemoryRunner(agent=_analyst_agent, app_name="analyst")


async def run_analyst(research_data: dict, session_id: str) -> dict:
    """
    Run the Analyst agent on research data and return structured insights.

    Args:
        research_data: Dict output from run_researcher()
        session_id:    Session ID for memory continuity

    Returns:
        Parsed dict with analyst insights
    """
    logger.info(f"[Analyst] Analyzing research on: '{research_data.get('topic', 'unknown')}'")

    #----------------------
    runner_session_service = _runner.session_service
    try:
        await runner_session_service.create_session(
            app_name="analyst",
            user_id="system",
            session_id=session_id
        )
    except:
        runner_session_service.get_session(
            app_name="analyst",
            user_id="system",
            session_id=session_id,
        )

    # Serialize research data as the user message
    research_json = json.dumps(research_data, indent=2)
    prompt = f"""Analyze the following research data and produce structured insights:

{research_json}

Provide your analysis in the exact JSON format specified in your instructions."""

    final_response_text = ""
    async for event in _runner.run_async(
        user_id="system",
        session_id=session_id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)]
        ),
    ):
        if event.is_final_response():
            final_response_text = event.content.parts[0].text

    logger.info(f"[Analyst] Completed analysis for: '{research_data.get('topic', 'unknown')}'")

    # Parse JSON response — robust extraction handles fences and prose around JSON
    result = _extract_json(final_response_text)
    if result is not None:
        return result
    logger.warning("[Analyst] Could not parse JSON response, returning raw text")
    return {
        "analysis_summary": final_response_text,
        "key_themes": [],
        "opportunities": [],
        "risks_or_challenges": [],
        "confidence_score": 0.0,
        "confidence_rationale": "Could not parse structured response",
        "recommended_next_steps": [],
    }
