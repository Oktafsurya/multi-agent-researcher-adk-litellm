#!/usr/bin/env python3
# guardrail_service.py
"""
Local guardrail service — mimics litellm/cf_harmful_illegal_weapons.

Implements LiteLLM's custom_callback_api contract:
  POST /guardrail/check
    Body: { "input": { "messages": [...], "model": "..." }, "call_type": "..." }
  Response:
    200 → request is ALLOWED
    400 → request is BLOCKED (body contains reason)

Usage:
    python3 guardrail_service.py

Swap to production guardrail by changing litellm_config.yaml:
    guardrail: litellm/cf_harmful_illegal_weapons
    (and remove guardrail_endpoint)
"""

import logging
from fastapi import FastAPI, HTTPException, Request
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Weapons Guardrail", version="1.0.0")

# Keywords that trigger a block — mirrors Cloudflare's cf_harmful_illegal_weapons categories
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    ("bomb", "explosive device"),
    ("explosive", "explosive material"),
    ("grenade", "explosive device"),
    ("make a weapon", "weapon manufacturing"),
    ("build a weapon", "weapon manufacturing"),
    ("illegal weapon", "illegal weapon"),
    ("untraceable gun", "illegal weapon"),
    ("homemade gun", "illegal weapon"),
    ("3d print gun", "illegal weapon"),
    ("ghost gun", "illegal weapon"),
    ("pipe bomb", "explosive device"),
    ("ied", "improvised explosive device"),
    ("c4 explosive", "explosive material"),
    ("dynamite", "explosive material"),
    ("how to make ammo", "illegal weapon manufacturing"),
    ("illegal firearm", "illegal weapon"),
]


def _scan(text: str) -> tuple[bool, str]:
    """Return (blocked, reason). Case-insensitive keyword scan."""
    lower = text.lower()
    for keyword, category in _BLOCKED_PATTERNS:
        if keyword in lower:
            return True, f"Content flagged as '{category}' (matched: '{keyword}')"
    return False, ""


@app.post("/guardrail/check")
async def check_guardrail(request: Request):
    body = await request.json()

    # Extract messages from LiteLLM's payload
    messages: list[dict] = body.get("input", {}).get("messages", [])

    for msg in messages:
        content = str(msg.get("content", ""))
        blocked, reason = _scan(content)
        if blocked:
            logger.warning(f"[Guardrail] BLOCKED — {reason}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "GuardrailViolation",
                    "guardrail": "weapons-check",
                    "message": f"Request blocked: {reason}",
                },
            )

    logger.info("[Guardrail] ALLOWED")
    return {"decision": "allow", "message": "Content passed weapons guardrail"}


@app.get("/health")
async def health():
    return {"status": "ok", "guardrail": "weapons-check"}


if __name__ == "__main__":
    print("Starting local guardrail service on http://0.0.0.0:8001")
    print("LiteLLM will call: http://host.docker.internal:8001/guardrail/check")
    print("Press Ctrl+C to stop.\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
