"""
ai/openrouter_client.py

Async OpenRouter API client for cloud-assisted reasoning.

Responsibilities:
  • Generate treatment recommendations for detected diseases
  • Produce structured JSON reports
  • Graceful offline fallback – the rover keeps working without internet

The AI model used here should be SMALL (e.g. Mistral-7B-Instruct) to
keep latency low and cost minimal.  The edge TFLite model does the heavy
lifting; OpenRouter only adds human-readable explanation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import API_CFG
from ai.classify import PredictionResult

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_TREATMENT_PROMPT = """\
You are an agricultural AI assistant integrated into an autonomous rover.
A TFLite model has detected the following plant disease:

Disease   : {disease}
Confidence: {confidence:.0%}
Severity  : {severity}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "summary": "<one-sentence description of the disease>",
  "immediate_actions": ["<action 1>", "<action 2>"],
  "treatments": ["<treatment 1>", "<treatment 2>"],
  "prevention": "<prevention tips>",
  "urgency": "low|medium|high"
}}
"""

_REPORT_PROMPT = """\
Summarise the following agricultural inspection session in a short paragraph:
{session_json}
Focus on: diseases found, severity, recommended priorities.
Reply in plain text only, max 120 words.
"""


# ── Offline fallback templates ────────────────────────────────────────────────

def _offline_treatment(result: PredictionResult) -> Dict[str, Any]:
    return {
        "summary":           f"Offline mode – {result.disease} detected at "
                             f"{result.confidence:.0%} confidence.",
        "immediate_actions": ["Isolate affected plants if possible."],
        "treatments":        ["Consult a local agronomist."],
        "prevention":        "Ensure good air circulation and avoid overhead watering.",
        "urgency":           "medium" if result.severity != "high" else "high",
        "offline":           True,
    }


# ── Client ────────────────────────────────────────────────────────────────────

class OpenRouterClient:
    """
    Thin async client around the OpenRouter /v1/chat/completions endpoint.

    Usage::

        client = OpenRouterClient()
        advice = await client.get_treatment_advice(prediction_result)
    """

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {API_CFG.API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://agri-rover",
            "X-Title":       "AgriRover",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _chat(self, prompt: str) -> Optional[str]:
        """
        Send a prompt and return the assistant text content.
        Returns None on network/API error.
        """
        payload = {
            "model":      API_CFG.MODEL,
            "max_tokens": API_CFG.MAX_TOKENS,
            "messages":   [{"role": "user", "content": prompt}],
        }

        try:
            async for attempt in AsyncRetrying(
                retry   = retry_if_exception_type(httpx.TransportError),
                stop    = stop_after_attempt(2),
                wait    = wait_exponential(min=1, max=4),
                reraise = False,
            ):
                with attempt:
                    async with httpx.AsyncClient(
                        timeout=API_CFG.TIMEOUT
                    ) as client:
                        resp = await client.post(
                            API_CFG.BASE_URL,
                            headers=self._headers,
                            json=payload,
                        )
                        resp.raise_for_status()
                        data    = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return content.strip()

        except Exception as exc:
            logger.warning("OpenRouter request failed: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_treatment_advice(
        self, result: PredictionResult
    ) -> Dict[str, Any]:
        """
        Query OpenRouter for disease explanation and treatment suggestions.

        Always returns a dict – uses offline fallback if API unavailable.
        """
        if not API_CFG.API_KEY:
            logger.info("No API key – using offline advice.")
            return _offline_treatment(result)

        if result.is_healthy:
            return {
                "summary":           "Plant appears healthy.",
                "immediate_actions": [],
                "treatments":        [],
                "prevention":        "Continue regular monitoring.",
                "urgency":           "low",
            }

        prompt = _TREATMENT_PROMPT.format(
            disease    = result.disease,
            confidence = result.confidence,
            severity   = result.severity,
        )

        raw = await self._chat(prompt)
        if raw is None:
            return _offline_treatment(result)

        try:
            # Strip possible markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()
            advice = json.loads(raw)
            advice["offline"] = False
            return advice
        except json.JSONDecodeError:
            logger.warning("OpenRouter returned non-JSON: %s", raw[:120])
            return _offline_treatment(result)

    async def generate_session_report(
        self, session_data: list[dict]
    ) -> str:
        """
        Summarise an inspection session (list of detection records).

        Returns a plain-text paragraph.
        """
        if not session_data:
            return "No detections recorded this session."

        prompt = _REPORT_PROMPT.format(
            session_json=json.dumps(session_data, indent=2)
        )
        raw = await self._chat(prompt)
        return raw if raw else "Report generation unavailable (offline)."
