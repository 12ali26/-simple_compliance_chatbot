from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

import requests

from src.model_service import get_openrouter_settings


class Intent(str, Enum):
    CASUAL = "casual"
    SOP_QUESTION = "sop_question"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    response: str
    search_query: str
    reason: str


CASUAL_RESPONSES = {
    "hello": "Hello. Ask me anything about the approved cleaning procedures.",
    "hi": "Hi. Ask me anything about the approved cleaning procedures.",
    "hey": "Hi. Ask me anything about the approved cleaning procedures.",
    "thanks": "You are welcome.",
    "thank you": "You are welcome.",
    "ok": "Okay.",
}

SOP_TERMS = {
    "blood",
    "body",
    "chemical",
    "clean",
    "cleaning",
    "disinfect",
    "floor",
    "glove",
    "gloves",
    "isolation",
    "mop",
    "ppe",
    "spill",
    "toilet",
    "washroom",
    "waste",
}

UNSUPPORTED_TERMS = {
    "attendance",
    "bonus",
    "contract",
    "doctor",
    "hr",
    "lawyer",
    "leave",
    "legal",
    "medical",
    "pay",
    "payroll",
    "salary",
    "sick",
    "vacation",
}


def normalize_message(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def keyword_route(message: str) -> IntentResult:
    normalized = normalize_message(message)
    if not normalized:
        return IntentResult(Intent.CASUAL, "Ask me a cleaning procedure question when you are ready.", "", "Empty message.")

    if normalized in CASUAL_RESPONSES:
        return IntentResult(Intent.CASUAL, CASUAL_RESPONSES[normalized], "", "Matched casual phrase.")

    words = set(re.findall(r"[a-z0-9]+", normalized))
    if words & UNSUPPORTED_TERMS:
        return IntentResult(
            Intent.UNSUPPORTED,
            "I can only help with approved cleaning procedures. Please ask your supervisor about that.",
            "",
            "Matched unsupported topic.",
        )

    if words & SOP_TERMS or "how do i" in normalized or "what should" in normalized or "what ppe" in normalized:
        return IntentResult(Intent.SOP_QUESTION, "", message.strip(), "Matched SOP terms.")

    if len(words) <= 3:
        return IntentResult(Intent.CASUAL, "I am here for cleaning procedure questions when you need me.", "", "Short non-SOP message.")

    return IntentResult(Intent.SOP_QUESTION, "", message.strip(), "Defaulted to SOP question for retrieval.")


def model_route(message: str) -> IntentResult | None:
    settings = get_openrouter_settings()
    if settings is None:
        return None

    system_prompt = (
        "Classify the user's message for a cleaning SOP assistant. "
        "Return only JSON with keys: intent, response, search_query, reason. "
        "intent must be one of casual, sop_question, unsupported. "
        "casual means greetings, thanks, or simple conversation. "
        "sop_question means cleaning, PPE, waste, spills, chemicals, room cleaning, washroom, floors, or safety procedure questions. "
        "unsupported means HR, payroll, attendance, medical, legal, or non-cleaning requests. "
        "For sop_question, rewrite search_query as a concise procedure search query. "
        "For casual, response should be a short friendly reply. "
        "For unsupported, response should say the assistant only helps with approved cleaning procedures and to ask a supervisor."
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": 0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    data = json.loads(content)
    intent = Intent(data.get("intent", Intent.SOP_QUESTION))
    return IntentResult(
        intent=intent,
        response=str(data.get("response", "")),
        search_query=str(data.get("search_query", "") or message),
        reason=str(data.get("reason", "Model routed message.")),
    )


def route_message(message: str) -> IntentResult:
    try:
        model_result = model_route(message)
    except Exception:
        model_result = None
    return model_result or keyword_route(message)
