"""
Gate Agent — LangChain Agent + OpenBox governance layer.

Mirrors the n8n nodes:
  OpenBox: Agent (CUSTOM.openBoxAgent) — ICP relevance gate
  Claude Haiku 4.5 via OpenRouter (Claude Sonnet 4 Gate in workflow)
  Signal Extractor (langchain agent, DeepSeek)
  Score Top 5 (chainLlm, DeepSeek)
  Combine & Date-Filter Pool
  Top 5 → Brief Emitter

The OpenBox middleware wraps every agent call, enforcing governance policies,
recording audit spans, and enabling HITL if configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from openbox_langchain import create_openbox_langchain_middleware, initialize

_logger = logging.getLogger("content_studio.gate_agent")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

ICP_GATE_SYSTEM = """You are a signal extraction agent for OpenBox AI — an enterprise AI trust and governance platform that provides runtime monitoring of AI agents, policy enforcement via OPA/Rego, cryptographic attestation (Merkle audit trail), human-in-the-loop (HITL) approvals, and a Trust Lifecycle (Assess → Authorize → Monitor → Verify → Adapt).

IDEAL CUSTOMER PROFILE:
- CISOs at mid-market and enterprise companies deploying AI agents in production who need audit trails and policy enforcement
- AI/ML Platform Leads managing agent infrastructure who need runtime observability and behavioral guardrails
- Compliance Officers navigating AI regulation (EU AI Act, NIST AI RMF, ISO 42001, FINRA, SOC 2) who need cryptographically verifiable evidence of AI decisions

OPENBOX PRODUCT SURFACE:
- Trust Score: real-time trustworthiness rating per agent action
- Behavioral Rules: OPA/Rego policies defining what agents can/cannot do
- Merkle Audit Trail: tamper-proof cryptographic log of every agent decision
- HITL: approval gates that pause agent execution for human review
- Session Replay: full playback of an agent decision sequence
- Trust Lifecycle: Assess → Authorize → Monitor → Verify → Adapt

COMPETITORS:
- Langfuse → LLM observability and tracing only, no governance or attestation
- Arize → ML model monitoring, no policy enforcement
- WhyLabs → data quality monitoring, no agent-level controls
OpenBox differentiator: the ONLY platform combining governance + cryptographic attestation + runtime policy enforcement in one layer.

CORE THEMES TO EXTRACT:
- Agentic AI risk, autonomous agent security, agent policy enforcement
- LLM security, prompt injection in enterprise
- AI governance frameworks, responsible AI, trustworthy AI
- Regulatory signals: EU AI Act, NIST AI RMF, ISO 42001, FINRA, SEC AI rules, SOC 2 for AI, HIPAA + AI
- Model risk management, AI audit requirements, AI compliance
- Human-in-the-loop, human oversight of AI

TASK: Review these feed items and return ONLY those that are ICP-relevant. Filter out consumer AI news, pure research papers without enterprise angle, and non-governance topics.

Return ONLY valid JSON — no markdown fences:
{"filtered": [{"title": "...", "url": "...", "published": "...", "source": "..."}]}"""

SIGNAL_EXTRACTOR_SYSTEM = """You are a signal extraction agent for OpenBox AI — an enterprise AI trust and governance platform.

Note: these items have ALREADY passed the OpenBox governance gate, so they are all ICP-relevant. Your job is to extract structured signal from them and filter out any remaining noise.

IDEAL CUSTOMER PROFILE:
- CISOs at mid-market and enterprise companies deploying AI agents in production
- AI/ML Platform Leads managing agent infrastructure
- Compliance Officers navigating AI regulation (EU AI Act, NIST AI RMF, ISO 42001, FINRA, SOC 2)

OPENBOX PRODUCT SURFACE:
- Trust Score, Behavioral Rules (OPA/Rego), Merkle Audit Trail, HITL, Session Replay, Trust Lifecycle

FOR EACH ITEM:
1. Confirm it's relevant (already gate-filtered, but reject pure noise)
2. Extract: title, url, published, source, summary (1 sentence), relevance_hint (max 10 words — why this matters for OpenBox)

Return ONLY valid JSON. No markdown fences:
{"extracted": [{"title": "...", "url": "...", "published": "...", "source": "...", "summary": "...", "relevance_hint": "..."}]}"""

SCORER_SYSTEM = """You are the OpenBox AI content scoring and brief agent.

OpenBox AI is an enterprise AI trust and governance platform.
Core: runtime monitoring of AI agents, OPA/Rego policy enforcement, cryptographic attestation (Merkle audit trail), HITL approvals, Trust Lifecycle (Assess→Authorize→Monitor→Verify→Adapt), EU AI Act, FINRA, SOC 2 compliance, LLMOps governance.
ICP: CISOs, AI Platform Leads, Compliance Officers at enterprises deploying AI agents.
Competitors: Langfuse (observability only), Arize (ML monitoring), WhyLabs (data quality). OpenBox differentiator: governance + attestation, not just logging.
OpenBox product features: Trust Score, Behavioral Rules (OPA/Rego), Merkle Audit Trail, HITL, Session Replay, Trust Lifecycle.

PHASE 1 — SCORE every item:
- relevance_score (1-10): How directly related to AI governance, agentic AI risk, LLM security, enterprise AI trust?
- urgency_score (1-10): Is this trending, breaking, emotionally charged for a CISO/Compliance audience?
- gap_score (1-10): Can OpenBox say something Langfuse/Arize/WhyLabs cannot? Higher = bigger gap.
- freshness_score (1-10): Items from the last few hours = 10. Items from yesterday = 7-8. Items from 3-5 days ago = 4-6. Items from 5-7 days ago = 2-3.
- composite_score = (relevance * 0.35) + (urgency * 0.30) + (gap * 0.20) + (freshness * 0.15)

PHASE 2 — Pick the TOP 5 by composite_score and generate a full content brief for each:
- rank (1-5)
- title (original)
- url, source, published, age_label
- relevance_score, urgency_score, gap_score, freshness_score, composite_score
- persona: CISO or AI Platform Lead or Compliance Officer
- content_type: thought_leadership or how_to or comparison or data_driven or news_commentary
- why_openbox (max 20 words): why this matters specifically for OpenBox's ICP
- headline: compelling blog/LinkedIn headline (max 70 chars)
- hook: opening sentence that stops scrolling (max 25 words)
- key_argument: the core insight (1-2 sentences)
- three_subheadings: array of 3 section headers
- openbox_product_tie: which OpenBox feature is most relevant
- cta: call-to-action sentence
- publish_format: blog_post or linkedin_post or twitter_thread

Return ONLY valid JSON. No markdown fences:
{"top5": [...]}"""


def _build_llm(model: str) -> ChatOpenAI:
    """Build a LangChain ChatOpenAI pointing at OpenRouter."""
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE,
        temperature=0,
        max_retries=2,
        timeout=120,
        default_headers={
            "HTTP-Referer": "https://openbox.ai",
            "X-Title": "OpenBox Content Studio",
        },
    )


def _build_middleware(agent_name: str) -> Any:
    """Initialize OpenBox governance and return config for the named agent."""
    initialize(
        api_url=os.environ.get("OPENBOX_API_URL", "https://core.openbox.ai"),
        api_key=os.environ["OPENBOX_API_KEY"],
        governance_timeout=30.0,
        validate=False,
        agent_did=os.environ.get("OPENBOX_AGENT_DID"),
        agent_private_key=os.environ.get("OPENBOX_AGENT_PRIVATE_KEY"),
    )
    return agent_name  # governance is initialized globally; return name for logging


def _parse_json_output(text: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON output from LLM."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def _age_label(published: str) -> str:
    """Convert ISO/RFC date string to human-readable age label."""
    if not published:
        return "unknown"
    try:
        from email.utils import parsedate_to_datetime
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            dt = parsedate_to_datetime(published).replace(tzinfo=UTC)
        now = datetime.now(UTC)
        hrs = max(0, int((now - dt).total_seconds() / 3600))
        if hrs < 1:
            return "less than 1 hour ago"
        if hrs < 24:
            return f"{hrs} hours ago"
        days = hrs // 24
        return f"{days} day{'s' if days != 1 else ''} ago"
    except Exception:
        return "unknown"


def _date_filter(items: list[dict[str, Any]], days: int = 7) -> list[dict[str, Any]]:
    """Keep only items published within `days` days (mirrors Combine & Date-Filter Pool)."""
    now = datetime.now(UTC)
    cutoff = now.timestamp() - days * 86400
    result = []
    for item in items:
        pub = item.get("published", "")
        if not pub:
            result.append({**item, "age_label": "unknown"})
            continue
        try:
            from email.utils import parsedate_to_datetime
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except ValueError:
                dt = parsedate_to_datetime(pub).replace(tzinfo=UTC)
            if dt.timestamp() >= cutoff:
                result.append({**item, "age_label": _age_label(pub)})
        except Exception:
            result.append({**item, "age_label": "unknown"})
    return result


def run_gate_agent(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Phase 1a — ICP Gate (mirrors n8n OpenBox: Agent node).

    Uses a governed LLM call (OpenBox initialized) to screen raw feed items
    for ICP relevance. Direct LLM invoke is used for pure JSON extraction
    (no tools needed in this stage).
    """
    if not raw_items:
        return []

    _build_middleware("ContentStudio-Gate")
    llm = _build_llm("anthropic/claude-haiku-4.5")

    payload = json.dumps({"items": raw_items}, ensure_ascii=False)
    _logger.info("Gate agent processing %d items", len(raw_items))

    response = llm.invoke([
        SystemMessage(content=ICP_GATE_SYSTEM),
        HumanMessage(content=payload),
    ])
    output = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_output(output)
        filtered = data.get("filtered", [])
        _logger.info("Gate passed %d items", len(filtered))
        return filtered
    except Exception as exc:
        _logger.warning("Gate agent JSON parse failed: %s — returning raw items", exc)
        return raw_items


def run_signal_extractor(filtered_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Phase 1b — Signal Extractor (mirrors n8n Signal Extractor langchain.agent node).

    Governed LLM call (DeepSeek Chat V3) — extracts structured signal
    (summary, relevance_hint) for each ICP-filtered item.
    """
    if not filtered_items:
        return []

    _build_middleware("ContentStudio-SignalExtractor")
    llm = _build_llm("anthropic/claude-haiku-4.5")

    payload = json.dumps({"items": filtered_items}, ensure_ascii=False)
    _logger.info("Signal extractor processing %d items", len(filtered_items))

    response = llm.invoke([
        SystemMessage(content=SIGNAL_EXTRACTOR_SYSTEM),
        HumanMessage(content=payload),
    ])
    output = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_output(output)
        extracted = data.get("extracted", [])
        _logger.info("Signal extractor extracted %d items", len(extracted))
        return extracted
    except Exception as exc:
        _logger.warning("Signal extractor parse failed: %s — using filtered items", exc)
        return filtered_items


def run_scorer(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Phase 1c — Score Top 5 (mirrors n8n Score Top 5 chainLlm node).

    Governed LLM call (DeepSeek Chat V3) — scores all items and returns
    the top 5 with full content briefs.
    """
    if not pool:
        return []

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    system = SCORER_SYSTEM + f"\n\nToday is {today}. You have {len(pool)} pre-filtered, governance-cleared signals."

    _build_middleware("ContentStudio-Scorer")
    llm = _build_llm("anthropic/claude-haiku-4.5")

    payload = json.dumps({"items": pool, "total": len(pool)}, ensure_ascii=False)
    _logger.info("Scorer processing %d items", len(pool))

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=payload),
    ])
    output = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_output(output)
        top5 = data.get("top5", [])
        _logger.info("Scorer returned %d briefs", len(top5))
        return top5
    except Exception as exc:
        _logger.warning("Scorer parse failed: %s — output preview: %s", exc, output[:300])
        return []


def run_intelligence_radar(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Full Phase 1 pipeline: Gate → Signal Extract → Date Filter → Score → Top 5 briefs.

    Returns list of up to 5 brief dicts, each containing topic, audience, angle,
    headline, hook, key_argument, openbox_product_tie, cta, etc.
    """
    _logger.info("Intelligence radar starting with %d raw items", len(raw_items))

    filtered = run_gate_agent(raw_items)
    if not filtered:
        _logger.warning("Gate returned 0 items — aborting radar")
        return []

    extracted = run_signal_extractor(filtered)
    if not extracted:
        _logger.warning("Signal extractor returned 0 items")
        return []

    pool = _date_filter(extracted, days=7)
    _logger.info("Date filter kept %d items", len(pool))
    if not pool:
        _logger.warning("All items filtered out by date — using unfiltered")
        pool = extracted

    top5 = run_scorer(pool)
    return top5
