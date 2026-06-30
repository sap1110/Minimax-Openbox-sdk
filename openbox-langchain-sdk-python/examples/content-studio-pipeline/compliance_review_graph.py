"""
Compliance + SEO Review Graph — LangGraph Agent with retry loop.

Mirrors the n8n nodes:
  Compliance + SEO Reviewer (langchain.agent, Gemini 2.5 Pro via OpenRouter)
  Verdict Extractor (code node)
  If (retry gate — _maxRetriesExceeded check)
  Publishing Pack Assembler (code node)

Architecture:
  LangGraph StateGraph with nodes:
    review_node     → calls Gemini 2.5 Pro agent (OpenBox governed)
    verdict_node    → extracts PASS/NEEDS_REVISION verdict
    assemble_node   → parses agent output into full publishing pack

  Edges:
    review_node → verdict_node
    verdict_node → assemble_node (PASS or max retries exceeded)
    verdict_node → content_strategy (NEEDS_REVISION, retry < MAX_RETRIES)
                   via returning "needs_revision" signal up to the pipeline orchestrator

The OpenBox middleware wraps the review agent, recording every compliance
decision as a governed audit event.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from openbox_langchain import initialize as _openbox_initialize  # noqa: F401

_logger = logging.getLogger("content_studio.compliance_review")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_RETRIES = 1

REVIEWER_SYSTEM = """You are a senior Compliance and SEO Reviewer for OpenBox AI content. Perform four independent checks:

**COPYRIGHT_CHECK:**
Scan for copied phrases, unattributed statistics, trademark conflicts. Rate: LOW / MEDIUM / HIGH. 2-3 sentence explanation.

**BRAND_SAFETY_CHECK:**
Flag controversial, politically charged, or off-brand content for a B2B AI governance brand. Rate: SAFE / CAUTION / UNSAFE.

**SEO_CHECK:**
- SEO_TITLE: length ≤60 chars? Primary keyword present?
- META_DESCRIPTION: length ≤155 chars? CTA present?
- KEYWORDS: relevance and variety
- HASHTAGS: platform appropriate, not too niche
- Readability: rough Flesch estimate
List concrete fixes.

**PLATFORM_COMPLIANCE:**
- LINKEDIN: ≤1300 chars, ≤3 hashtags
- X: ≤280 chars
- INSTAGRAM: ≥3 hashtags
- FACEBOOK: question present?
- THREADS: ≤500 chars
Flag violations.

**VERDICT:** PASS or NEEDS_REVISION

**REVISION_NOTES:**
If NEEDS_REVISION: numbered list of specific fixes with section names. If PASS: None.

Be strict but constructive."""

SECTIONS = [
    "UNIFIED_BRIEF", "LINKEDIN", "X", "INSTAGRAM", "FACEBOOK", "THREADS",
    "BLOG_TITLE", "BLOG_SUMMARY", "CTA", "HASHTAGS",
    "IMAGE_PROMPT_HERO", "IMAGE_PROMPT_SQUARE", "IMAGE_PROMPT_STORY",
    "SEO_TITLE", "META_DESCRIPTION", "KEYWORDS", "POSTING_SCHEDULE",
]


class ReviewState(TypedDict):
    """LangGraph state for the compliance review graph."""
    campaign_id: str
    topic: str
    audience: str
    platforms: list[str]
    raw_text: str
    brief: str
    agent_output: str
    review_raw: str
    verdict: str
    reviewer_notes: str
    retry_count: int
    pack: dict[str, Any]
    radar: dict[str, Any]
    _needs_revision: bool
    _max_retries_exceeded: bool


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="anthropic/claude-haiku-4.5",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE,
        temperature=0,
        max_retries=2,
        timeout=180,
        default_headers={
            "HTTP-Referer": "https://openbox.ai",
            "X-Title": "OpenBox Content Studio",
        },
    )


def _build_middleware(campaign_id: str) -> None:
    from openbox_langchain import initialize
    initialize(
        api_url=os.environ.get("OPENBOX_API_URL", "https://core.openbox.ai"),
        api_key=os.environ["OPENBOX_API_KEY"],
        governance_timeout=30.0,
        validate=False,
        agent_did=os.environ.get("OPENBOX_AGENT_DID"),
        agent_private_key=os.environ.get("OPENBOX_AGENT_PRIVATE_KEY"),
    )


def _parse_sections(raw: str) -> dict[str, str]:
    """Extract **SECTION:** content from agent output (mirrors n8n Publishing Pack Assembler)."""
    positions: list[tuple[str, int, int]] = []
    for label in SECTIONS:
        pattern = re.compile(r"\*\*" + label + r":\*\*", re.IGNORECASE)
        m = pattern.search(raw)
        if m:
            positions.append((label, m.start(), m.end()))
    positions.sort(key=lambda x: x[1])

    result: dict[str, str] = {}
    for k, (label, start, end) in enumerate(positions):
        stop = positions[k + 1][1] if k + 1 < len(positions) else len(raw)
        result[label] = raw[end:stop].strip()
    return result


def _parse_schedule(schedule_raw: str) -> dict[str, str]:
    """Parse POSTING_SCHEDULE section into per-platform times."""
    defaults = {
        "linkedin": "Tue 08:30",
        "x": "Wed 12:00",
        "instagram": "Thu 18:00",
        "facebook": "Fri 10:00",
        "threads": "Thu 19:00",
    }
    schedule: dict[str, str] = {}
    for line in schedule_raw.split("\n"):
        parts = line.split(":", 1)
        if len(parts) == 2:
            platform = re.sub(r"[^a-z]", "", parts[0].strip().lower())
            schedule[platform] = parts[1].strip()
    return {k: schedule.get(k, v) for k, v in defaults.items()}


def _parse_sections_flexible(raw: str) -> dict[str, str]:
    """Parse all sections using any header format: **SECTION:**, ## **SECTION**, SECTION:"""
    # Collect ALL section header positions across all formats
    positions: list[tuple[str, int, int]] = []
    for label in SECTIONS:
        pattern = re.compile(
            r"(?:^|\n)(?:#{1,3}\s+)?\*{0,2}" + re.escape(label) + r"\*{0,2}:?\s*",
            re.IGNORECASE | re.MULTILINE,
        )
        m = pattern.search(raw)
        if m:
            positions.append((label, m.start(), m.end()))

    if not positions:
        return {}

    positions.sort(key=lambda x: x[1])
    result: dict[str, str] = {}
    for k, (label, _start, end) in enumerate(positions):
        stop = positions[k + 1][1] if k + 1 < len(positions) else len(raw)
        result[label] = raw[end:stop].strip()
    return result


def _clean_image_prompt(text: str) -> str:
    """Strip size annotations, markdown bold markers, and leading dashes from prompts."""
    if not text:
        return ""
    # Remove leading size annotation like "(1200×630):**" or "(1080×1080):**\n\n"
    text = re.sub(r"^\s*\([\d×xX]+\)\s*:\*{0,2}\s*", "", text.strip())
    # Remove leading --- separators
    text = re.sub(r"^-{2,}\s*", "", text.strip())
    # Remove trailing --- separators
    text = re.sub(r"\s*-{2,}\s*$", "", text.strip())
    # Collapse the result and trim
    return text.strip()[:600]


def _extract_image_prompts(f: dict[str, str], raw: str) -> dict[str, str]:
    """Extract hero/square/story image prompts from parsed sections or raw text."""
    hero = _clean_image_prompt(f.get("IMAGE_PROMPT_HERO", ""))
    square = _clean_image_prompt(f.get("IMAGE_PROMPT_SQUARE", ""))
    story = _clean_image_prompt(f.get("IMAGE_PROMPT_STORY", ""))
    if hero or square or story:
        return {"hero": hero, "square": square, "story": story}
    # Fallback: search raw text for image prompt lines
    for variant, key in [
        (r"(?i)IMAGE_PROMPT_HERO[:\*\s]+(.+?)(?=\n\*\*IMAGE_PROMPT|\n##|\Z)", "hero"),
        (r"(?i)IMAGE_PROMPT_SQUARE[:\*\s]+(.+?)(?=\n\*\*IMAGE_PROMPT|\n##|\Z)", "square"),
        (r"(?i)IMAGE_PROMPT_STORY[:\*\s]+(.+?)(?=\n\*\*SEO_TITLE|\n##|\Z)", "story"),
    ]:
        m = re.search(variant, raw, re.DOTALL)
        if m:
            val = _clean_image_prompt(m.group(1))
            if key == "hero":
                hero = val
            elif key == "square":
                square = val
            elif key == "story":
                story = val
    return {"hero": hero, "square": square, "story": story}


def _assemble_pack(state: ReviewState) -> dict[str, Any]:
    """Build the full publishing pack dict from parsed sections (mirrors n8n Publishing Pack Assembler)."""
    raw = state["agent_output"]
    f = _parse_sections_flexible(raw)

    posting_times = _parse_schedule(f.get("POSTING_SCHEDULE", ""))

    return {
        "campaignId": state["campaign_id"],
        "topic": state["topic"],
        "audience": state["audience"],
        "platforms": state["platforms"],
        "radar": state["radar"],
        "brief": f.get("UNIFIED_BRIEF", ""),
        "content": {
            "linkedin": f.get("LINKEDIN", ""),
            "x": f.get("X", ""),
            "instagram": f.get("INSTAGRAM", ""),
            "facebook": f.get("FACEBOOK", ""),
            "threads": f.get("THREADS", ""),
        },
        "blog": {
            "title": f.get("BLOG_TITLE", ""),
            "summary": f.get("BLOG_SUMMARY", ""),
        },
        "cta": f.get("CTA", ""),
        "hashtags": f.get("HASHTAGS", ""),
        "imagePrompts": _extract_image_prompts(f, raw),
        "seo": {
            "title": f.get("SEO_TITLE", ""),
            "meta": f.get("META_DESCRIPTION", ""),
            "keywords": f.get("KEYWORDS", ""),
        },
        "postingTimes": posting_times,
        "review": state["review_raw"],
        "rawAgentOutput": raw,
    }


def review_node(state: ReviewState) -> dict[str, Any]:
    """
    LangGraph node: Compliance + SEO Reviewer agent.

    Calls Gemini 2.5 Pro via OpenRouter, wrapped in OpenBox governance middleware.
    """
    campaign_id = state["campaign_id"]
    agent_output = state["agent_output"]

    prompt_text = f"CAMPAIGN ID: {campaign_id}\n\nCONTENT TO REVIEW:\n{agent_output}"

    _logger.info("Compliance review: campaign=%s retry=%d", campaign_id, state["retry_count"])

    _build_middleware(campaign_id)
    llm = _build_llm()

    response = llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=prompt_text),
    ])
    review_raw = response.content if hasattr(response, "content") else str(response)

    return {"review_raw": review_raw}


def verdict_node(state: ReviewState) -> dict[str, Any]:
    """
    LangGraph node: extract verdict and revision notes from review output.
    Mirrors n8n Verdict Extractor code node.
    """
    review_raw = state.get("review_raw", "")

    verdict_match = re.search(r"\*\*VERDICT:\*\*\s*(PASS|NEEDS_REVISION)", review_raw, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "NEEDS_REVISION"

    notes_match = re.search(
        r"\*\*REVISION_NOTES:\*\*([\s\S]*?)(?=\*\*[A-Z]|$)", review_raw, re.IGNORECASE
    )
    reviewer_notes = notes_match.group(1).strip() if notes_match else ""

    retry_count = state.get("retry_count", 0)
    _needs_revision = verdict == "NEEDS_REVISION"
    _max_retries_exceeded = retry_count >= MAX_RETRIES

    _logger.info(
        "Verdict: %s | retry=%d | max_exceeded=%s", verdict, retry_count, _max_retries_exceeded
    )

    return {
        "verdict": verdict,
        "reviewer_notes": reviewer_notes,
        "_needs_revision": _needs_revision,
        "_max_retries_exceeded": _max_retries_exceeded,
    }


def assemble_node(state: ReviewState) -> dict[str, Any]:
    """
    LangGraph node: assemble the final publishing pack from agent output + review.
    Mirrors n8n Publishing Pack Assembler code node.
    """
    pack = _assemble_pack(state)
    _logger.info("Pack assembled: campaign=%s", state["campaign_id"])
    return {"pack": pack}


def _route_after_verdict(
    state: ReviewState,
) -> Literal["assemble", "needs_revision"]:
    """
    Conditional edge: route to assemble if PASS or max retries exceeded,
    otherwise signal needs_revision (handled by pipeline orchestrator).
    """
    if not state["_needs_revision"] or state["_max_retries_exceeded"]:
        return "assemble"
    return "needs_revision"


def build_compliance_graph() -> StateGraph:
    """Build and compile the compliance review LangGraph StateGraph."""
    graph = StateGraph(ReviewState)

    graph.add_node("review", review_node)
    graph.add_node("verdict", verdict_node)
    graph.add_node("assemble", assemble_node)

    graph.add_edge(START, "review")
    graph.add_edge("review", "verdict")
    graph.add_conditional_edges(
        "verdict",
        _route_after_verdict,
        {"assemble": "assemble", "needs_revision": END},
    )
    graph.add_edge("assemble", END)

    return graph.compile()


def run_compliance_review(content_state: dict[str, Any]) -> dict[str, Any]:
    """
    Entry point: run the full compliance review graph for one content state.

    Accepts the dict returned by content_strategy_agent.run_content_strategy().
    Returns updated state with verdict, reviewer_notes, pack (if PASS), and
    _needs_revision flag so the pipeline orchestrator can trigger retry.
    """
    graph = build_compliance_graph()

    initial_state: ReviewState = {
        "campaign_id": content_state.get("campaignId", ""),
        "topic": content_state.get("topic", ""),
        "audience": content_state.get("audience", ""),
        "platforms": content_state.get("platforms", []),
        "raw_text": content_state.get("rawText", ""),
        "brief": content_state.get("brief", ""),
        "agent_output": content_state.get("agentOutput", ""),
        "review_raw": "",
        "verdict": "",
        "reviewer_notes": "",
        "retry_count": content_state.get("retryCount", 0),
        "pack": {},
        "radar": content_state.get("_radar", {}),
        "_needs_revision": False,
        "_max_retries_exceeded": False,
    }

    result = graph.invoke(initial_state)

    return {
        **content_state,
        "reviewRaw": result.get("review_raw", ""),
        "verdict": result.get("verdict", ""),
        "reviewerNotes": result.get("reviewer_notes", ""),
        "_pass": result.get("verdict", "") == "PASS",
        "_needsRevision": result.get("_needs_revision", False),
        "_maxRetriesExceeded": result.get("_max_retries_exceeded", False),
        "pack": result.get("pack", {}),
    }
