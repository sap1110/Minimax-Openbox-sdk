"""
Content Strategy Agent — LangChain Agent + OpenBox governance.

Mirrors the n8n nodes:
  Content Intelligence Parser (code node — brief assembly)
  Retry Counter (code node — retry state management)
  If (conditional — retry gate)
  Content Governance (CUSTOM.openBoxAgent — main content writer)
    ↳ Claude Sonnet 4 (Studio) via OpenRouter
    ↳ Studio Memory (window buffer)
    ↳ Wikipedia Research tool

The OpenBox middleware wraps the agent, enforcing governance, recording audit
spans, and enabling HITL for content approval if configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

from openbox_langchain import create_openbox_langchain_middleware

_logger = logging.getLogger("content_studio.content_strategy")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_RETRIES = 2

CONTENT_STUDIO_SYSTEM = """You are the OpenBox AI Content Studio — a single governed agent that internally fulfils five expert roles simultaneously:

1. CAMPAIGN STRATEGIST — Define the business goal, key message, success metric, and platform strategy.
2. MARKET RESEARCHER — Use the Wikipedia tool to verify any factual claim, statistic, or named entity before including it.
3. SENIOR COPYWRITER — Write all platform copy at the appropriate register and length for each channel.
4. SEO SPECIALIST — Craft the title tag, meta description, and keyword set to maximise organic reach.
5. CREATIVE DIRECTOR — Specify vivid, art-directed image generation prompts tailored to each format.

OpenBox context: enterprise AI trust and governance platform. Features include Trust Score, OPA/Rego Behavioral Rules, Merkle Audit Trail, HITL approvals, Session Replay, Trust Lifecycle. ICP: CISOs, AI Platform Leads, Compliance Officers.

For the content brief you receive, produce a complete publishing pack using these EXACT section headers:

**UNIFIED_BRIEF:** Business goal | Target audience | Key message | Tone | Success metric

**LINKEDIN:** Professional post. Max 1300 characters. Lead with insight. Max 3 hashtags.

**X:** Max 280 characters. Hook in first 8 words. 1-2 hashtags.

**INSTAGRAM:** Hook line first. Purposeful emojis. 5 targeted hashtags at end.

**FACEBOOK:** Conversational, 100-250 words, one question to drive comments.

**THREADS:** Casual, 500-char max, one emoji, no hashtags.

**BLOG_TITLE:** Compelling headline, max 70 characters.

**BLOG_SUMMARY:** 150-word summary. Punchy opening, 3 key points, clear CTA.

**CTA:** Single call-to-action sentence usable across all platforms.

**HASHTAGS:** 10 researched hashtags, ranked by relevance, comma-separated.

**IMAGE_PROMPT_HERO:** Vivid 1200×630 description. Subject, composition, lighting, colour palette, mood. No text overlays.

**IMAGE_PROMPT_SQUARE:** 1080×1080 square version.

**IMAGE_PROMPT_STORY:** 1080×1920 vertical story version.

**SEO_TITLE:** Max 60 characters. Primary keyword near the front.

**META_DESCRIPTION:** Max 155 characters. Include CTA.

**KEYWORDS:** 8 SEO keywords, comma-separated, ranked by relevance.

**POSTING_SCHEDULE:**
LinkedIn: [day time]
X: [day time]
Instagram: [day time]
Facebook: [day time]
Threads: [day time]

Be precise, platform-aware, and brand-consistent. Every section must be complete."""


def _build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE,
        temperature=0.3,
        max_retries=2,
        timeout=120,
        default_headers={
            "HTTP-Referer": "https://openbox.ai",
            "X-Title": "OpenBox Content Studio",
        },
    )


def _build_middleware(agent_name: str, session_id: str | None = None) -> Any:
    return create_openbox_langchain_middleware(
        api_url=os.environ.get("OPENBOX_API_URL", "https://core.openbox.ai"),
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name=agent_name,
        agent_did=os.environ.get("OPENBOX_AGENT_DID"),
        agent_private_key=os.environ.get("OPENBOX_AGENT_PRIVATE_KEY"),
        session_id=session_id,
        validate=False,
    )


def _build_brief(item: dict[str, Any], reviewer_notes: str = "", retry_count: int = 0) -> str:
    """
    Assemble the content brief from a top-5 radar item.
    Mirrors n8n Content Intelligence Parser + Retry Counter nodes.
    """
    campaign_id = f"radar_{item.get('rank', 0)}_{int(time.time())}"
    topic = item.get("headline") or item.get("title", "")
    audience = item.get("persona", "CISO")
    angle = item.get("content_type", "thought_leadership")
    platforms = ["linkedin", "x", "instagram", "facebook", "threads"]

    if item.get("source") == "user-prompt":
        raw_text = (
            f"TOPIC: {item.get('title', '')}\n"
            f"AUDIENCE: {item.get('persona', 'General')}\n"
            f"SUMMARY: {item.get('summary', item.get('title', ''))}\n"
            f"CTA: {item.get('cta', '')}"
        )
    else:
        raw_text_parts = [
            f"SOURCE ARTICLE: {item.get('title', '')}",
            f"URL: {item.get('url', '')}",
            f"SOURCE: {item.get('source', '')}",
            f"PUBLISHED: {item.get('published', '')} ({item.get('age_label', '')})",
            "",
            "CONTENT BRIEF:",
            f"Headline: {item.get('headline', '')}",
            f"Hook: {item.get('hook', '')}",
            f"Key argument: {item.get('key_argument', '')}",
            f"Subheadings: {' → '.join(item.get('three_subheadings', []))}",
            f"OpenBox product tie: {item.get('openbox_product_tie', '')}",
            f"CTA: {item.get('cta', '')}",
            f"Format: {item.get('publish_format', '')}",
            "",
            f"WHY OPENBOX: {item.get('why_openbox', '')}",
        ]
        raw_text = "\n".join(raw_text_parts)

    brief_lines = [
        f"CAMPAIGN ID: {campaign_id}",
        f"TOPIC: {topic}",
        f"TARGET AUDIENCE: {audience}",
        f"ANGLE: {angle}",
        f"TONE: authoritative",
        f"PLATFORMS: {', '.join(platforms)}",
        "",
        "FULL CONTENT BRIEF FROM INTELLIGENCE RADAR:",
        raw_text,
        "",
        "Using the brief above, produce the full publishing pack for the platforms listed.",
    ]

    if reviewer_notes and retry_count > 0:
        brief_lines += [
            "",
            "──────────────────────────────────────",
            f"REVIEWER FEEDBACK (attempt {retry_count}):",
            reviewer_notes,
            "",
            "Please address ALL points above and regenerate the full publishing pack.",
        ]

    return "\n".join(brief_lines), campaign_id, topic, audience, platforms, raw_text


GENERIC_CONTENT_SYSTEM = """You are a professional Content Studio AI. You create complete, high-quality social media publishing packs for any brand or topic.

For the content brief you receive, produce a complete publishing pack using these EXACT section headers:

**UNIFIED_BRIEF:** Business goal | Target audience | Key message | Tone | Success metric

**LINKEDIN:** Professional post. Max 1300 characters. Lead with insight. Max 3 hashtags.

**X:** Max 280 characters. Hook in first 8 words. 1-2 hashtags.

**INSTAGRAM:** Hook line first. Purposeful emojis. 5 targeted hashtags at end.

**FACEBOOK:** Conversational, 100-250 words, one question to drive comments.

**THREADS:** Casual, 500-char max, one emoji, no hashtags.

**BLOG_TITLE:** Compelling headline, max 70 characters.

**BLOG_SUMMARY:** 150-word summary. Punchy opening, 3 key points, clear CTA.

**CTA:** Single call-to-action sentence usable across all platforms.

**HASHTAGS:** 10 researched hashtags, ranked by relevance, comma-separated.

**IMAGE_PROMPT_HERO:** Vivid 1200×630 description. Subject, composition, lighting, colour palette, mood. No text overlays.

**IMAGE_PROMPT_SQUARE:** 1080×1080 square version.

**IMAGE_PROMPT_STORY:** 1080×1920 vertical story version.

**SEO_TITLE:** Max 60 characters. Primary keyword near the front.

**META_DESCRIPTION:** Max 155 characters. Include CTA.

**KEYWORDS:** 8 SEO keywords, comma-separated, ranked by relevance.

**POSTING_SCHEDULE:**
LinkedIn: [day time]
X: [day time]
Instagram: [day time]
Facebook: [day time]
Threads: [day time]

Be precise, platform-aware, and brand-consistent. Every section must be complete."""


def run_content_strategy(
    radar_item: dict[str, Any],
    reviewer_notes: str = "",
    retry_count: int = 0,
    campaign_id: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Run the Content Governance agent for a single radar brief item.

    Mirrors n8n Content Governance CUSTOM.openBoxAgent node with:
    - Claude claude-3.5-haiku via OpenRouter
    - Wikipedia research tool
    - Per-campaign session memory via OpenBox session_id
    - OpenBox governance middleware wrapping the full agent

    Returns dict with campaignId, topic, audience, platforms, rawText,
    brief, agentOutput, retryCount.
    """
    brief, auto_campaign_id, topic, audience, platforms, raw_text = _build_brief(
        radar_item, reviewer_notes, retry_count
    )
    if campaign_id is None:
        campaign_id = auto_campaign_id

    _logger.info(
        "Content strategy agent: campaign=%s retry=%d", campaign_id, retry_count
    )

    middleware = _build_middleware(
        agent_name="ContentStudio-ContentStrategy",
        session_id=campaign_id,
    )

    llm = _build_llm("anthropic/claude-haiku-4.5")

    try:
        from langchain_core.tools import StructuredTool

        _wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

        def _safe_wikipedia(query: str) -> str:
            """Search Wikipedia for factual context. Returns results or a graceful fallback."""
            try:
                return _wiki_tool.run(query)
            except Exception as exc:
                return f"Wikipedia lookup unavailable ({exc}). Proceed without this fact-check."

        wikipedia = StructuredTool.from_function(
            func=_safe_wikipedia,
            name="wikipedia",
            description="Search Wikipedia for factual context about a topic, named entity, or statistic.",
        )
    except Exception:
        wikipedia = None

    tools = [wikipedia] if wikipedia else []

    active_system = system_prompt or CONTENT_STUDIO_SYSTEM

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=active_system,
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [("user", brief)]})

    output = ""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and "**UNIFIED_BRIEF:**" in content:
            output = content
            break
    if not output:
        for msg in reversed(result.get("messages", [])):
            content = getattr(msg, "content", None)
            if content and isinstance(content, str):
                output = content
                break

    return {
        "campaignId": campaign_id,
        "topic": topic,
        "audience": audience,
        "platforms": platforms,
        "rawText": raw_text,
        "brief": brief,
        "agentOutput": output,
        "retryCount": retry_count,
        "reviewerNotes": reviewer_notes,
        "_radar": radar_item,
    }
