"""
Image Agent — MiniMax image generation via OpenRouter + OpenBox governance.

Mirrors the n8n nodes:
  Image Content Governance (CUSTOM.openBoxAgent) — image prompt censorship check
  Image Generation (code node) — replaced with actual MiniMax API calls

MiniMax model: MiniMax-Image-01 via OpenRouter
  - Generates actual images (base64 → saved to disk)
  - Three formats: hero (1792×1024), square (1024×1024), story (1024×1792)

The image censorship check uses a LangChain Agent wrapped in OpenBox governance
to screen image prompts before generation.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from openbox_langchain import initialize as _openbox_initialize  # noqa: F401

_logger = logging.getLogger("content_studio.image_agent")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

IMAGE_CENSORSHIP_SYSTEM = """You are an image content safety reviewer for OpenBox AI — a B2B enterprise AI governance brand.

Review each image generation prompt and flag any that contain:
- Violence, gore, or disturbing imagery
- Sexually explicit or suggestive content
- Discriminatory, hateful, or politically charged content
- Content that violates brand safety for a B2B audience
- Trademarked characters, logos, or copyrighted imagery
- Real named individuals depicted in compromising or unauthorized scenarios

For each prompt, respond with:
PROMPT: [original]
SAFE: YES or NO
REASON: [brief explanation if NO]

At the end, output a JSON summary:
{"prompts": [{"original": "...", "safe": true/false, "sanitized": "...or same if safe"}]}"""

IMAGE_FORMATS = {
    "hero": {"width": 1792, "height": 1024, "label": "Hero (1792×1024)"},
    "square": {"width": 1024, "height": 1024, "label": "Square (1024×1024)"},
    "story": {"width": 1024, "height": 1792, "label": "Story (1024×1792)"},
}


def _build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE,
        temperature=0,
        max_retries=2,
        timeout=60,
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


def run_image_censorship_check(
    image_prompts: dict[str, str],
    campaign_id: str,
) -> dict[str, str]:
    """
    Image Content Governance agent (mirrors n8n Image Content Governance CUSTOM.openBoxAgent).

    Screens hero/square/story image prompts through an OpenBox-governed LangChain agent.
    Returns sanitized prompts (same if safe, modified if flagged).
    """
    prompt_text = "\n".join([
        f"{k.upper()} PROMPT: {v}" for k, v in image_prompts.items()
    ])

    _logger.info("Image censorship check: campaign=%s", campaign_id)

    _build_middleware(campaign_id)
    llm = _build_llm("anthropic/claude-haiku-4.5")

    response = llm.invoke([
        SystemMessage(content=IMAGE_CENSORSHIP_SYSTEM),
        HumanMessage(content=prompt_text),
    ])
    output = response.content if hasattr(response, "content") else str(response)

    sanitized: dict[str, str] = dict(image_prompts)
    try:
        import json
        import re
        json_match = re.search(r'\{[\s\S]*"prompts"[\s\S]*\}', output)
        if json_match:
            data = json.loads(json_match.group())
            prompts_list = data.get("prompts", [])
            keys = list(image_prompts.keys())
            for i, entry in enumerate(prompts_list):
                if i < len(keys):
                    sanitized_prompt = entry.get("sanitized", entry.get("original", ""))
                    if sanitized_prompt:
                        sanitized[keys[i]] = sanitized_prompt
    except Exception as exc:
        _logger.warning("Image censorship JSON parse failed: %s — using originals", exc)

    return sanitized


def _generate_single_image_minimax(
    prompt: str,
    width: int,
    height: int,
    campaign_id: str,
    format_name: str,
    output_dir: Path,
) -> str | None:
    """
    Call MiniMax-Image-01 via OpenRouter images endpoint.
    Returns saved file path or None on failure.

    OpenRouter images API: POST /images/generations
    """
    url = f"{OPENROUTER_BASE}/images/generations"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openbox.ai",
        "X-Title": "OpenBox Content Studio",
    }

    payload = {
        "model": "minimax/minimax-image-01",
        "prompt": prompt[:400],
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
    }

    _logger.info("MiniMax image gen: campaign=%s format=%s size=%dx%d", campaign_id, format_name, width, height)

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        image_data = data.get("data", [])
        if not image_data:
            _logger.warning("MiniMax returned no image data for %s", format_name)
            return None

        b64 = image_data[0].get("b64_json", "")
        if not b64:
            _logger.warning("MiniMax returned empty b64_json for %s", format_name)
            return None

        safe_campaign = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{safe_campaign}_{format_name}.png"

        image_bytes = base64.b64decode(b64)
        file_path.write_bytes(image_bytes)
        _logger.info("Image saved: %s (%d bytes)", file_path, len(image_bytes))
        return str(file_path)

    except httpx.HTTPStatusError as exc:
        _logger.error("MiniMax HTTP error %s: %s", exc.response.status_code, exc.response.text[:300])
        return None
    except Exception as exc:
        _logger.error("MiniMax image generation failed (%s): %s", format_name, exc)
        return None


def run_image_generation(
    pack: dict[str, Any],
    campaign_id: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full image pipeline:
    1. Image Content Governance (censorship check via OpenBox-governed agent)
    2. MiniMax-Image-01 generation via OpenRouter for hero/square/story formats
    3. Save to disk, return paths

    Mirrors n8n Image Content Governance → Image Generation nodes.
    """
    if output_dir is None:
        output_dir = Path(os.environ.get("OUTPUT_DIR", "./output")) / "images"

    image_prompts = pack.get("imagePrompts", {})
    hero_prompt = (
        image_prompts.get("hero")
        or pack.get("IMAGE_PROMPT_HERO")
        or pack.get("image_prompt_hero", "")
    )
    square_prompt = (
        image_prompts.get("square")
        or pack.get("IMAGE_PROMPT_SQUARE")
        or pack.get("image_prompt_square", "")
    )
    story_prompt = (
        image_prompts.get("story")
        or pack.get("IMAGE_PROMPT_STORY")
        or pack.get("image_prompt_story", "")
    )

    # Last-resort: regex scan rawAgentOutput
    if not any([hero_prompt, square_prompt, story_prompt]):
        import re
        raw = pack.get("rawAgentOutput", "")
        if raw:
            for pattern, key in [
                (r"(?i)IMAGE_PROMPT_HERO[:\*\s]+(.+?)(?=\n\*\*|\n##|\Z)", "hero"),
                (r"(?i)IMAGE_PROMPT_SQUARE[:\*\s]+(.+?)(?=\n\*\*|\n##|\Z)", "square"),
                (r"(?i)IMAGE_PROMPT_STORY[:\*\s]+(.+?)(?=\n\*\*|\n##|\Z)", "story"),
            ]:
                m = re.search(pattern, raw, re.DOTALL)
                if m:
                    val = m.group(1).strip()[:500]
                    if key == "hero":
                        hero_prompt = val
                    elif key == "square":
                        square_prompt = val
                    elif key == "story":
                        story_prompt = val

    if not any([hero_prompt, square_prompt, story_prompt]):
        _logger.warning("No image prompts found in pack — skipping image generation")
        _logger.debug("Pack keys: %s", list(pack.keys()))
        return {"hero": None, "square": None, "story": None}

    def _clean(p: str) -> str:
        import re
        p = re.sub(r"^\s*\([\d×xX]+\)\s*:\*{0,2}\s*", "", p.strip())
        p = re.sub(r"^-{2,}\s*", "", p.strip())
        p = re.sub(r"\s*-{2,}\s*$", "", p.strip())
        return p.strip()

    raw_prompts = {
        "hero": _clean(hero_prompt),
        "square": _clean(square_prompt),
        "story": _clean(story_prompt),
    }

    sanitized_prompts = run_image_censorship_check(raw_prompts, campaign_id)

    images: dict[str, str | None] = {}
    for format_name, fmt_config in IMAGE_FORMATS.items():
        prompt = sanitized_prompts.get(format_name, "")
        if not prompt:
            images[format_name] = None
            continue

        path = _generate_single_image_minimax(
            prompt=prompt,
            width=fmt_config["width"],
            height=fmt_config["height"],
            campaign_id=campaign_id,
            format_name=format_name,
            output_dir=output_dir,
        )
        images[format_name] = path

        if format_name != "story":
            time.sleep(1)

    _logger.info(
        "Image generation complete: hero=%s square=%s story=%s",
        images.get("hero"), images.get("square"), images.get("story"),
    )
    return images
