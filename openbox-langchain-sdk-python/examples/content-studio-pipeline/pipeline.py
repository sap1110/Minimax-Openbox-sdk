"""
Content Studio Pipeline — Orchestrator

Wires together all pipeline stages in the exact order the n8n workflow executes:

  Phase 1 — Intelligence Radar
  ─────────────────────────────
  ingestion.fetch_all_feeds()
      ↓
  gate_agent.run_intelligence_radar()
      ├─ Gate Agent          (LangChain + OpenBox governance)
      ├─ Signal Extractor    (LangChain + OpenBox governance)
      └─ Scorer / Top 5      (LangChain + OpenBox governance)
      ↓ [list of up to 5 brief items]

  Phase 2 — Content Studio (per brief, with retry loop)
  ──────────────────────────────────────────────────────
  for each brief:
      content_strategy_agent.run_content_strategy()
          (LangChain + OpenBox governance, Wikipedia tool)
          ↓
      compliance_review_graph.run_compliance_review()
          (LangGraph graph: review_node → verdict_node → assemble_node)
          (LangChain Gemini reviewer + OpenBox governance)
          ↓
      if NEEDS_REVISION and retry < MAX_RETRIES:
          → inject reviewer notes → retry content strategy
      else:
          → proceed to image generation
              ↓
      image_agent.run_image_generation()
          ├─ Image censorship check  (LangChain + OpenBox governance)
          └─ MiniMax-Image-01        (actual image generation via OpenRouter)

  Phase 3 — Final Report
  ────────────────────────
  Build final report dict + save to disk as JSON + Markdown

Usage:
    python pipeline.py
    python pipeline.py --dry-run      # skip image generation
    python pipeline.py --brief-only   # only run Phase 1 radar
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

logging.basicConfig(level=logging.WARNING)
for logger_name in [
    "content_studio.ingestion",
    "content_studio.gate_agent",
    "content_studio.content_strategy",
    "content_studio.compliance_review",
    "content_studio.image_agent",
    "content_studio.pipeline",
]:
    logging.getLogger(logger_name).setLevel(logging.INFO)

_logger = logging.getLogger("content_studio.pipeline")
console = Console()

MAX_RETRIES = 1
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))


def _check_env() -> None:
    """Validate required environment variables."""
    required = ["OPENROUTER_API_KEY", "OPENBOX_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        console.print(f"[bold red]Missing required env vars: {', '.join(missing)}[/]")
        console.print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def _save_output(campaign_id: str, data: dict[str, Any]) -> Path:
    """Save final report as JSON to output directory."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign_id)
    out_dir = OUTPUT_DIR / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = out_dir / "report.md"
    md_path.write_text(_build_markdown(data), encoding="utf-8")

    return out_dir


def _build_markdown(data: dict[str, Any]) -> str:
    """Build a Markdown report from the final pipeline output (mirrors n8n Final Report Builder)."""
    pack = data.get("pack", {})
    images = data.get("images", {})
    content = pack.get("content", {})
    blog = pack.get("blog", {})
    seo = pack.get("seo", {})
    pt = pack.get("postingTimes", {})
    radar = pack.get("radar", {})

    lines = [
        f"# OpenBox Content Pack — {pack.get('topic', '')[:80]}",
        f"**Campaign:** {pack.get('campaignId', '')}  |  **Audience:** {pack.get('audience', '')}",
        f"**Source signal:** {radar.get('title', '')}",
        f"**OpenBox product tie:** {radar.get('openbox_product_tie', '')}",
        "---",
        "## Campaign Brief",
        pack.get("brief", ""),
        "---",
        "## Platform Copy",
        "### LinkedIn",
        content.get("linkedin", ""),
        "### X / Twitter",
        content.get("x", ""),
        "### Instagram",
        content.get("instagram", ""),
        "### Facebook",
        content.get("facebook", ""),
        "### Threads",
        content.get("threads", ""),
        "---",
        "## Blog",
        f"**Title:** {blog.get('title', '')}",
        blog.get("summary", ""),
        "---",
        f"## CTA\n{pack.get('cta', '')}",
        f"## Hashtags\n{pack.get('hashtags', '')}",
        "---",
        "## SEO",
        f"**Title tag:** {seo.get('title', '')}",
        f"**Meta description:** {seo.get('meta', '')}",
        f"**Keywords:** {seo.get('keywords', '')}",
        "---",
        "## Generated Images",
    ]

    for fmt, path in images.items():
        if path:
            lines.append(f"**{fmt.capitalize()}:** `{path}`")
        else:
            lines.append(f"**{fmt.capitalize()}:** (not generated)")

    lines += [
        "---",
        "## Posting Schedule",
        *[f"- **{p.capitalize()}:** {t}" for p, t in pt.items()],
        "---",
        "## Compliance + SEO Review",
        data.get("reviewRaw", ""),
    ]

    return "\n\n".join(line for line in lines if line != "")


def run_from_prompt(topic: str, audience: str = "AI Platform Lead", dry_run: bool = False) -> dict[str, Any]:
    """
    Run the full Content Studio pipeline for a single user-supplied topic/prompt.
    Bypasses RSS ingestion and goes directly to content generation.
    Returns the final result dict.
    """
    _check_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from content_strategy_agent import run_content_strategy, GENERIC_CONTENT_SYSTEM
    from compliance_review_graph import run_compliance_review
    from image_agent import run_image_generation

    campaign_id = f"prompt_{int(time.time())}"

    # Build a brief item that mimics what the scorer produces
    brief_item: dict[str, Any] = {
        "headline": topic[:120],
        "title": topic[:120],
        "persona": audience,
        "composite_score": 9.0,
        "score": 9.0,
        "summary": topic,
        "url": "",
        "source": "user-prompt",
        "published": datetime.now(UTC).isoformat(),
        "age_label": "now",
        "why_openbox": "User-supplied topic for content generation.",
        "angle": "thought_leadership",
        "tone": "authoritative",
        "platforms": ["linkedin", "x", "instagram", "facebook", "threads"],
        "cta": f"Learn how OpenBox can help with {topic[:60]}.",
    }

    content_state = run_content_strategy(
        radar_item=brief_item,
        campaign_id=campaign_id,
        retry_count=0,
        reviewer_notes="",
        system_prompt=GENERIC_CONTENT_SYSTEM,
    )

    review_result: dict[str, Any] | None = None
    retry_count = 0
    reviewer_notes = ""

    while True:
        content_state["retryCount"] = retry_count
        content_state["reviewerNotes"] = reviewer_notes
        review_result = run_compliance_review(content_state)

        verdict = review_result.get("verdict", "")
        if verdict == "PASS" or review_result.get("_maxRetriesExceeded", False):
            break
        if retry_count >= MAX_RETRIES:
            break

        reviewer_notes = review_result.get("reviewerNotes", "")
        retry_count += 1
        content_state = run_content_strategy(
            radar_item=brief_item,
            campaign_id=f"{campaign_id}_{retry_count}",
            retry_count=retry_count,
            reviewer_notes=reviewer_notes,
            system_prompt=GENERIC_CONTENT_SYSTEM,
        )

    if review_result is None:
        return {"error": "Pipeline failed — no review result"}

    pack = review_result.get("pack") or {}
    if not pack.get("content") and not pack.get("rawAgentOutput"):
        pack["rawAgentOutput"] = content_state.get("agentOutput", "")
        pack["campaignId"] = campaign_id

    images: dict[str, str | None] = {"hero": None, "square": None, "story": None}
    if not dry_run:
        images = run_image_generation(
            pack=pack,
            campaign_id=campaign_id,
            output_dir=OUTPUT_DIR / "images",
        )

    final = {
        **review_result,
        "campaignId": campaign_id,
        "userPrompt": topic,
        "images": images,
        "generatedAt": datetime.now(UTC).isoformat(),
    }

    out_dir = _save_output(campaign_id, final)
    _logger.info("Prompt-based campaign saved to %s", out_dir)
    return final


def run_pipeline(dry_run: bool = False, brief_only: bool = False) -> list[dict[str, Any]]:
    """
    Run the full Content Studio pipeline end-to-end.

    Returns list of final result dicts (one per brief that completed).
    """
    _check_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(Panel(
        "[bold blue]OpenBox Content Studio Pipeline[/]\n"
        "[dim]LangChain · LangGraph · MiniMax · OpenBox Governance[/]",
        border_style="blue",
    ))
    console.print()

    from ingestion import fetch_all_feeds
    from gate_agent import run_intelligence_radar
    from content_strategy_agent import run_content_strategy
    from compliance_review_graph import run_compliance_review
    from image_agent import run_image_generation

    # ── Phase 1: Intelligence Radar ─────────────────────────────────
    console.print("[bold cyan]Phase 1 — Intelligence Radar[/]")
    console.print("[dim]Fetching feeds from 9 RSS/API sources...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching & deduplicating feeds...", total=None)
        raw_items = fetch_all_feeds()
        progress.update(task, description=f"Fetched {len(raw_items)} items. Running governance radar...")
        top5 = run_intelligence_radar(raw_items)
        progress.update(task, description=f"Radar complete — {len(top5)} briefs ready.")

    if not top5:
        console.print("[bold red]No briefs from radar — pipeline halted.[/]")
        return []

    console.print(f"\n[bold green]✓ Radar identified {len(top5)} content opportunities:[/]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Topic / Headline", width=60)
    table.add_column("Persona", width=20)
    table.add_column("Score", width=8)
    for i, brief in enumerate(top5, 1):
        table.add_row(
            str(i),
            (brief.get("headline") or brief.get("title", ""))[:58],
            brief.get("persona", ""),
            str(round(float(brief.get("composite_score") or brief.get("score", 0)), 1)),
        )
    console.print(table)

    if brief_only:
        console.print("\n[dim]--brief-only: stopping after Phase 1.[/]")
        return []  # no full campaign results, but Phase 1 succeeded — caller exits 0 via try/except

    # ── Phase 2 & 3: Content Studio (per brief) ──────────────────────
    console.print(f"\n[bold cyan]Phase 2 — Content Studio ({len(top5)} campaigns)[/]")

    results: list[dict[str, Any]] = []

    for idx, brief_item in enumerate(top5, 1):
        headline = (brief_item.get("headline") or brief_item.get("title", ""))[:60]
        console.print(f"\n[bold]Campaign {idx}/{len(top5)}:[/] {headline}")

        content_state: dict[str, Any] | None = None
        review_result: dict[str, Any] | None = None
        retry_count = 0
        reviewer_notes = ""

        while retry_count <= MAX_RETRIES:
            if retry_count > 0:
                console.print(f"  [yellow]↻ Retry {retry_count}/{MAX_RETRIES} (reviewer notes injected)[/]")

            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                t = p.add_task(f"  Content strategy agent (attempt {retry_count + 1})...", total=None)
                content_state = run_content_strategy(
                    radar_item=brief_item,
                    reviewer_notes=reviewer_notes,
                    retry_count=retry_count,
                )
                p.update(t, description="  ✓ Content generated. Running compliance review...")
                review_result = run_compliance_review(content_state)
                verdict = review_result.get("verdict", "UNKNOWN")
                p.update(t, description=f"  ✓ Compliance review: {verdict}")

            verdict = review_result.get("verdict", "UNKNOWN")
            verdict_color = "green" if verdict == "PASS" else "yellow"
            console.print(f"  Verdict: [bold {verdict_color}]{verdict}[/]")

            if verdict == "PASS" or review_result.get("_maxRetriesExceeded", False):
                break

            reviewer_notes = review_result.get("reviewerNotes", "")
            retry_count += 1
            content_state["retryCount"] = retry_count

        if review_result is None:
            console.print(f"  [red]✗ Campaign {idx} failed (no result)[/]")
            continue

        pack = review_result.get("pack") or {}
        if not pack.get("content") and content_state:
            pack["agentOutput"] = content_state.get("agentOutput", "")
            pack["campaignId"] = content_state.get("campaignId", f"campaign_{idx}")
            pack["topic"] = content_state.get("topic", "")
            pack["audience"] = content_state.get("audience", "")
            pack["platforms"] = content_state.get("platforms", [])
            pack["radar"] = content_state.get("_radar", {})
        campaign_id = pack.get("campaignId", f"campaign_{idx}")

        images: dict[str, str | None] = {"hero": None, "square": None, "story": None}
        if not dry_run:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
                t = p.add_task("  Generating images (MiniMax via OpenRouter)...", total=None)
                images = run_image_generation(
                    pack=pack,
                    campaign_id=campaign_id,
                    output_dir=OUTPUT_DIR / "images",
                )
                p.update(t, description="  ✓ Images generated")

            generated = sum(1 for v in images.values() if v)
            console.print(f"  Images: [green]{generated}/3 generated[/]")
        else:
            console.print("  [dim]--dry-run: skipping image generation[/]")

        final = {
            **review_result,
            "images": images,
            "generatedAt": datetime.now(UTC).isoformat(),
        }

        out_dir = _save_output(campaign_id, final)
        console.print(f"  [dim]Report saved → {out_dir}[/]")

        results.append(final)
        console.print(f"  [bold green]✓ Campaign {idx} complete[/]")

    # ── Summary ──────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]Pipeline complete![/]\n"
        f"{len(results)}/{len(top5)} campaigns successfully produced.\n"
        f"Output directory: [cyan]{OUTPUT_DIR.resolve()}[/]",
        border_style="green",
    ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenBox Content Studio Pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip image generation (useful for testing)",
    )
    parser.add_argument(
        "--brief-only",
        action="store_true",
        help="Run Phase 1 (Intelligence Radar) only, print briefs and exit",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Run pipeline for a specific user-supplied topic (bypasses RSS)",
    )
    parser.add_argument(
        "--audience",
        type=str,
        default="AI Platform Lead",
        help="Target persona when using --topic",
    )
    args = parser.parse_args()

    try:
        if args.topic:
            result = run_from_prompt(topic=args.topic, audience=args.audience, dry_run=args.dry_run)
            console.print(Panel(
                f"[bold green]Done![/]\nCampaign: {result.get('campaignId')}\n"
                f"Output: [cyan]{OUTPUT_DIR.resolve()}[/]",
                border_style="green",
            ))
            sys.exit(0)
        results = run_pipeline(dry_run=args.dry_run, brief_only=args.brief_only)
        sys.exit(0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline error:[/] {exc}")
        _logger.exception("Unhandled pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    main()
