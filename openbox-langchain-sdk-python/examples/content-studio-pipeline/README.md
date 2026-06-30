# OpenBox Content Studio Pipeline

Python SDK implementation of the **Content Studio Workflow** — a fully governed AI content pipeline for OpenBox AI.

## Architecture

```
Phase 1 — Intelligence Radar
─────────────────────────────────────────────────────────────────────
ingestion.py        → fetch 9 RSS/API sources, dedup
gate_agent.py       → LangChain Agent + OpenBox governance (ICP filter)
                    → LangChain Agent + OpenBox governance (signal extraction)
                    → LangChain Agent + OpenBox governance (score & top 5 briefs)

Phase 2 — Content Studio (per brief, up to 2 retries)
─────────────────────────────────────────────────────────────────────
content_strategy_agent.py  → LangChain Agent + OpenBox governance
                              Claude 3.5 Haiku via OpenRouter
                              Wikipedia tool for fact-checking
                              Full publishing pack (LinkedIn, X, Instagram,
                              Facebook, Threads, Blog, SEO, Image prompts)

compliance_review_graph.py → LangGraph StateGraph + OpenBox governance
                              Gemini 2.5 Pro via OpenRouter
                              4-check review: copyright, brand safety, SEO, platform
                              PASS → proceed | NEEDS_REVISION → retry loop

image_agent.py             → LangChain Agent (censorship gate) + OpenBox governance
                              MiniMax-Image-01 via OpenRouter (actual images saved)
                              3 formats: hero 1792×1024, square 1024×1024, story 1024×1792

Phase 3 — Final Report
─────────────────────────────────────────────────────────────────────
pipeline.py         → Markdown + JSON report saved to ./output/<campaign_id>/
```

## Governance Layer (OpenBox SDK)

Every agent in the pipeline is wrapped in `OpenBoxLangChainMiddleware`:

- **Audit trail**: Every LLM call, tool call, and agent lifecycle event is recorded as a cryptographically attested span in OpenBox's Merkle audit trail
- **Guardrails**: Pre-screen of user prompts via OPA/Rego behavioral rules
- **HITL**: Human-in-the-loop approval gates (configurable per agent)
- **Trust Score**: Real-time trustworthiness rating per agent action
- **Session tracking**: Per-campaign session IDs enable full session replay

## Setup

```bash
# Install dependencies
pip install -e "../../"   # install openbox-langchain-sdk-python
pip install -r requirements.txt
# or with uv:
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your actual keys
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key (Claude, DeepSeek, Gemini, MiniMax) |
| `OPENBOX_API_KEY` | OpenBox governance API key (`obx_live_...`) |
| `OPENBOX_API_URL` | OpenBox API URL (default: `https://core.openbox.ai`) |
| `OPENBOX_AGENT_DID` | OpenBox agent DID for cryptographic attestation |
| `OPENBOX_AGENT_PRIVATE_KEY` | Ed25519 private key seed for signing |
| `OUTPUT_DIR` | Output directory for reports and images (default: `./output`) |

## Usage

```bash
# Full pipeline (all 3 phases + image generation)
python pipeline.py

# Skip image generation (faster for testing)
python pipeline.py --dry-run

# Phase 1 only — just run the radar and print briefs
python pipeline.py --brief-only
```

## Output

Each campaign produces a folder at `./output/<campaign_id>/`:

```
output/
└── radar_1_1234567890/
    ├── report.json      # Full structured output
    ├── report.md        # Human-readable Markdown report
    └── images/          # (sibling directory)
        ├── radar_1_1234567890_hero.png     (1792×1024)
        ├── radar_1_1234567890_square.png   (1024×1024)
        └── radar_1_1234567890_story.png    (1024×1792)
```

## Module Map (n8n → Python)

| n8n Node | Python Module | Agent Type |
|---|---|---|
| `OpenBox: Agent` (Gate) | `gate_agent.run_gate_agent()` | LangChain + OpenBox |
| `Signal Extractor` | `gate_agent.run_signal_extractor()` | LangChain + OpenBox |
| `Score Top 5` | `gate_agent.run_scorer()` | LangChain + OpenBox |
| `Content Governance` | `content_strategy_agent.run_content_strategy()` | LangChain + OpenBox |
| `Compliance + SEO Reviewer` | `compliance_review_graph.review_node` | LangGraph + OpenBox |
| `Verdict Extractor` | `compliance_review_graph.verdict_node` | LangGraph (pure) |
| `Publishing Pack Assembler` | `compliance_review_graph.assemble_node` | LangGraph (pure) |
| `Image Content Governance` | `image_agent.run_image_censorship_check()` | LangChain + OpenBox |
| `Image Generation` | `image_agent.run_image_generation()` | MiniMax via OpenRouter |
