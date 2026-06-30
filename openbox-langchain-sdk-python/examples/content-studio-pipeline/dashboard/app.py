"""
OpenBox Content Studio — Chat Dashboard
Run: python dashboard/app.py
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

BASE = Path(__file__).parent.parent
OUTPUT_DIR = BASE / "output"
IMAGES_DIR = OUTPUT_DIR / "images"

app = Flask(__name__, template_folder="templates", static_folder="static")

# Per-run SSE queues: run_id -> Queue
_run_queues: dict[str, queue.Queue] = {}
_run_results: dict[str, dict] = {}


def _load_campaign(folder: str) -> dict | None:
    path = OUTPUT_DIR / folder / "report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_pack_sections(data: dict) -> dict:
    """Pull flat sections out of rawAgentOutput if pack.content is empty."""
    pack = data.get("pack") or {}
    raw = pack.get("rawAgentOutput") or data.get("agentOutput", "")
    if not raw:
        return pack

    import re
    SECTIONS = [
        "UNIFIED_BRIEF", "LINKEDIN", "X", "INSTAGRAM", "FACEBOOK", "THREADS",
        "BLOG_TITLE", "BLOG_SUMMARY", "CTA", "HASHTAGS",
        "IMAGE_PROMPT_HERO", "IMAGE_PROMPT_SQUARE", "IMAGE_PROMPT_STORY",
        "SEO_TITLE", "META_DESCRIPTION", "KEYWORDS", "POSTING_SCHEDULE",
    ]
    positions = []
    for label in SECTIONS:
        m = re.search(
            r"(?:^|\n)(?:#{1,3}\s+)?\*{0,2}" + re.escape(label) + r"\*{0,2}:?\s*",
            raw, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            positions.append((label, m.start(), m.end()))
    positions.sort(key=lambda x: x[1])
    f: dict[str, str] = {}
    for k, (label, _s, end) in enumerate(positions):
        stop = positions[k + 1][1] if k + 1 < len(positions) else len(raw)
        f[label] = raw[end:stop].strip()

    pack.update(f)
    pack["rawAgentOutput"] = raw
    return pack


def _run_pipeline_stream(run_id: str, topic: str, audience: str, dry_run: bool) -> None:
    """Background thread: runs pipeline subprocess and streams output via SSE queue."""
    q = _run_queues[run_id]

    def emit(event: str, data: str) -> None:
        q.put(f"event: {event}\ndata: {json.dumps(data)}\n\n")

    args = [sys.executable, str(BASE / "pipeline.py"),
            "--topic", topic, "--audience", audience]
    if dry_run:
        args.append("--dry-run")

    emit("status", "🚀 Starting pipeline...")

    try:
        proc = subprocess.Popen(
            args,
            cwd=str(BASE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )

        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                emit("log", line)

        proc.wait(timeout=600)

        # Find the most recently written report (any folder)
        candidates = sorted(
            OUTPUT_DIR.glob("*/report.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            data = json.loads(candidates[0].read_text(encoding="utf-8"))
            pack = _extract_pack_sections(data)
            data["pack"] = pack
            _run_results[run_id] = data
            emit("done", candidates[0].parent.name)
        else:
            emit("error", "Pipeline finished but no report found.")

    except subprocess.TimeoutExpired:
        proc.kill()
        emit("error", "Pipeline timed out after 10 minutes.")
    except Exception as exc:
        emit("error", str(exc))
    finally:
        q.put(None)  # sentinel


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start a pipeline run, return run_id for SSE polling."""
    body = request.get_json(force=True)
    topic = (body.get("topic") or "").strip()
    audience = body.get("audience") or "AI Platform Lead"
    dry_run = bool(body.get("dry_run", True))

    if not topic:
        return jsonify({"error": "topic is required"}), 400

    run_id = str(uuid.uuid4())
    _run_queues[run_id] = queue.Queue()

    t = threading.Thread(target=_run_pipeline_stream,
                         args=(run_id, topic, audience, dry_run), daemon=True)
    t.start()

    return jsonify({"run_id": run_id})


@app.route("/api/stream/<run_id>")
def api_stream(run_id: str):
    """SSE endpoint — streams log lines and final done/error event."""
    if run_id not in _run_queues:
        return jsonify({"error": "unknown run_id"}), 404

    def generate():
        q = _run_queues[run_id]
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        _run_queues.pop(run_id, None)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/result/<folder>")
def api_result(folder: str):
    """Return the full parsed report for a campaign folder."""
    data = _load_campaign(folder)
    if not data:
        return jsonify({"error": "not found"}), 404
    data["pack"] = _extract_pack_sections(data)
    return jsonify(data)


@app.route("/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(str(IMAGES_DIR), filename)


if __name__ == "__main__":
    print(f"\nOpenBox Content Studio Chat")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Dashboard  : http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
