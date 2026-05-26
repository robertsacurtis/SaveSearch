"""
app.py — Flask web server providing the API for the SaveSearch UI.

Endpoints:
  GET  /api/search?q=...&mode=hybrid&platform=...&limit=20
  GET  /api/stats
  GET  /api/videos?platform=...&limit=100
  POST /api/ingest  { "urls": ["..."] }
  GET  /            → serves index.html

Usage:
  python app.py
  # Then open http://localhost:5000
"""

import sys
import os
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Allow imports from backend/ when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from search import search, get_stats, get_all_videos

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ─── Track ingestion jobs ────────────────────────────────────────────────────────

ingest_jobs: dict[str, dict] = {}
job_lock = threading.Lock()


# ─── Static Files ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ─── Search API ─────────────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    query    = request.args.get("q", "").strip()
    mode     = request.args.get("mode", "hybrid")
    platform = request.args.get("platform") or None
    limit    = int(request.args.get("limit", 20))

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    if mode not in ("keyword", "semantic", "hybrid"):
        return jsonify({"error": "mode must be keyword, semantic, or hybrid"}), 400

    try:
        results = search(query, mode=mode, platform=platform, limit=limit)

        # Sanitize results for JSON
        clean = []
        for r in results:
            clean.append({
                "id":         r.get("id") or "",
                "url":        r.get("url") or "",
                "title":      r.get("title") or "Untitled",
                "creator":    r.get("creator") or "Unknown",
                "platform":   r.get("platform") or "other",
                "thumbnail":  r.get("thumbnail") or "",
                "duration":   r.get("duration") or 0,
                "date_saved": r.get("date_saved") or "",
                "score":      round(r.get("score", 0), 4),
                "match_type": r.get("match_type") or mode,
                # Transcript excerpt for hit highlighting
                "excerpt":    _make_excerpt(r.get("transcript") or "", query),
                "caption_preview": (r.get("caption") or "")[:200],
            })

        return jsonify({"results": clean, "count": len(clean), "query": query})

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "setup_needed": True}), 503
    except Exception as e:
        app.logger.error(f"Search error: {e}")
        return jsonify({"error": "Search failed", "detail": str(e)}), 500


def _make_excerpt(transcript: str, query: str, window: int = 200) -> str:
    """Find and return the most relevant portion of the transcript."""
    if not transcript:
        return ""

    # Find first occurrence of any query word
    words = query.lower().split()
    text_lower = transcript.lower()

    best_pos = len(transcript)
    for word in words:
        pos = text_lower.find(word)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(transcript):
        # No match found, return start
        return transcript[:window].strip() + ("..." if len(transcript) > window else "")

    # Return window around match
    start = max(0, best_pos - 50)
    end = min(len(transcript), start + window)
    excerpt = transcript[start:end].strip()

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(transcript) else ""
    return prefix + excerpt + suffix


# ─── Stats API ──────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    try:
        stats = get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"total": 0, "by_platform": {}, "error": str(e)})


# ─── All Videos API ─────────────────────────────────────────────────────────────

@app.route("/api/videos")
def api_videos():
    platform = request.args.get("platform") or None
    limit    = int(request.args.get("limit", 100))

    try:
        videos = get_all_videos(platform=platform, limit=limit)
        clean = [{
            "id":         v.get("id") or "",
            "url":        v.get("url") or "",
            "title":      v.get("title") or "Untitled",
            "creator":    v.get("creator") or "Unknown",
            "platform":   v.get("platform") or "other",
            "thumbnail":  v.get("thumbnail") or "",
            "duration":   v.get("duration") or 0,
            "date_saved": v.get("date_saved") or "",
            "has_transcript": bool(v.get("transcript")),
        } for v in videos]

        return jsonify({"videos": clean, "count": len(clean)})
    except Exception as e:
        return jsonify({"videos": [], "error": str(e)})


# ─── Ingest API (background job) ────────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """
    Start a background ingestion job for a list of URLs.
    Returns a job ID to poll for progress.
    """
    data = request.json or {}
    urls = data.get("urls", [])
    whisper_model = data.get("whisper_model", "base")

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    import uuid
    job_id = str(uuid.uuid4())[:8]

    with job_lock:
        ingest_jobs[job_id] = {
            "status": "running",
            "total": len(urls),
            "done": 0,
            "errors": 0,
            "current": "",
        }

    def run_ingest():
        # Import here to avoid loading Whisper at startup
        try:
            backend_dir = str(Path(__file__).parent)
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from ingest import init_db, process_url
        except ImportError as ie:
            with job_lock:
                ingest_jobs[job_id]["status"] = "error"
                ingest_jobs[job_id]["error"] = f"ingest module not found: {ie}"
            return

        init_db()
        for url in urls:
            with job_lock:
                ingest_jobs[job_id]["current"] = url

            try:
                process_url(url, whisper_model=whisper_model)
                with job_lock:
                    ingest_jobs[job_id]["done"] += 1
            except Exception as e:
                with job_lock:
                    ingest_jobs[job_id]["errors"] += 1
                app.logger.error(f"Ingest error for {url}: {e}")


        with job_lock:
            ingest_jobs[job_id]["status"] = "done"
            ingest_jobs[job_id]["current"] = ""

    thread = threading.Thread(target=run_ingest, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "message": f"Ingesting {len(urls)} URL(s)"})


@app.route("/api/ingest/<job_id>")
def api_ingest_status(job_id):
    with job_lock:
        job = ingest_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


# ─── Debug ───────────────────────────────────────────────────────────────────────

@app.route("/api/debug")
def api_debug():
    import os
    from pathlib import Path
    data_dir = os.environ.get("DATA_DIR", "data")
    db_path = Path(data_dir) / "videos.db"
    return jsonify({
        "DATA_DIR": data_dir,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "db_size": db_path.stat().st_size if db_path.exists() else 0,
        "cwd": os.getcwd(),
        "files_in_data": [str(f) for f in Path(data_dir).iterdir()] if Path(data_dir).exists() else [],
    })


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  SaveSearch running at http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
