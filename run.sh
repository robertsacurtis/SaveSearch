#!/usr/bin/env bash
# run.sh — One-shot setup and launch for SaveSearch
# Usage:
#   chmod +x run.sh
#   ./run.sh              # install + start server
#   ./run.sh ingest       # ingest URLs from urls.txt
#   ./run.sh index        # build semantic index
#   ./run.sh search "query"

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DATA_DIR="$PROJECT_DIR/data"

# ── Colors ──────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${BLUE}▸ $1${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }

# ── Setup venv if needed ─────────────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  ok "Virtual environment created."
fi

source "$VENV_DIR/bin/activate"

log "Installing / checking dependencies..."
pip install -q -r "$PROJECT_DIR/requirements.txt"

# Check ffmpeg (required by whisper)
if ! command -v ffmpeg &>/dev/null; then
  warn "ffmpeg not found — required for audio extraction."
  warn "Install with: brew install ffmpeg  (macOS) or apt install ffmpeg (Linux)"
fi

mkdir -p "$DATA_DIR"

# ── Commands ─────────────────────────────────────────────────────────────────────

case "${1:-server}" in

  ingest)
    shift
    log "Running ingestion pipeline..."
    cd "$PROJECT_DIR/backend"
    python ingest.py "$@"
    ;;

  index)
    log "Building semantic search index..."
    cd "$PROJECT_DIR/backend"
    python index.py "$@"
    ;;

  search)
    shift
    cd "$PROJECT_DIR/backend"
    python search.py "$@"
    ;;

  server|"")
    log "Starting SaveSearch server..."
    echo ""
    echo "  Open http://localhost:5000 in your browser"
    echo ""
    cd "$PROJECT_DIR/backend"
    python app.py
    ;;

  *)
    echo "Usage: ./run.sh [ingest|index|search|server]"
    echo ""
    echo "  server               Start the web interface (default)"
    echo "  ingest --url URL     Ingest a single video"
    echo "  ingest --urls FILE   Ingest from a text file of URLs"
    echo "  index                Build/update semantic search index"
    echo "  search 'query'       CLI search"
    ;;

esac
