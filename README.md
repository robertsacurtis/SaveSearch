# SaveSearch — Video Memory Engine

Search your saved videos by topic, keyword, or concept — across YouTube, Instagram, TikTok, and more.

## How it works

```
Your saved URLs
      ↓
  yt-dlp          ← downloads audio from almost any platform
      ↓
  Whisper         ← transcribes speech to text (runs locally, free)
      ↓
  SQLite FTS5     ← keyword search across titles, captions, transcripts
  ChromaDB        ← semantic/concept search via vector embeddings
      ↓
  Web UI          ← search everything at http://localhost:5000
```

---

## Requirements

- Python 3.10+
- ffmpeg (for audio extraction)

Install ffmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

---

## Setup

```bash
git clone <this-repo>
cd savesearch
chmod +x run.sh
./run.sh          # installs deps + starts server
```

The first run installs all Python packages automatically.

---

## Step 1 — Ingest videos

### Single URL
```bash
./run.sh ingest --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Bulk from a text file
Create `urls.txt` with one URL per line:
```
https://www.youtube.com/watch?v=...
https://www.tiktok.com/@user/video/...
https://www.instagram.com/reel/...
```
Then run:
```bash
./run.sh ingest --urls urls.txt
```

### From Instagram data export
1. Go to Instagram → Settings → Your activity → Download your information
2. Choose JSON format, select "Saved posts"
3. Download and unzip
4. Run:
```bash
./run.sh ingest --instagram-export path/to/saved_posts.json
```

### Whisper model sizes (accuracy vs speed tradeoff)
| Model  | Size  | Speed  | Accuracy |
|--------|-------|--------|----------|
| tiny   | 75MB  | ~32x   | Good     |
| base   | 145MB | ~16x   | Better   |
| small  | 466MB | ~6x    | Great    |
| medium | 1.5GB | ~2x    | Excellent |

Default is `base`. For better accuracy:
```bash
./run.sh ingest --urls urls.txt --whisper-model small
```

---

## Step 2 — Build semantic index

After ingesting, run once to build the vector search index:
```bash
./run.sh index
```

Re-run anytime you add new videos. Only new videos are re-indexed.

---

## Step 3 — Search

### Web UI
```bash
./run.sh server
# Open http://localhost:5000
```

### CLI
```bash
./run.sh search "morning routine productivity"
./run.sh search "stoic philosophy" --mode semantic
./run.sh search "david goggins" --mode keyword --platform youtube
```

---

## Search modes

| Mode     | What it does |
|----------|-------------|
| **Hybrid** | Combines keyword + semantic — best for most queries |
| **Semantic** | Finds conceptually related videos even without exact words. "videos about dealing with failure" works even if those words never appear |
| **Keyword** | Exact word matching. Best for names, specific terms, quotes |

---

## Project structure

```
savesearch/
├── backend/
│   ├── ingest.py     # Download + transcribe videos
│   ├── index.py      # Build semantic vector index
│   ├── search.py     # Search engine (keyword + semantic)
│   └── app.py        # Flask web server
├── frontend/
│   └── index.html    # Web UI
├── data/             # Created automatically
│   ├── videos.db     # SQLite: all video records + FTS index
│   └── chroma/       # ChromaDB: vector embeddings
├── requirements.txt
└── run.sh
```

---

## Tips

- **No transcript?** Some videos are music or have no speech. The title and caption are still searchable.
- **Instagram private videos** — yt-dlp can download them if you're logged in. Pass cookies via `--cookies-from-browser chrome` in yt-dlp options inside ingest.py.
- **Slow transcription?** Use `--whisper-model tiny` for speed, or run on a GPU (Whisper auto-detects CUDA).
- **Re-index after adding videos:** `./run.sh index` (no `--rebuild` needed unless something broke).
- **Force re-process a URL:** `./run.sh ingest --url URL --force`

---

## Platform support

| Platform  | Download | Transcript | Notes |
|-----------|----------|------------|-------|
| YouTube   | ✅        | ✅          | Full API support |
| TikTok    | ✅        | ✅          | Public videos only |
| Instagram | ⚠️        | ✅          | Reels work; saved posts need data export |
| Twitter/X | ✅        | ✅          | Public videos |
| Other     | ✅        | ✅          | Any yt-dlp compatible site |
