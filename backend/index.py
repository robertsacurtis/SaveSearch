"""
index.py — Build and manage the semantic search index using ChromaDB.

This runs AFTER ingest.py has populated the SQLite database.
It takes all transcripts + metadata and creates vector embeddings
so you can search by meaning, not just keywords.

Usage:
  python index.py           # index everything not yet indexed
  python index.py --rebuild # wipe and rebuild from scratch
"""

import sqlite3
import argparse
from pathlib import Path
import os
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "videos.db"
CHROMA_DIR = DATA_DIR / "chroma"

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError("Run: pip install chromadb")


# ─── ChromaDB Setup ─────────────────────────────────────────────────────────────

def get_chroma_collection(reset: bool = False):
    """Get or create the ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Use the lightweight all-MiniLM model — fast, good quality, runs locally
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    if reset:
        try:
            client.delete_collection("videos")
            print("Cleared existing vector index.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="videos",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# ─── Document Building ──────────────────────────────────────────────────────────

def build_document(video: dict) -> str:
    """
    Combine all text fields into one document for embedding.
    Weight title and caption higher by repeating them.
    """
    parts = []

    title = video.get("title") or ""
    creator = video.get("creator") or ""
    caption = video.get("caption") or ""
    transcript = video.get("transcript") or ""

    if title:
        parts.append(f"Title: {title}")
        parts.append(title)  # repeat for weight

    if creator:
        parts.append(f"Creator: {creator}")

    if caption:
        # Truncate captions — they can be very long
        cap_preview = caption[:500]
        parts.append(f"Description: {cap_preview}")

    if transcript:
        # Truncate transcript for embedding (model has token limits)
        # Full transcript is still in SQLite for keyword search
        transcript_preview = transcript[:2000]
        parts.append(f"Transcript: {transcript_preview}")

    return "\n".join(parts)


# ─── Indexing ───────────────────────────────────────────────────────────────────

def get_indexed_ids(collection) -> set[str]:
    """Get all IDs currently in ChromaDB."""
    try:
        result = collection.get(include=[])
        return set(result["ids"])
    except Exception:
        return set()


def index_all(reset: bool = False):
    """Read from SQLite, embed, and store in ChromaDB."""
    if not DB_PATH.exists():
        print("No database found. Run ingest.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    videos = conn.execute("""
        SELECT id, url, title, creator, platform, caption,
               duration, thumbnail, date_saved, transcript
        FROM videos
        ORDER BY date_saved DESC
    """).fetchall()

    if not videos:
        print("No videos in database. Run ingest.py first.")
        conn.close()
        return

    print(f"Found {len(videos)} videos in database.")

    collection = get_chroma_collection(reset=reset)
    already_indexed = get_indexed_ids(collection)

    to_index = [v for v in videos if v["id"] not in already_indexed]

    if not to_index:
        print(f"All {len(videos)} videos already indexed. Use --rebuild to force.")
        conn.close()
        return

    print(f"Indexing {len(to_index)} new videos...")

    # Process in batches (ChromaDB handles large batches fine but this shows progress)
    BATCH_SIZE = 50

    for i in tqdm(range(0, len(to_index), BATCH_SIZE), desc="Embedding batches"):
        batch = to_index[i : i + BATCH_SIZE]

        ids = []
        documents = []
        metadatas = []

        for video in batch:
            v = dict(video)
            ids.append(v["id"])
            documents.append(build_document(v))
            metadatas.append({
                "url":       v["url"] or "",
                "title":     v["title"] or "",
                "creator":   v["creator"] or "",
                "platform":  v["platform"] or "",
                "thumbnail": v["thumbnail"] or "",
                "duration":  v["duration"] or 0,
                "date_saved": v["date_saved"] or "",
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    count = collection.count()
    print(f"\n✓ Vector index now contains {count} videos.")
    conn.close()


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build semantic search index from ingested videos."
    )
    parser.add_argument("--rebuild", action="store_true",
        help="Wipe and rebuild the entire vector index from scratch")

    args = parser.parse_args()
    index_all(reset=args.rebuild)


if __name__ == "__main__":
    main()
