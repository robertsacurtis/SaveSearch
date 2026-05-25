"""
search.py — Unified search engine combining keyword (FTS5) and semantic (vector) search.

Two modes:
  - Keyword:  Fast exact matching using SQLite FTS5. Good for names, specific terms.
  - Semantic: Meaning-based using ChromaDB + sentence-transformers.
              Good for topics, concepts, "videos about X".
  - Hybrid:   Both, with result merging and deduplication (default).

Usage (as a module):
  from search import search
  results = search("meal prep tips", mode="hybrid", limit=10)

Usage (CLI):
  python search.py "meal prep tips"
  python search.py "stoicism" --mode semantic
  python search.py "david goggins" --mode keyword --platform youtube
"""

import sqlite3
import argparse
import json
from pathlib import Path
from typing import Literal

DB_PATH = Path("data/videos.db")
CHROMA_DIR = Path("data/chroma")


# ─── Database Connection ────────────────────────────────────────────────────────

def get_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run ingest.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── ChromaDB Connection ────────────────────────────────────────────────────────

_chroma_collection = None

def get_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    if not CHROMA_DIR.exists():
        return None

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        _chroma_collection = client.get_collection(
            name="videos",
            embedding_function=ef,
        )
        return _chroma_collection
    except Exception as e:
        print(f"[!] Could not load vector index: {e}")
        return None


# ─── Keyword Search ──────────────────────────────────────────────────────────────

def keyword_search(query: str, platform: str | None = None,
                   limit: int = 20) -> list[dict]:
    """
    Full-text search using SQLite FTS5.
    Searches title, creator, caption, and transcript simultaneously.
    """
    conn = get_db()

    # FTS5 supports prefix queries (*), phrase queries ("..."), AND/OR/NOT
    # We make the query prefix-friendly for partial word matches
    fts_query = " OR ".join(
        f'"{word}"*' for word in query.split() if word
    )

    sql = """
        SELECT
            v.id, v.url, v.title, v.creator, v.platform,
            v.caption, v.duration, v.thumbnail, v.date_saved, v.transcript,
            rank
        FROM videos v
        JOIN videos_fts fts ON v.id = fts.id
        WHERE videos_fts MATCH ?
    """
    params = [fts_query]

    if platform:
        sql += " AND v.platform = ?"
        params.append(platform)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
        results = [dict(row) for row in rows]

        # Add score and mode tag
        for i, r in enumerate(results):
            r["score"] = 1.0 - (i / max(len(results), 1)) * 0.3
            r["match_type"] = "keyword"

        return results
    except sqlite3.OperationalError as e:
        print(f"[!] FTS error (query: {fts_query!r}): {e}")
        return []
    finally:
        conn.close()


# ─── Semantic Search ─────────────────────────────────────────────────────────────

def semantic_search(query: str, platform: str | None = None,
                    limit: int = 20) -> list[dict]:
    """
    Vector similarity search using ChromaDB.
    Finds videos that are conceptually related to the query,
    even if they don't contain the exact words.
    """
    collection = get_collection()
    if collection is None:
        return []

    where = {}
    if platform:
        where["platform"] = {"$eq": platform}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(limit, collection.count()),
            where=where if where else None,
            include=["metadatas", "distances", "documents"],
        )
    except Exception as e:
        print(f"[!] Semantic search error: {e}")
        return []

    if not results["ids"] or not results["ids"][0]:
        return []

    # Fetch full records from SQLite (ChromaDB only stores metadata)
    conn = get_db()
    ids = results["ids"][0]
    distances = results["distances"][0]

    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM videos WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()

    # Map by id for ordering
    row_map = {row["id"]: dict(row) for row in rows}

    output = []
    for vid_id, distance in zip(ids, distances):
        if vid_id not in row_map:
            continue
        record = row_map[vid_id]
        # Convert cosine distance to a 0-1 similarity score
        record["score"] = round(1.0 - distance, 4)
        record["match_type"] = "semantic"
        output.append(record)

    return output


# ─── Hybrid Search ───────────────────────────────────────────────────────────────

def hybrid_search(query: str, platform: str | None = None,
                  limit: int = 20) -> list[dict]:
    """
    Combine keyword and semantic results, deduplicate, and re-rank.

    Scoring:
      - Keyword match: base score from FTS rank
      - Semantic match: cosine similarity score
      - Appears in both: bonus (reciprocal rank fusion inspired)
    """
    keyword_results = keyword_search(query, platform=platform, limit=limit)
    semantic_results = semantic_search(query, platform=platform, limit=limit)

    # Build score map
    scores: dict[str, dict] = {}

    for i, r in enumerate(keyword_results):
        vid_id = r["id"]
        scores[vid_id] = r.copy()
        scores[vid_id]["_kw_score"] = r["score"]
        scores[vid_id]["_sem_score"] = 0.0

    for i, r in enumerate(semantic_results):
        vid_id = r["id"]
        if vid_id in scores:
            scores[vid_id]["_sem_score"] = r["score"]
            scores[vid_id]["match_type"] = "both"
        else:
            scores[vid_id] = r.copy()
            scores[vid_id]["_kw_score"] = 0.0
            scores[vid_id]["_sem_score"] = r["score"]

    # Final score: weighted combination
    for vid_id, record in scores.items():
        kw = record.get("_kw_score", 0.0)
        sem = record.get("_sem_score", 0.0)
        # If it shows up in both, it's very likely relevant
        bonus = 0.15 if record.get("match_type") == "both" else 0.0
        record["score"] = round(kw * 0.4 + sem * 0.6 + bonus, 4)

    # Sort by combined score
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


# ─── Public API ─────────────────────────────────────────────────────────────────

def search(
    query: str,
    mode: Literal["keyword", "semantic", "hybrid"] = "hybrid",
    platform: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Main search function.

    Args:
        query:    Search query string
        mode:     'keyword' | 'semantic' | 'hybrid'
        platform: Filter by platform ('youtube', 'instagram', 'tiktok', etc.)
        limit:    Maximum results to return

    Returns:
        List of video dicts sorted by relevance score (descending)
    """
    if not query.strip():
        return []

    if mode == "keyword":
        return keyword_search(query, platform=platform, limit=limit)
    elif mode == "semantic":
        return semantic_search(query, platform=platform, limit=limit)
    else:
        return hybrid_search(query, platform=platform, limit=limit)


def get_stats() -> dict:
    """Return database statistics."""
    stats = {"total": 0, "by_platform": {}, "has_transcript": 0, "vector_indexed": 0}

    try:
        conn = get_db()
        stats["total"] = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        stats["has_transcript"] = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE transcript != '' AND transcript IS NOT NULL"
        ).fetchone()[0]

        rows = conn.execute(
            "SELECT platform, COUNT(*) as n FROM videos GROUP BY platform"
        ).fetchall()
        stats["by_platform"] = {r["platform"]: r["n"] for r in rows}
        conn.close()
    except Exception:
        pass

    try:
        col = get_collection()
        if col:
            stats["vector_indexed"] = col.count()
    except Exception:
        pass

    return stats


def get_all_videos(platform: str | None = None, limit: int = 100) -> list[dict]:
    """Return all videos, optionally filtered by platform."""
    try:
        conn = get_db()
        if platform:
            rows = conn.execute(
                "SELECT * FROM videos WHERE platform = ? ORDER BY date_saved DESC LIMIT ?",
                (platform, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM videos ORDER BY date_saved DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Search your saved videos.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--mode", default="hybrid",
        choices=["keyword", "semantic", "hybrid"])
    parser.add_argument("--platform", help="Filter by platform (youtube, instagram, etc.)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results = search(args.query, mode=args.mode, platform=args.platform, limit=args.limit)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\n{len(results)} result(s) for '{args.query}' [{args.mode}]\n")
    print("─" * 60)

    for i, r in enumerate(results, 1):
        duration = r.get("duration") or 0
        mins = duration // 60
        secs = duration % 60
        print(f"\n{i}. {r.get('title') or 'Untitled'}")
        print(f"   Creator:   {r.get('creator') or 'Unknown'}")
        print(f"   Platform:  {r.get('platform') or '?'}")
        print(f"   Duration:  {mins}:{secs:02d}")
        print(f"   Score:     {r.get('score', 0):.3f} ({r.get('match_type', '?')})")
        print(f"   URL:       {r.get('url') or '?'}")

        transcript = r.get("transcript") or ""
        if transcript:
            preview = transcript[:200].replace("\n", " ")
            print(f"   Excerpt:   \"{preview}...\"")


if __name__ == "__main__":
    main()
