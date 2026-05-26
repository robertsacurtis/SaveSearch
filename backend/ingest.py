"""
ingest.py — Download videos and extract transcripts using faster-whisper.
"""

import json
import argparse
import tempfile
import sqlite3
import threading
from pathlib import Path
import os
from datetime import datetime, timezone
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yt_dlp
except ImportError:
    raise ImportError("Run: pip install yt-dlp")

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("Run: pip install faster-whisper")

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "videos.db"
AUDIO_DIR = DATA_DIR / "audio_temp"

_whisper_model = None
_whisper_lock = threading.Lock()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id          TEXT PRIMARY KEY,
            url         TEXT NOT NULL,
            title       TEXT,
            creator     TEXT,
            platform    TEXT,
            caption     TEXT,
            duration    INTEGER,
            thumbnail   TEXT,
            date_saved  TEXT,
            transcript  TEXT,
            indexed_at  TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts
            USING fts5(id UNINDEXED, title, creator, caption, transcript);
    """)
    conn.commit()
    conn.close()


def save_video(video):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        INSERT OR REPLACE INTO videos
            (id, url, title, creator, platform, caption, duration, thumbnail,
             date_saved, transcript, indexed_at)
        VALUES
            (:id, :url, :title, :creator, :platform, :caption, :duration,
             :thumbnail, :date_saved, :transcript, :indexed_at)
    """, video)
    conn.execute("DELETE FROM videos_fts WHERE id = ?", (video["id"],))
    conn.execute("""
        INSERT INTO videos_fts (id, title, creator, caption, transcript)
        VALUES (:id, :title, :creator, :caption, :transcript)
    """, video)
    conn.commit()
    conn.close()


def video_exists(video_id):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    row = conn.execute(
        "SELECT transcript FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    conn.close()
    return row is not None and row[0] is not None


def detect_platform(url):
    url = url.lower()
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "instagram.com" in url:
        return "instagram"
    if "tiktok.com" in url:
        return "tiktok"
    if "twitter.com" in url or "x.com" in url:
        return "twitter"
    return "other"


def extract_metadata(url):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        print(f"  [!] Metadata extraction failed for {url}: {e}")
        return None


def download_audio(url, output_path):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_template = str(output_path / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "unknown")
            audio_file = output_path / f"{video_id}.mp3"
            if audio_file.exists() and audio_file.stat().st_size > 1000:
                return audio_file
            for ext in ["m4a", "webm", "ogg"]:
                f = output_path / f"{video_id}.{ext}"
                if f.exists() and f.stat().st_size > 1000:
                    return f
    except Exception as e:
        print(f"  [!] Audio download failed: {e}")
    return None


def load_whisper(model_size="base"):
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading faster-whisper model '{model_size}'...")
        # cpu + int8 = fastest CPU mode, no accuracy loss vs tiny
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Model ready.")
    return _whisper_model


def transcribe(audio_path, model_size="base"):
    with _whisper_lock:
        model = load_whisper(model_size)
        segments, info = model.transcribe(str(audio_path), beam_size=5)
        transcript = " ".join(segment.text for segment in segments).strip()
        return transcript


def process_url(url, whisper_model="base", force=False):
    url = url.strip()
    if not url or url.startswith("#"):
        return False

    print(f"\n-> {url}")

    meta = extract_metadata(url)
    if not meta:
        return False

    video_id = meta.get("id", url)
    platform = detect_platform(url)

    if not force and video_exists(video_id):
        print(f"  Already indexed, skipping.")
        return True

    print(f"  Title:    {meta.get('title', 'Unknown')}")
    print(f"  Creator:  {meta.get('uploader', 'Unknown')}")
    print(f"  Downloading audio...")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = download_audio(url, Path(tmp))
        if audio_path is None:
            print(f"  [!] Audio unavailable — saving metadata only.")
            transcript = ""
        else:
            print(f"  Transcribing...")
            try:
                transcript = transcribe(audio_path, whisper_model)
                words = len(transcript.split())
                print(f"  Transcript: {words} words")
            except Exception as e:
                print(f"  [!] Transcription failed: {e}")
                transcript = ""

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id":         video_id,
        "url":        url,
        "title":      meta.get("title") or "",
        "creator":    meta.get("uploader") or meta.get("channel") or "",
        "platform":   platform,
        "caption":    meta.get("description") or "",
        "duration":   meta.get("duration") or 0,
        "thumbnail":  meta.get("thumbnail") or "",
        "date_saved": now,
        "transcript": transcript,
        "indexed_at": now,
    }

    save_video(record)
    print(f"  Saved.")
    return True


def parse_instagram_export(export_path):
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    urls = []
    items = data
    if isinstance(data, dict):
        for key in ["saved_saved_media", "saved_collections", "items"]:
            if key in data:
                items = data[key]
                break
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                url = _find_url_in_dict(item)
                if url:
                    urls.append(url)
    print(f"Found {len(urls)} URLs in Instagram export.")
    return urls


def _find_url_in_dict(d, depth=0):
    if depth > 5:
        return None
    for key, val in d.items():
        if key in ("href", "url", "uri") and isinstance(val, str):
            if val.startswith("http"):
                return val
        if isinstance(val, dict):
            result = _find_url_in_dict(val, depth + 1)
            if result:
                return result
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    result = _find_url_in_dict(item, depth + 1)
                    if result:
                        return result
    return None


def main():
    parser = argparse.ArgumentParser(description="Ingest and transcribe saved videos.")
    parser.add_argument("--url", help="Single video URL to process")
    parser.add_argument("--urls", help="Path to a text file of URLs (one per line)")
    parser.add_argument("--instagram-export", help="Path to Instagram saved_posts.json")
    parser.add_argument("--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size. 'base' with int8 is fast and accurate.")
    parser.add_argument("--force", action="store_true",
        help="Re-process URLs already in the database")

    args = parser.parse_args()

    if not any([args.url, args.urls, args.instagram_export]):
        parser.print_help()
        return

    init_db()
    urls = []

    if args.url:
        urls.append(args.url)

    if args.urls:
        with open(args.urls) as f:
            urls.extend(line.strip() for line in f if line.strip())

    if args.instagram_export:
        urls.extend(parse_instagram_export(args.instagram_export))

    print(f"\nProcessing {len(urls)} URL(s) with faster-whisper '{args.whisper_model}'...")
    print("Downloads run in parallel. Transcription queued with lock.\n")

    load_whisper(args.whisper_model)

    success = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_url, url, args.whisper_model, args.force): url
            for url in urls
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall progress", unit="video"):
            try:
                if future.result():
                    success += 1
            except Exception as e:
                print(f"  [!] Error: {e}")

    print(f"\nDone. {success}/{len(urls)} videos indexed successfully.")


if __name__ == "__main__":
    main()
