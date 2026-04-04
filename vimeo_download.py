#!/usr/bin/env python3
"""
Vimeo video downloader and transcriber.
Downloads a Vimeo video in high quality and transcribes it to English.

Usage:
    python vimeo_download.py <vimeo_url>
    python vimeo_download.py <vimeo_url> --output-dir /path/to/dir
    python vimeo_download.py <vimeo_url> --model large
"""

import argparse
import datetime
import json
import os
import re
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def log(msg: str, *, file=None) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", file=file)


def resolve_url(url: str) -> tuple[str, str | None]:
    """
    Resolve a Vimeo review URL to a player URL that yt-dlp can handle.

    Returns (resolved_url, referer) where referer is the original page URL
    needed for embed-only player URLs, or None for direct URLs.

    Handles two URL formats:
    - Old style: vimeo.com/{user}/review/{id}/{hash}?version=1
    - New style: vimeo.com/reviews/{uuid}/videos/{id}

    The new-style review URLs are not matched by yt-dlp's VimeoReviewIE,
    so we fetch the review page (a Next.js app that embeds the full video
    config in a __NEXT_DATA__ JSON blob) and extract the player URL
    including the private hash token.
    """
    if "/reviews/" in url:
        # New-style review URL — must fetch the page to get the hash token
        return _resolve_reviews_url(url), url
    elif "/review/" in url:
        # Old-style review URL — extract video_id/hash from path
        # Format: vimeo.com/{user}/review/{video_id}/{hash}?version=...
        # Use player.vimeo.com embed URL so yt-dlp passes the hash correctly
        parts = url.split('?')[0].rstrip('/').split('/')
        # Find "review" in the path and grab the two segments after it
        try:
            idx = parts.index('review')
            video_id = parts[idx + 1]
            video_hash = parts[idx + 2]
            return f"https://player.vimeo.com/video/{video_id}?h={video_hash}", url
        except (ValueError, IndexError):
            return url, None
    return url, None


def _resolve_reviews_url(url: str) -> str:
    """Fetch a new-style Vimeo review page and extract the player URL."""
    log("Resolving Vimeo review URL…")
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
    except Exception as exc:
        log(f"Warning: could not fetch review page ({exc}), trying URL as-is", file=sys.stderr)
        return url

    # Try __NEXT_DATA__ JSON blob first
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group(1))
            player_match = re.search(
                r'player\.vimeo\.com/video/(\d+)\?h=([0-9a-f]+)',
                json.dumps(raw),
            )
            if player_match:
                player_url = f"https://player.vimeo.com/video/{player_match.group(1)}?h={player_match.group(2)}"
                log(f"Resolved to: {player_url}")
                return player_url
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: scan the raw HTML for the player URL
    player_match = re.search(r'player\.vimeo\.com/video/(\d+)\?h=([0-9a-f]+)', html)
    if player_match:
        player_url = f"https://player.vimeo.com/video/{player_match.group(1)}?h={player_match.group(2)}"
        log(f"Resolved to: {player_url}")
        return player_url

    log("Warning: could not extract player hash from review page, trying URL as-is", file=sys.stderr)
    return url
    

def download_video(url: str, output_dir: Path) -> Path:
    """Download a Vimeo video using yt-dlp in the best available quality."""
    resolved_url, referer = resolve_url(url)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--replace-in-metadata", "title", r"[^a-zA-Z0-9]+", "-",
        "--output", output_template,
        "--no-playlist",
        "--print", "after_move:filepath",
    ]
    if referer:
        cmd += ["--referer", referer]
    cmd.append(resolved_url)

    log(f"Downloading: {resolved_url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Error downloading video:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # yt-dlp prints the final filepath via --print after_move:filepath
    filepath = result.stdout.strip().splitlines()[-1]
    video_path = Path(filepath)

    if not video_path.exists():
        # Fall back: find the most recently modified mp4 in output_dir
        mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4_files:
            log("Could not locate downloaded video file.", file=sys.stderr)
            sys.exit(1)
        video_path = mp4_files[0]

    # Strip any leading/trailing hyphens left after sanitisation (e.g. "-title-.mp4" → "title.mp4")
    clean_stem = video_path.stem.strip("-")
    if clean_stem != video_path.stem:
        clean_path = video_path.with_name(clean_stem + video_path.suffix)
        video_path.rename(clean_path)
        video_path = clean_path

    log(f"Downloaded: {video_path}")
    return video_path


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video as a 16 kHz mono WAV for Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]

    log("Extracting audio…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Error extracting audio:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def transcribe(audio_path: Path, model_name: str, output_dir: Path, stem: str | None = None) -> Path:
    """Transcribe audio to English using OpenAI Whisper."""
    import whisper

    output_stem = stem if stem is not None else audio_path.stem

    log(f"Loading Whisper model '{model_name}'…")
    model = whisper.load_model(model_name)

    log("Transcribing (this may take a while)…")
    result = model.transcribe(str(audio_path), language="en", task="transcribe")

    transcript_path = output_dir / (output_stem + ".txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
        f.write("\n")

    # Also write an SRT subtitle file
    srt_path = output_dir / (output_stem + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = _format_srt_time(segment["start"])
            end = _format_srt_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    return transcript_path


def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def download_description(url: str, output_dir: Path) -> Path:
    """Download the video description and save it as a markdown file."""
    resolved, referer = resolve_url(url)
    cmd = ["yt-dlp", "--print", "description"]
    if referer:
        cmd += ["--referer", referer]
    cmd.append(resolved)

    log(f"Fetching description: {resolved}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Error fetching description:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    description = result.stdout.strip()
    # Convert to markdown: replace leading asterisks with hyphens for list items
    lines = description.splitlines()
    for i, line in enumerate(lines):
        if re.match(r'^(\s*)\*', line):
            lines[i] = re.sub(r'^(\s*)\*', r'\1-', line)
    description = '\n'.join(lines)
    # Extract YYYYMMDD date from the output directory name
    match = re.search(r'(\d{8})', output_dir.name)
    if not match:
        log("Error: could not extract a YYYYMMDD date from the output directory name.", file=sys.stderr)
        sys.exit(1)
    date_str = match.group(1)
    desc_path = output_dir / f"description-{date_str}.md"

    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(description)
        f.write("\n")

    log(f"Description saved: {desc_path}")
    return desc_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Vimeo video and transcribe it to English.")
    parser.add_argument("url", help="Vimeo video URL")
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to save the video and transcript (default: current directory)",
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the extracted audio WAV file after transcription.",
    )
    parser.add_argument(
        "-desc", "--description",
        action="store_true",
        help="Download the video description and save as a markdown file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download description if requested
    desc_path = None
    if args.description:
        desc_path = download_description(args.url, output_dir)

    # Download
    video_path = download_video(args.url, output_dir)

    # Extract audio to a temp file (or output dir if --keep-audio)
    if args.keep_audio:
        audio_path = output_dir / (video_path.stem + ".wav")
        extract_audio(video_path, audio_path)
        transcript_path = transcribe(audio_path, args.model, output_dir, stem=video_path.stem)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)
        try:
            extract_audio(video_path, audio_path)
            transcript_path = transcribe(audio_path, args.model, output_dir, stem=video_path.stem)
        finally:
            audio_path.unlink(missing_ok=True)

    log("Done!")
    log(f"  Video:      {video_path}")
    log(f"  Transcript: {transcript_path}")
    srt_path = transcript_path.with_suffix(".srt")
    if srt_path.exists():
        log(f"  Subtitles:  {srt_path}")
    if desc_path:
        log(f"  Description: {desc_path}")


if __name__ == "__main__":
    main()
