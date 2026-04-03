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


def resolve_url(url: str) -> str:
    """
    Resolve a Vimeo review URL to a player URL that yt-dlp can handle.

    The new-style review URLs (vimeo.com/reviews/{uuid}/videos/{id}) are not
    matched by yt-dlp's VimeoReviewIE, so it falls back to the plain VimeoIE
    which requests the video by ID alone and gets a 404 for private videos.

    We fetch the review page (which is a Next.js app that embeds the full
    video config server-side in a __NEXT_DATA__ JSON blob) and pull out the
    player URL including the private hash token, then hand that to yt-dlp.

    https://vimeo.com/opentrader/review/436577886/7021a182d7?version=1
    """
    if "/review/" not in url:
        return url
    
    buffy = '/'.join(url.split('/')[-2:]).split('?')[0]
    return f"https://vimeo.com/{buffy}"
    

def download_video(url: str, output_dir: Path) -> Path:
    """Download a Vimeo video using yt-dlp in the best available quality."""
    url = resolve_url(url)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--replace-in-metadata", "title", r"[^a-zA-Z0-9]+", "-",
        "--output", output_template,
        "--no-playlist",
        "--print", "after_move:filepath",
        url,
    ]

    log(f"Downloading: {url}")
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
    resolved = resolve_url(url)
    cmd = ["yt-dlp", "--print", "description", resolved]

    log(f"Fetching description: {resolved}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Error fetching description:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    description = result.stdout.strip()
    date_str = datetime.datetime.now().strftime("%Y%m%d")
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
