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
import ssl
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def _make_ssl_context():
    """Create an SSL context that skips certificate verification."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch_url(url: str, headers: dict | None = None) -> str:
    """Fetch a URL and return the response body as a string."""
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, context=_make_ssl_context()) as resp:
        return resp.read().decode("utf-8")


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

    Both formats require fetching the review page to obtain the
    embedPlayerConfigUrl (which contains bypass_privacy tokens), then
    extracting the authenticated player URL from the config.
    """
    if url in _resolved_url_cache:
        return _resolved_url_cache[url]

    result: tuple[str, str | None]
    if "/reviews/" in url:
        # New-style review URL — must fetch the page to get the hash token
        result = _resolve_reviews_url(url), url
    elif "/review/" in url:
        # Old-style review URL — fetch the review data page to get config
        result = _resolve_old_review_url(url), url
    else:
        result = url, None

    _resolved_url_cache[url] = result
    return result


def _resolve_old_review_url(url: str) -> str:
    """Fetch an old-style Vimeo review data page and extract the player config URL.

    The /review/data/ endpoint returns an HTML page with __NEXT_DATA__ that
    contains an embedPlayerConfigUrl with bypass_privacy tokens.  We fetch
    the config JSON from that URL to extract the HLS master playlist, which
    yt-dlp can download directly without embed-domain restrictions.
    """
    parts = url.split('?')[0].rstrip('/').split('/')
    try:
        idx = parts.index('review')
        video_id = parts[idx + 1]
        video_hash = parts[idx + 2]
    except (ValueError, IndexError):
        return url

    # Find the username (segment before "review")
    username = parts[idx - 1] if idx > 0 else None
    if not username:
        return url

    data_url = f"https://vimeo.com/{username}/review/data/{video_id}/{video_hash}"
    log(f"Fetching review data from: {data_url}")

    try:
        html = _fetch_url(data_url)
    except Exception as exc:
        log(f"Warning: could not fetch review data page ({exc}), trying URL as-is", file=sys.stderr)
        return url

    # Extract embedPlayerConfigUrl and description from __NEXT_DATA__
    config_url, og_description = _extract_embed_config_and_description(html)
    if not config_url:
        log("Warning: could not find embedPlayerConfigUrl, trying URL as-is", file=sys.stderr)
        return url

    # Pre-cache ogDescription from the review page (player config often lacks it)
    if og_description:
        _video_metadata["description"] = og_description

    # Fetch the player config to get the HLS/DASH stream URL
    return _resolve_stream_url_from_config(config_url, video_id, url)


def _extract_embed_config_and_description(html: str) -> tuple[str | None, str | None]:
    """Extract the embedPlayerConfigUrl and ogDescription from __NEXT_DATA__."""
    scripts = re.findall(
        r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
        html, re.DOTALL,
    )
    for script_body in scripts:
        try:
            data = json.loads(script_body)
            page_props = data.get("props", {}).get("pageProps", {})
            config_url = page_props.get("embedPlayerConfigUrl")
            if config_url:
                og_desc = page_props.get("ogDescription", "")
                return config_url, og_desc
        except (json.JSONDecodeError, AttributeError):
            continue
    return None, None


def _resolve_stream_url_from_config(config_url: str, video_id: str, original_url: str) -> str:
    """Fetch the player config JSON and return a direct stream URL for yt-dlp.

    Also caches video metadata (title, description) in _video_metadata.
    """
    log(f"Fetching player config for video {video_id}…")
    try:
        config_body = _fetch_url(config_url, {"Referer": "https://vimeo.com/"})
        config = json.loads(config_body)
    except Exception as exc:
        log(f"Warning: could not fetch player config ({exc}), trying URL as-is", file=sys.stderr)
        return original_url

    # Cache video metadata for later use by download_video / download_description
    video_info = config.get("video", {})
    if video_info.get("title"):
        _video_metadata["title"] = video_info["title"]
    # Only overwrite description if the config has one (ogDescription may already be cached)
    if video_info.get("description"):
        _video_metadata["description"] = video_info["description"]

    files = config.get("request", {}).get("files", {})

    # Prefer HLS (yt-dlp handles it natively with format selection)
    for stream_type in ("hls", "dash"):
        stream = files.get(stream_type, {})
        cdn = stream.get("default_cdn")
        if cdn:
            cdn_info = stream.get("cdns", {}).get(cdn, {})
            stream_url = cdn_info.get("avc_url") or cdn_info.get("url")
            if stream_url:
                log(f"Resolved to {stream_type.upper()} stream for video {video_id}")
                return stream_url

    log("Warning: no stream URL found in config, trying original URL as-is", file=sys.stderr)
    return original_url


# Cache for video metadata extracted during URL resolution
_video_metadata: dict[str, str] = {}
# Cache for resolved URL results to avoid repeated network fetches
_resolved_url_cache: dict[str, tuple[str, str | None]] = {}


def _resolve_reviews_url(url: str) -> str:
    """Fetch a new-style Vimeo review page and extract the player URL."""
    log("Resolving Vimeo review URL…")
    try:
        html = _fetch_url(url)
    except Exception as exc:
        log(f"Warning: could not fetch review page ({exc}), trying URL as-is", file=sys.stderr)
        return url

    # Try to get embedPlayerConfigUrl from __NEXT_DATA__ (preferred path)
    config_url, og_desc = _extract_embed_config_and_description(html)
    if config_url:
        if og_desc:
            _video_metadata["description"] = og_desc
        # Extract video_id from the config URL
        vid_match = re.search(r'/video/(\d+)/', config_url)
        video_id = vid_match.group(1) if vid_match else "unknown"
        return _resolve_stream_url_from_config(config_url, video_id, url)

    # Fallback: scan for player.vimeo.com URL with hash token
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

    # If we have a cached title from the player config, sanitise it for the filename
    cached_title = _video_metadata.get("title", "")
    if cached_title:
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "-", cached_title).strip("-")
        output_template = str(output_dir / f"{safe_title}.%(ext)s")
    else:
        output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--print", "after_move:filepath",
    ]
    if not cached_title:
        cmd += ["--replace-in-metadata", "title", r"[^a-zA-Z0-9]+", "-"]
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
    """Download the video description and save it as a markdown file.

    Uses cached metadata from resolve_url() when available, falling back
    to yt-dlp --print description for non-review URLs.
    """
    description = _video_metadata.get("description", "")

    if not description:
        # Fallback for non-review URLs: use yt-dlp
        resolved, referer = resolve_url(url)
        cmd = ["yt-dlp", "--no-check-certificates", "--print", "description"]
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
    parser.add_argument(
        "--desc-only", "--do",
        action="store_true",
        help="Download only the video description (as markdown) and stop. No video download or transcription.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the URL early so metadata is cached for description & download
    resolve_url(args.url)

    # Download description if requested
    desc_path = None
    if args.description or args.desc_only:
        desc_path = download_description(args.url, output_dir)

    # If --desc-only, stop after downloading the description
    if args.desc_only:
        log("Done!")
        if desc_path:
            log(f"  Description: {desc_path}")
        return

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
