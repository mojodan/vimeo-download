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
import os
import sys
import subprocess
import tempfile
from pathlib import Path


def download_video(url: str, output_dir: Path) -> Path:
    """Download a Vimeo video using yt-dlp in the best available quality."""
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--print", "after_move:filepath",
        url,
    ]

    print(f"Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error downloading video:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # yt-dlp prints the final filepath via --print after_move:filepath
    filepath = result.stdout.strip().splitlines()[-1]
    video_path = Path(filepath)

    if not video_path.exists():
        # Fall back: find the most recently modified mp4 in output_dir
        mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4_files:
            print("Could not locate downloaded video file.", file=sys.stderr)
            sys.exit(1)
        video_path = mp4_files[0]

    print(f"Downloaded: {video_path}")
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

    print("Extracting audio…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error extracting audio:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def transcribe(audio_path: Path, model_name: str, output_dir: Path) -> Path:
    """Transcribe audio to English using OpenAI Whisper."""
    import whisper

    print(f"Loading Whisper model '{model_name}'…")
    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while)…")
    result = model.transcribe(str(audio_path), language="en", task="transcribe")

    transcript_path = output_dir / (audio_path.stem + ".txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
        f.write("\n")

    # Also write an SRT subtitle file
    srt_path = output_dir / (audio_path.stem + ".srt")
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    video_path = download_video(args.url, output_dir)

    # Extract audio to a temp file (or output dir if --keep-audio)
    if args.keep_audio:
        audio_path = output_dir / (video_path.stem + ".wav")
        extract_audio(video_path, audio_path)
        transcript_path = transcribe(audio_path, args.model, output_dir)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)
        try:
            extract_audio(video_path, audio_path)
            transcript_path = transcribe(audio_path, args.model, output_dir)
        finally:
            audio_path.unlink(missing_ok=True)

    print(f"\nDone!")
    print(f"  Video:      {video_path}")
    print(f"  Transcript: {transcript_path}")
    srt_path = transcript_path.with_suffix(".srt")
    if srt_path.exists():
        print(f"  Subtitles:  {srt_path}")


if __name__ == "__main__":
    main()
