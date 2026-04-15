#!/usr/bin/env python3
"""
Summary generator that consolidates markdown files from date-named subdirectories.

Iterates through YYYYMMDD-named subdirectories under an input directory,
locates the markdown file within each, and appends them in chronological
order into a single output markdown file.

Usage:
    python summary.py --input /path/to/input --output /path/to/output
    python summary.py -i /path/to/input -o /path/to/output
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


DATE_DIR_PATTERN = re.compile(r"^\d{8}$")


def parse_date_dir(name: str) -> datetime | None:
    """Parse a YYYYMMDD directory name into a datetime, or None if invalid."""
    if not DATE_DIR_PATTERN.match(name):
        return None
    try:
        return datetime.strptime(name, "%Y%m%d")
    except ValueError:
        return None


def find_markdown_file(directory: Path) -> Path | None:
    """Find a single markdown file in the given directory.

    Returns the path to the markdown file, or None if no markdown file is found.
    Prints a warning to stderr if multiple markdown files are found (uses the first
    one sorted alphabetically).
    """
    md_files = sorted(directory.glob("*.md"))
    if not md_files:
        return None
    if len(md_files) > 1:
        print(
            f"Warning: multiple markdown files in {directory.name}, "
            f"using {md_files[0].name}",
            file=sys.stderr,
        )
    return md_files[0]


def build_summary(input_dir: Path, output_path: Path) -> None:
    """Build a summary markdown file from date-named subdirectories."""
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect valid date-named subdirectories
    dated_dirs: list[tuple[datetime, Path]] = []
    for entry in input_dir.iterdir():
        if not entry.is_dir():
            continue
        dt = parse_date_dir(entry.name)
        if dt is not None:
            dated_dirs.append((dt, entry))

    if not dated_dirs:
        print(f"Error: no YYYYMMDD subdirectories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Sort chronologically
    dated_dirs.sort(key=lambda pair: pair[0])

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files_included = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for dt, subdir in dated_dirs:
            md_file = find_markdown_file(subdir)
            if md_file is None:
                print(
                    f"Warning: no markdown file found in {subdir.name}, skipping",
                    file=sys.stderr,
                )
                continue

            content = md_file.read_text(encoding="utf-8")

            # Add a date heading before each file's content
            date_heading = dt.strftime("%B %d, %Y")
            out.write(f"## {date_heading}\n\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("\n")
            files_included += 1

    print(f"Summary written to {output_path} ({files_included} file(s) included)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate markdown files from date-named subdirectories into a single summary.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input directory containing YYYYMMDD subdirectories.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output markdown file (extension .md is implied if omitted).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    # Ensure .md extension
    if output_path.suffix != ".md":
        output_path = output_path.with_suffix(output_path.suffix + ".md")

    build_summary(input_dir, output_path)


if __name__ == "__main__":
    main()
