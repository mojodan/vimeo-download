#!/usr/bin/env python3

import argparse
import os
import re
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate description markdown files from date-named subdirectories into a single summary."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input directory containing YYYYMMDD subdirectories.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output markdown file (implied .md extension).",
    )
    return parser.parse_args()


def resolve_output_path(output_path):
    """Ensure .md extension and make the filename unique if it already exists."""
    if not output_path.endswith(".md"):
        output_path += ".md"

    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"{base}_{timestamp}{ext}"

    return output_path


def get_sorted_date_dirs(input_dir):
    """Return subdirectory names matching YYYYMMDD sorted in reverse chronological order."""
    date_pattern = re.compile(r"^\d{4}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$")
    date_dirs = []
    for entry in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, entry)) and date_pattern.match(entry):
            date_dirs.append(entry)
    date_dirs.sort(reverse=True)
    return date_dirs


def find_description_file(directory):
    """Find the first file starting with 'description-' in the given directory."""
    for filename in os.listdir(directory):
        if filename.startswith("description-") and os.path.isfile(os.path.join(directory, filename)):
            return os.path.join(directory, filename)
    return None


def main():
    args = parse_args()

    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_path = resolve_output_path(args.output)
    date_dirs = get_sorted_date_dirs(input_dir)

    if not date_dirs:
        print("No valid YYYYMMDD subdirectories found.")
        return

    with open(output_path, "w", encoding="utf-8") as out_file:
        for dir_name in date_dirs:
            subdir_path = os.path.join(input_dir, dir_name)
            desc_file = find_description_file(subdir_path)
            if desc_file is None:
                continue

            # Convert YYYYMMDD to MM-DD-YYYY for the heading
            formatted_date = f"{dir_name[4:6]}-{dir_name[6:8]}-{dir_name[:4]}"
            out_file.write(f"## {formatted_date}\n")

            with open(desc_file, "r", encoding="utf-8") as df:
                print(f"Adding content from {desc_file}...")
                content = df.read()
            out_file.write(content)

            # Ensure a trailing newline between sections
            if not content.endswith("\n"):
                out_file.write("\n")

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
