#!/usr/bin/env python3

import argparse
import os
import re


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
    """Ensure .md extension."""
    if not output_path.endswith(".md"):
        output_path += ".md"
    return output_path


def get_cutoff_date_from_file(output_path):
    """Read the first ## heading from the output file and return its date as YYYYMMDD."""
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("##"):
                date_str = line[2:].strip()  # "MM-DD-YYYY"
                parts = date_str.split("-")
                if len(parts) == 3:
                    mm, dd, yyyy = parts
                    return f"{yyyy}{mm}{dd}"
    return None


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

    if os.path.exists(output_path):
        # Incremental update: only process directories newer than the most recent entry
        cutoff = get_cutoff_date_from_file(output_path)
        if cutoff is not None:
            date_dirs = [d for d in date_dirs if d > cutoff]

        if not date_dirs:
            print("No new entries to add.")
            return

        new_content = []
        for dir_name in date_dirs:
            subdir_path = os.path.join(input_dir, dir_name)
            desc_file = find_description_file(subdir_path)
            if desc_file is None:
                continue

            formatted_date = f"{dir_name[4:6]}-{dir_name[6:8]}-{dir_name[:4]}"
            new_content.append(f"## {formatted_date}\n")

            with open(desc_file, "r", encoding="utf-8") as df:
                print(f"Adding content from {desc_file}...")
                content = df.read()
            new_content.append(content)

            if not content.endswith("\n"):
                new_content.append("\n")

        if not new_content:
            print("No new entries to add.")
            return

        with open(output_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(new_content))
            f.write(existing_content)

        print(f"New entries prepended to {output_path}")
    else:
        # Fresh file: write all entries
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
