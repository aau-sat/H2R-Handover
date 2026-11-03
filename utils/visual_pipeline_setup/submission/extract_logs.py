#!/usr/bin/env python3
"""Searches for specific keywords in log files within a given directory.

This script scans all `.log` files in a specified directory for lines
containing a specific target word (default: 'SUBMISSION') and writes
the matching lines to an output text file. It utilizes pathlib for 
robust, cross-platform path handling and argparse for CLI integration.
"""

import argparse
from pathlib import Path
from typing import List


def find_and_extract_logs(log_dir: Path, output_file: Path, target_word: str = "SUBMISSION") -> None:
    """Searches for a specific word in all log files within a directory.

    Iterates through all files with a .log extension in the specified
    directory, reads them line by line to minimize memory consumption,
    and extracts lines containing the target word.

    Args:
        log_dir: The directory containing the log files to search.
        output_file: The path to the file where matching logs will be saved.
        target_word: The keyword to search for in the log files.
    """
    if not log_dir.is_dir():
        print(f"Error: Directory '{log_dir}' does not exist.")
        return

    matched_lines: List[str] = []

    # Find all .log files in the directory
    for log_path in log_dir.glob("*.log"):
        try:
            # Using errors="replace" prevents crashes if logs contain invalid UTF-8 bytes
            with log_path.open(mode="r", encoding="utf-8", errors="replace") as file:
                for line_number, line in enumerate(file, start=1):
                    if target_word in line:
                        # Prepend filename and line number for better traceability
                        matched_lines.append(f"[{log_path.name}:{line_number}] {line}")
        except PermissionError:
            print(f"Warning: Permission denied for reading {log_path.name}")
        except Exception as e:
            print(f"Warning: Could not read {log_path.name} due to: {e}")

    # Write the collected lines to the output file
    try:
        # Ensure destination directory exists, then create the file if missing.
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not output_file.exists():
            output_file.touch()

        with output_file.open(mode="w", encoding="utf-8") as out_file:
            out_file.writelines(matched_lines)
        print(f"Successfully extracted {len(matched_lines)} lines to '{output_file}'.")
    except Exception as e:
        print(f"Error: Could not write to output file '{output_file}' due to: {e}")


def main() -> None:
    """Parses command-line arguments and executes the log extraction."""
    parser = argparse.ArgumentParser(description="Extract specific log messages from a directory.")
    parser.add_argument("log_dir", type=str, help="Path to the folder containing the log files.")
    parser.add_argument("-o", "--output", type=str, default="/home/student-team/competition/extracted_competitions.txt",
                        help="Path to the output text file (default: /home/student-team/competition/extracted_competitions.txt).")
    parser.add_argument("-w", "--word", type=str, default="COMPETITION",
                        help="The word to search for (default: COMPETITION).")

    args = parser.parse_args()

    log_directory = Path(args.log_dir)
    output_filepath = Path(args.output)

    find_and_extract_logs(log_directory, output_filepath, args.word)


if __name__ == "__main__":
    main()