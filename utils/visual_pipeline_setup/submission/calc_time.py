#!/usr/bin/env python3
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

def find_target_files(folder_path: Path) -> Tuple[Path, Path]:
    """Finds the first JSON and log file in the specified folder.

    Args:
        folder_path (Path): The directory to search for the files.

    Returns:
        Tuple[Path, Path]: A tuple containing the path to the JSON file 
        and the path to the log file.

    Raises:
        FileNotFoundError: If the folder does not exist, or if a .json 
        or .log file is missing.
    """
    if not folder_path.is_dir():
        raise FileNotFoundError(f"The directory {folder_path} was not found.")

    # Best Practice: Use pathlib's glob to efficiently search for file extensions
    json_files = list(folder_path.glob("*.json"))
    log_files = list(folder_path.glob("*.log"))

    if not json_files:
        raise FileNotFoundError("No .json file found in the directory.")
    if not log_files:
        raise FileNotFoundError("No .log file found in the directory.")

    # Warn the user if multiple files exist, to ensure transparency
    if len(json_files) > 1 or len(log_files) > 1:
        print("[WARNING] Multiple .json or .log files found. Using the first ones detected.")

    return json_files[0], log_files[0]

def read_file_content(file_path: Path) -> str:
    """Reads the content of a file.

    Args:
        file_path (Path): The path to the file to be read.

    Returns:
        str: The content of the file.
    """
    with file_path.open('r', encoding='utf-8') as file:
        return file.read()

def calculate_time_elapsed(json_data: str, log_data: str) -> Optional[float]:
    """Calculates the elapsed time between robot contact and system start.

    Args:
        json_data (str): The raw string content containing the JSON-like data.
        log_data (str): The raw string content of the log file.

    Returns:
        Optional[float]: The elapsed time in seconds as a float, or None if 
        timestamps cannot be parsed.
    """
    contact_match = re.search(r't_robot_last_contact_ms[\s":]*(\d+)', json_data)
    start_match = re.search(r'"start_time":\s*([\d\.]+)', log_data)

    if not contact_match or not start_match:
        print("Error: Required timestamps could not be found in the provided data.")
        return None

    contact_time_s = float(contact_match.group(1)) / 1000.0
    start_time_s = float(start_match.group(1))

    return contact_time_s - start_time_s

def main() -> None:
    """Main execution function to parse arguments and orchestrate logic."""
    parser = argparse.ArgumentParser(
        description="Calculate time elapsed using JSON and log files from a specific folder."
    )
    parser.add_argument("folder", type=Path, help="Path to the folder containing the files")
    
    args = parser.parse_args()

    try:
        json_path, log_path = find_target_files(args.folder)
        
        print(f"Processing: {json_path.name} and {log_path.name}")
        
        json_content = read_file_content(json_path)
        log_content = read_file_content(log_path)
        
        elapsed = calculate_time_elapsed(json_content, log_content) * 1000  # Convert back to milliseconds for output
        
        if elapsed is not None:
            print(f"Elapsed Time: {elapsed:.7f} ms")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()