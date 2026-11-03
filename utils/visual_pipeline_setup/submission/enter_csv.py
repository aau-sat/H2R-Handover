import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Any

# Constants for initial robot pose
ROBOT_INITIAL_POSE = {
    "robot_initial_pose_x": 0.0,
    "robot_initial_pose_y": 0.0,
    "robot_initial_pose_z": 0.0,
    "robot_initial_pose_q1": 1.0,
    "robot_initial_pose_q2": 0.0,
    "robot_initial_pose_q3": 0.0,
    "robot_initial_pose_q4": 0.0,
}

MASS_SEQUENCE_G = [3.0, 120.0, 11.0, 348.0, 8.0, 395.0, 137.0, 389.0]


def derive_mass_from_config_id(config_id: str) -> float:
    """Map config IDs to the repeating 9-ID mass schedule."""
    config_index = int(config_id)
    if config_index < 1:
        raise ValueError(f"config_id must be >= 1, got {config_id}")
    mass_index = ((config_index - 1) // 9) % len(MASS_SEQUENCE_G)
    return MASS_SEQUENCE_G[mass_index]

def extract_submission_data(log_dir: Path) -> Dict[str, Any]:
    """
    Scans a log directory for files, searching for 'SUBMISSION' lines to extract JSON data.

    Args:
        log_dir (Path): The directory path containing the log files for a specific config_id.

    Returns:
        Dict[str, Any]: A dictionary containing the aggregated extracted JSON key-value pairs.
    """
    extracted_data = {}
    
    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Warning: Log directory {log_dir} does not exist.")
        return extracted_data

    # Scan all files in the given directory
    for log_file in log_dir.rglob("*.log"): # Assumes .log extension; adjust if necessary
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "SUBMISSION" in line:
                    # Attempt to find JSON structure in the line
                    match = re.search(r'(\{.*\})', line)
                    if match:
                        try:
                            json_payload = json.loads(match.group(1))
                            extracted_data.update(json_payload)
                            print(f"Extracted data from {log_file.name}: {json_payload}")
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON in line: {line.strip()}")

    return extracted_data

def update_csv_submission(csv_path: Path, config_id: str, initial_mass: float | None, base_log_dir: Path) -> None:
    """
    Updates a specific row in the submission CSV file based on config_id.

    Args:
        csv_path (Path): Path to the target CSV file.
        config_id (str): The configuration ID matching the target row.
        initial_mass (float | None): Optional override for the initial mass in grams.
        base_log_dir (Path): The root directory containing the config logs.
    """
    derived_mass = derive_mass_from_config_id(config_id)
    if initial_mass is not None and float(initial_mass) != derived_mass:
        print(
            f"Info: overriding provided mass {initial_mass}g with derived mass {derived_mass}g for config_id {config_id}."
        )

    log_dir = base_log_dir / f"log{config_id}"
    extracted_log_data = extract_submission_data(log_dir)
    print(f"Extracted log data for config_id {config_id}: {extracted_log_data}")

    fixed_value_columns = set(ROBOT_INITIAL_POSE.keys()) | {
        "initial_mass_measured_g",
        "robot_mass_est_available",
        "spill_observed_during_human_maneuvering",
        "final_mass_null_flag",
        "final_mass_measured_g",
    }

    def find_problematic_fields(row_data, headers):
        problematic = []
        for key in headers:
            value = row_data.get(key)
            normalized = str(value).strip().lower()
            if normalized == "-1" or normalized == "unknown":
                problematic.append(key)
        return problematic

    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', dir=csv_path.parent)
    row_updated = False
    updated_value_count = 0
    problematic_field_names = []

    with open(csv_path, "r", encoding="utf-8") as infile, temp_file:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if not fieldnames:
            print("Error: CSV file is empty or missing headers.")
            return

        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if row.get("config_id") == config_id:
                row_updated = True
                
                # 1. Apply hardcoded constant parameters
                for col, value in ROBOT_INITIAL_POSE.items():
                    if col in fieldnames:
                        row[col] = value
                        updated_value_count += 1

                # 2. Apply fixed rules
                row["initial_mass_measured_g"] = derived_mass
                updated_value_count += 1
                row["robot_mass_est_available"] = 1
                updated_value_count += 1
                row["spill_observed_during_human_maneuvering"] = 0
                updated_value_count += 1
                row["final_mass_null_flag"] = 0
                updated_value_count += 1
                row["final_mass_measured_g"] = derived_mass
                updated_value_count += 1
                
                # find start time
                start_time = 0
                for key, value in extracted_log_data.items():
                    if key == 'trial_begin_time':
                        start_time = value
                        break
                
                # 3. Apply extracted JSON data from logs
                for col in fieldnames:
                    # handle timestamp data if present
                    found = False
                    val_timestamp_0 = None
                    val_timestamp_1 = None
                    for key, value in extracted_log_data.items():
                        if key.lower() == col.lower():
                            # if it's a timestamp, we'll deduct start time from it
                            if (col.lower()).endswith("_ms"):
                                row[col] = int(float(value) - start_time)
                            else:
                                row[col] = value
                            updated_value_count += 1
                            found = True
                            break
                        elif f'{col.lower()}_t0' == key.lower():
                            val_timestamp_0 = value
                        elif f'{col.lower()}_t1' == key.lower():
                            val_timestamp_1 = value
                    if not found and (val_timestamp_0 is not None or val_timestamp_1 is not None):
                        # check for both
                        if val_timestamp_0 is not None and val_timestamp_1 is not None:
                            row[col] = f'range_ms:{int(float(val_timestamp_0) - start_time)}-{int(float(val_timestamp_1) - start_time)}'
                        elif val_timestamp_0 is not None:
                            row[col] = f'single_ms:{int(float(val_timestamp_0) - start_time)}'
                        elif val_timestamp_1 is not None:
                            row[col] = f'single_ms:{int(float(val_timestamp_1) - start_time)}'
                        updated_value_count += 1
                        found = True
                    if not found and col not in fixed_value_columns and col != 'config_id':
                        print(f"Warning: CSV header '{col}' not found in log files.")

                problematic_field_names = find_problematic_fields(row, fieldnames)

            writer.writerow(row)

    # Atomically replace the old file with the new one (Best practice on Linux)
    os.replace(temp_file.name, csv_path)
    
    if row_updated:
        print(
            f"Successfully updated row for config_id: {config_id} | "
            f"rows_updated=1 | values_updated={updated_value_count}"
        )
        print(
            f"Problematic values after update: count={len(problematic_field_names)} | "
            f"fields={problematic_field_names}"
        )
    else:
        print(
            f"Config ID '{config_id}' not found in the CSV. No rows updated. | "
            f"rows_updated=0 | values_updated=0"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update submission CSV based on config ID and logs.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the submission CSV file.")
    parser.add_argument("--config_id", required=True, type=str, help="Configuration ID to update.")
    parser.add_argument("--mass", required=False, type=float, help="Optional legacy mass argument; derived automatically from config_id.")
    parser.add_argument("--log_dir", required=True, type=Path, help="Base directory containing log folders.")
    
    args = parser.parse_args()

    update_csv_submission(
        csv_path=args.csv,
        config_id=args.config_id,
        initial_mass=args.mass,
        base_log_dir=args.log_dir
    )
    
    
    # python3 enter_csv_extended.py --csv /home/student-team/submission/submission_form_template.csv --config_id 1 --log_dir /home/student-team/.ros/submission --mass_est_json_dir /home/student-team/submission/log_files_combined/subject_1/volunteer1_mass --contact_json_dir /home/student-team/submission/log_files_combined/subject_1/volunteer1_backup
