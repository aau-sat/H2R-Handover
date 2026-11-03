import argparse
import csv
import json
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
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
    Scans a log directory for files, searching for 'COMPETITION' lines to extract JSON data.

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
                if "COMPETITION" in line:
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


def find_json_by_config_id(base_dir: Path, config_id: str) -> Path | None:
    """Find the first JSON file named with a trailing _<config_id>.json pattern."""
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Warning: JSON directory {base_dir} does not exist.")
        return None

    pattern = re.compile(rf".*_{re.escape(str(config_id))}\.json$", re.IGNORECASE)
    matches = [path for path in base_dir.rglob("*.json") if pattern.match(path.name)]

    if not matches:
        print(f"Warning: No JSON file ending with _{config_id}.json found in {base_dir}.")
        return None

    if len(matches) > 1:
        print(
            f"Warning: Multiple JSON files ending with _{config_id}.json found in {base_dir}. "
            f"Using the first one: {matches[0]}"
        )

    return matches[0]


def extract_mass_estimation_data(mass_est_file: Path | None) -> Dict[str, Any]:
    """Extract fields from mass estimation JSON list payload."""
    if mass_est_file is None:
        return {}

    try:
        with open(mass_est_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to read mass estimation file {mass_est_file}: {exc}")
        return {}

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        print(f"Warning: Unexpected mass estimation payload format in {mass_est_file}.")
        return {}

    data = payload[0]
    fields = {
        "robot_mass_est_available",
        "robot_mass_est_g",
        "robot_mass_est_timepoint_t0",
        "robot_mass_est_timepoint_t1",
    }
    extracted = {k: data[k] for k in fields if k in data}
    print(f"Extracted mass estimation data from {mass_est_file.name}: {extracted}")
    return extracted


def extract_contact_timing_data(contact_file: Path | None) -> Dict[str, Any]:
    """Extract only *_ms contact timing values from robot state JSON payload."""
    if contact_file is None:
        return {}

    try:
        with open(contact_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to read contact timing file {contact_file}: {exc}")
        return {}

    if not isinstance(payload, dict):
        print(f"Warning: Unexpected contact timing payload format in {contact_file}.")
        return {}

    fields = {
        "t_human_last_contact_ms",
        "t_robot_first_contact_ms",
        "t_robot_last_contact_ms",
    }
    extracted = {k: payload[k] for k in fields if k in payload}
    print(f"Extracted contact timing data from {contact_file.name}: {extracted}")
    return extracted


def _parse_ods_row_values(row_element: ET.Element, table_ns: str, text_ns: str) -> list[str]:
    """Parse a single ODS row element into a flat list of cell strings.

    Important: this intentionally does not expand repeated rows to avoid large memory/time overhead.
    """
    values: list[str] = []
    for cell in row_element.findall(f"{{{table_ns}}}table-cell"):
        repeat_cols = int(cell.attrib.get(f"{{{table_ns}}}number-columns-repeated", "1"))
        text_chunks = [
            "".join(p.itertext()).strip()
            for p in cell.findall(f"{{{text_ns}}}p")
        ]
        cell_text = " ".join(chunk for chunk in text_chunks if chunk)
        values.extend([cell_text] * repeat_cols)
    return values


def parse_ods_first_sheet(ods_file: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Parse the first ODS sheet and return (header, raw rows)."""
    namespaces = {
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    }

    try:
        with zipfile.ZipFile(ods_file, "r") as zf:
            with zf.open("content.xml") as content:
                root = ET.parse(content).getroot()
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(f"Warning: Failed to parse ODS file {ods_file}: {exc}")
        return [], []

    table = root.find(".//table:table", namespaces)
    if table is None:
        print(f"Warning: No sheet found in ODS file {ods_file}.")
        return [], []

    header: list[str] | None = None
    parsed_rows: list[dict[str, str]] = []

    for row in table.findall("table:table-row", namespaces):
        values = _parse_ods_row_values(row, namespaces["table"], namespaces["text"])
        if not any(str(v).strip() for v in values):
            continue

        if header is None:
            header = [str(v).strip() for v in values]
            continue

        if len(values) < len(header):
            values = values + [""] * (len(header) - len(values))
        row_dict = {header[idx]: str(values[idx]).strip() for idx in range(len(header)) if header[idx]}
        parsed_rows.append(row_dict)

    return header or [], parsed_rows


def extract_delivery_location_from_ods(ods_file: Path | None, config_id: str) -> Dict[str, Any]:
    """Extract delivery location columns by rowId/config_id and convert cm to mm for x/y."""
    if ods_file is None:
        return {}
    if not ods_file.exists() or not ods_file.is_file():
        print(f"Warning: ODS file {ods_file} does not exist.")
        return {}

    header, rows = parse_ods_first_sheet(ods_file)
    if not rows or not header:
        return {}

    def normalize_id(value: str) -> str:
        text = str(value).strip()
        if not text:
            return text
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            return text
        return text

    config_id_norm = normalize_id(config_id)
    id_candidates = {"rowid", "config_id", "id"}
    x_candidates = {"delivery_location_est_x_mm", "delivery_location_est_x_cm", "delivery_location_est_x"}
    y_candidates = {"delivery_location_est_y_mm", "delivery_location_est_y_cm", "delivery_location_est_y"}
    notes_candidates = {"notes", "note"}
    success_values = {"successfull", "successful", "success", "Success"}

    def find_column(row_dict: dict[str, str], candidates: set[str]) -> str | None:
        for key in row_dict.keys():
            if key.strip().lower() in candidates:
                return key
        return None

    matching_rows: list[dict[str, str]] = []
    for row in rows:
        id_key = find_column(row, id_candidates)
        if id_key is None:
            continue
        if normalize_id(row.get(id_key, "")) == config_id_norm:
            matching_rows.append(row)

    if not matching_rows:
        print(f"Warning: No ODS row found for config_id {config_id} in {ods_file}; setting delivery location to -1.")
        return {
            "delivery_location_est_x_mm": -1,
            "delivery_location_est_y_mm": -1,
            "delivery_location_est_z_mm": -1,
        }

    notes_key = find_column(matching_rows[0], notes_candidates)
    successful_rows: list[dict[str, str]] = []
    if notes_key is not None:
        successful_rows = [
            row for row in matching_rows
            if row.get(notes_key, "").strip().lower() in success_values
        ]

    if not successful_rows:
        print(
            f"Warning: No row with Notes=Successfull for config_id {config_id} in {ods_file}; "
            "setting delivery location to -1."
        )
        return {
            "delivery_location_est_x_mm": -1,
            "delivery_location_est_y_mm": -1,
            "delivery_location_est_z_mm": -1,
        }

    chosen_row = successful_rows[0]
    x_key = find_column(chosen_row, x_candidates)
    y_key = find_column(chosen_row, y_candidates)
    if x_key is None or y_key is None:
        print(
            f"Warning: Missing delivery_location_est_x/y columns in successful ODS row for config_id {config_id}; "
            "setting delivery location to -1."
        )
        return {
            "delivery_location_est_x_mm": -1,
            "delivery_location_est_y_mm": -1,
            "delivery_location_est_z_mm": -1,
        }

    try:
        x_mm = float(chosen_row.get(x_key, "0")) * 10.0
        y_mm = float(chosen_row.get(y_key, "0")) * 10.0
    except ValueError:
        print(
            f"Warning: Non-numeric delivery location values in successful ODS row for config_id {config_id}; "
            "setting delivery location to -1."
        )
        return {
            "delivery_location_est_x_mm": -1,
            "delivery_location_est_y_mm": -1,
            "delivery_location_est_z_mm": -1,
        }

    extracted = {
        "delivery_location_est_x_mm": x_mm,
        "delivery_location_est_y_mm": y_mm,
        "delivery_location_est_z_mm": 0,
    }
    print(f"Extracted delivery location from {ods_file.name} for config_id {config_id}: {extracted}")
    return extracted

def update_csv_submission(
    csv_path: Path,
    config_id: str,
    initial_mass: float | None,
    base_log_dir: Path,
    mass_est_json_dir: Path | None,
    contact_json_dir: Path | None,
    subject_ods_file: Path | None,
) -> None:
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

    mass_est_file = find_json_by_config_id(mass_est_json_dir, config_id) if mass_est_json_dir else None
    contact_file = find_json_by_config_id(contact_json_dir, config_id) if contact_json_dir else None
    mass_est_data = extract_mass_estimation_data(mass_est_file)
    contact_timing_data = extract_contact_timing_data(contact_file)
    delivery_location_data = extract_delivery_location_from_ods(subject_ods_file, config_id)
    extracted_log_data.update(mass_est_data)
    extracted_log_data.update(contact_timing_data)
    extracted_log_data.update(delivery_location_data)

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
            if str(row.get("config_id", "")).strip() == str(config_id):
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
    parser = argparse.ArgumentParser(description="Update submission CSV based on config ID, logs, and optional JSON sources.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the submission CSV file.")
    parser.add_argument("--config_id", required=True, type=str, help="Configuration ID to update.")
    parser.add_argument("--mass", required=False, type=float, help="Optional legacy mass argument; derived automatically from config_id.")
    parser.add_argument("--log_dir", required=True, type=Path, help="Base directory containing log folders.")
    parser.add_argument(
        "--mass_est_json_dir",
        required=False,
        type=Path,
        help="Optional directory tree containing mass estimation JSON files named *_<config_id>.json.",
    )
    parser.add_argument(
        "--contact_json_dir",
        required=False,
        type=Path,
        help="Optional directory tree containing contact timing JSON files named *_<config_id>.json.",
    )
    parser.add_argument(
        "--subject_ods_file",
        required=False,
        type=Path,
        help="Optional ODS file containing rowId and delivery location columns.",
    )
    
    args = parser.parse_args()

    update_csv_submission(
        csv_path=args.csv,
        config_id=args.config_id,
        initial_mass=args.mass,
        base_log_dir=args.log_dir,
        mass_est_json_dir=args.mass_est_json_dir,
        contact_json_dir=args.contact_json_dir,
        subject_ods_file=args.subject_ods_file,
    )
    
    #python3 enter_csv.py --csv /home/student-team/submission/submission_form_template.csv --config_id "1" --mass 150.5 --log_dir /home/student-team/.ros