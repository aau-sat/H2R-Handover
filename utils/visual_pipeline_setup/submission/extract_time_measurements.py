from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional


def find_json_by_config_id(base_dir: Optional[Path], config_id: str) -> Optional[Path]:
    """Find the first JSON file named with a trailing _<config_id>.json pattern.

    Returns None if the directory is missing or no match is found.
    """
    if base_dir is None:
        return None
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    pattern = re.compile(rf".*_{re.escape(str(config_id))}\.json$", re.IGNORECASE)
    matches = [p for p in base_dir.rglob("*.json") if pattern.match(p.name)]
    if not matches:
        return None
    if len(matches) > 1:
        # prefer the one in top-most directory (shortest path)
        matches.sort(key=lambda p: len(str(p)))
    return matches[0]


def extract_contact_timing_data(contact_file: Optional[Path]) -> Dict[str, Any]:
    """Extract *_ms contact timing values from a robot state JSON payload.

    Returns an empty dict if the file is missing or cannot be parsed.
    """
    if contact_file is None:
        return {}
    try:
        with open(contact_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    fields = {
        "t_human_last_contact_ms",
        "t_robot_first_contact_ms",
        "t_robot_last_contact_ms",
    }
    return {k: payload[k] for k in fields if k in payload}


def extract_mass_estimation_times(mass_est_file: Optional[Path]) -> Dict[str, Any]:
    """Extract timepoint fields from a mass estimation JSON list payload.

    The mass estimation files are expected to be a list whose first element is a dict.
    """
    if mass_est_file is None:
        return {}
    try:
        with open(mass_est_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return {}

    data = payload[0]
    fields = {
        "robot_mass_est_timepoint_t0",
        "robot_mass_est_timepoint_t1",
        "robot_mass_est_timestamp",
    }
    return {k: data[k] for k in fields if k in data}


def extract_submission_times_from_logs(log_dir: Optional[Path]) -> Dict[str, Any]:
    """Scan logs in `log_dir` for JSON payloads containing timing fields.

    Looks for lines containing the marker 'COMPETITION' and attempts to parse JSON
    objects from the remainder of the line. Any keys containing 'time' or ending
    with '_ms' or '_timestamp' will be collected.
    """
    if log_dir is None:
        return {}
    if not log_dir.exists() or not log_dir.is_dir():
        return {}

    aggregated: Dict[str, Any] = {}
    time_key_pattern = re.compile(r"(time|_ms$|_timestamp$)", re.IGNORECASE)

    for log_file in log_dir.rglob("*.log"):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "COMPETITION" not in line:
                        continue
                    # try to extract a JSON substring from the line
                    json_part = _extract_json_substring(line)
                    if not json_part:
                        continue
                    try:
                        obj = json.loads(json_part)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if time_key_pattern.search(k):
                                aggregated[k] = v
                    elif isinstance(obj, list):
                        # if list, inspect first element if dict
                        if obj and isinstance(obj[0], dict):
                            for k, v in obj[0].items():
                                if time_key_pattern.search(k):
                                    aggregated[k] = v
        except OSError:
            continue

    return aggregated


def _extract_json_substring(text: str) -> Optional[str]:
    """Try to locate a JSON object or array substring inside `text`.

    Returns the JSON substring or None.
    """
    # find first '{' or '[' and last matching '}' or ']' roughly
    start_idx = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start_idx = i
            break
    if start_idx is None:
        return None

    # naive approach: try progressively larger substrings until json.loads succeeds
    for end in range(len(text), start_idx, -1):
        candidate = text[start_idx:end].strip()
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def aggregate_time_measurements(
    base_log_dir: Path,
    config_id: Optional[str] = None,
    mass_json_dir: Optional[Path] = None,
    contact_json_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate timing measurements for one or many config IDs.

    If `config_id` is provided, only that config is processed. Otherwise the
    function will try to infer config_ids from directories named like 'log<id>'.
    """
    results: Dict[str, Dict[str, Any]] = {}

    candidates = []
    if config_id:
        candidates = [config_id]
    else:
        if base_log_dir.exists() and base_log_dir.is_dir():
            for p in base_log_dir.iterdir():
                m = re.match(r"log(\d+)$", p.name)
                if m:
                    candidates.append(m.group(1))

    for cid in candidates:
        entry: Dict[str, Any] = {}
        log_dir = base_log_dir / f"log{cid}"
        entry.update(extract_submission_times_from_logs(log_dir))

        mass_file = find_json_by_config_id(mass_json_dir, cid) if mass_json_dir else None
        contact_file = find_json_by_config_id(contact_json_dir, cid) if contact_json_dir else None

        entry.update(extract_mass_estimation_times(mass_file))
        entry.update(extract_contact_timing_data(contact_file))

        results[cid] = entry

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timing measurements for submissions.")
    parser.add_argument("--base_log_dir", required=True, type=Path, help="Base logs directory (contains log<id> subfolders)")
    parser.add_argument("--config_id", required=False, type=str, help="Optional single config_id to extract")
    parser.add_argument("--mass_json_dir", required=False, type=Path, help="Directory containing mass estimation JSONs")
    parser.add_argument("--contact_json_dir", required=False, type=Path, help="Directory containing contact robot state JSONs")
    parser.add_argument("--output_csv", required=False, type=Path, help="Optional output CSV path")

    args = parser.parse_args()

    results = aggregate_time_measurements(
        args.base_log_dir,
        config_id=args.config_id,
        mass_json_dir=args.mass_json_dir,
        contact_json_dir=args.contact_json_dir,
    )

    if args.output_csv:
        # write CSV with header as union of all keys
        all_keys = set()
        for d in results.values():
            all_keys.update(d.keys())
        header = ["config_id"] + sorted(all_keys)
        try:
            with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(header) + "\n")
                for cid, data in results.items():
                    row = [cid] + [str(data.get(k, "")) for k in header[1:]]
                    fh.write(",".join(row) + "\n")
            print(f"Wrote results to {args.output_csv}")
        except OSError as exc:
            print(f"Failed to write CSV: {exc}")
    else:
        for cid, data in results.items():
            print(f"Config {cid}:")
            for k, v in sorted(data.items()):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
