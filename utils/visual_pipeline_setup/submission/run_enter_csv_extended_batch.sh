#!/bin/sh

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <group_id 1-4>"
    exit 1
fi

group_id="$1"
case "$group_id" in
    1|2|3|4) ;;
    *)
    echo "Error: group_id must be a number between 1 and 4."
    exit 1
    ;;
esac

csv_path="/home/student-team/submission/submission_form_template.csv"
log_dir="/home/student-team/.ros/submission"
mass_est_json_dir="/home/student-team/submission/log_files_combined/subject_${group_id}/volunteer${group_id}_mass"
contact_json_dir="/home/student-team/submission/log_files_combined/subject_${group_id}/volunteer${group_id}_backup"
subject_ods_file="/home/student-team/submission/log_files_combined/subject_${group_id}/Subject_${group_id}.ods"

start_config=$(( (group_id - 1) * 72 + 1 ))
end_config=$(( start_config + 71 ))

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

config_id="$start_config"
while [ "$config_id" -le "$end_config" ]; do
    echo "Running config_id ${config_id} for group ${group_id}"
    python3 "${script_dir}/enter_csv_extended.py" \
        --csv "${csv_path}" \
        --config_id "${config_id}" \
        --log_dir "${log_dir}" \
        --mass_est_json_dir "${mass_est_json_dir}" \
        --contact_json_dir "${contact_json_dir}" \
        --subject_ods_file "${subject_ods_file}"
    config_id=$((config_id + 1))
done

# sh run_enter_csv_extended_batch.sh 1
