# end_effector_s8 Task Documentation

This note describes the two CSV files exported, and how each value is computed.

## Output files

- `s8_submission_Broken_Arm.csv`: main submission csv
- `s8_accuracy.csv`: log file with reached pose accuracy reporting (not part of submission, but useful for debugging and analysis)

## Main submission CSV

Header:

`run_id,repetition,target_id,x,y,z,qx,qy,qz,qw,start_time,end_time,motion_time_ms`

Meaning:

- `run_id`: user/config run identifier (`run_id` parameter)
- `repetition`: repetition index (1..`repetitions`)
- `target_id`: target index (1..6)
- `x,y,z`: reached pose position in **custom/home frame**, in **mm**
- `qx,qy,qz,qw`: reached pose orientation quaternion in custom/home frame
- `start_time,end_time`: outbound motion start/end in UTC ISO8601 with ms (`YYYY-MM-DDTHH:MM:SS.mmmZ`)
- `motion_time_ms`: outbound motion duration in milliseconds

How it is computed in code:

- Robot executes each outbound move to a target.
- Reached world pose is read from robot forward kinematics.
- Reached pose is converted to custom/home frame.
- Position is converted from meters to millimeters (`* 1000.0`).
- Timestamps come from `start_tp`/`end_tp`, formatted by `toIso8601Utc(...)`.
- `motion_time_ms` uses `duration_cast<milliseconds>(end_tp - start_tp)`.

## 2 Accuracy CSV

Header:

`run_id,repetition,target_id,world_dx_mm,world_dy_mm,world_dz_mm,world_dnorm_mm,world_dang_deg,custom_dx_mm,custom_dy_mm,custom_dz_mm,custom_dnorm_mm,custom_dang_deg,start_time,end_time,motion_time_ms`

Meaning:

- `world_*`: error between reached and target in world/base_link frame
- `custom_*`: error between reached and ideal target in custom/home frame
- `*_dnorm_mm`: Euclidean norm of position error
- `*_dang_deg`: angular error between reached and target quaternions (degrees)
- `start_time,end_time,motion_time_ms`: same outbound timing values as the main submission CSV

How it is computed in code:

- World position error:
  - `world_pos_error = reached_target_world.position - target_pose_world.position`
  - Components and norm are converted to mm (`* 1000.0`).
- Custom position error:
  - `reached_target_home = expressWorldPoseInHomeFrame(reporting_home_pose, reached_target_world.pose)`
  - `ideal_target_home = target_pose_custom`
  - `custom_pos_error = reached_target_home.position - ideal_target_home.position`
  - Components and norm are converted to mm.

## Important reporting note

Both CSV rows are logged per outbound target motion (home -> target). The return motion (target -> home) is executed and time-checked, but its values are **not** written as a row in these two CSV files.
Reachability parameters were reported in team metadata JSON.