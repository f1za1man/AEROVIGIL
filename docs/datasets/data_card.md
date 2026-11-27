# AEROVIGIL - Dataset Data Card

## 1. Dataset identification

| Item | Value |
|---|---|
| Name | UAV GPS-Spoofing Dataset (PX4 Simulation) |
| Author | Anthony Finn |
| Source | <https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset> (the archive is linked from its README, hosted on Google Drive) |
| Simulation | PX4 SITL and Gazebo Classic, quadcopter, collected in 2023 |
| Archive used | `PX4_Simulation_Data.zip`, 2,577,647,220 bytes, SHA-256 `9E932C713AB2A880C167E6C9073CACB1B6C99561896B7C331E9AE3D98F8B668C` |
| Get it | `python src\aerovigil\data\download.py` (or `make download`): downloads, checks the SHA-256, unpacks to `data/raw/uav_gps_spoofing/` |
| Paper | Detecting Stealthy GPS Spoofing Attack Against UAVs Using Onboard Sensors, <https://par.nsf.gov/servlets/purl/10541394> |

## 2. Licence and permission

- The dataset repository carries **no licence file** (checked through the GitHub API on
  27 November 2026: `"license": null`). Without a licence the default is that all
  rights are reserved, so the data is **not redistributed** through the AEROVIGIL
  repository; the download script fetches it from the author's own link.
- The dataset is publicly available from the author's repository. AEROVIGIL uses it
  for non-commercial academic research only, cites the author and the paper, and does
  not redistribute it.
- Permission: requested from the author via LinkedIn; no reply yet (27 November 2026).
  Any reply will be kept locally as project documentation, not published here.

## 3. What is in it

Simulated flights of one quadcopter in three flight plans (straight, curved, random),
each flown normally and under a GPS-spoofing attack that moves the reported position
step by step away from the true one.

| Version | Content | Used by AEROVIGIL |
|---|---|---|
| Raw | one folder per flight with six streams: `gps`, `raw_gps`, `imu`, `barometer`, `magnetometer`, `groundtruth` (not synchronised) | **yes, since Module 03 v2** |
| Merged | streams joined on shared timestamps (keeps about 70% of rows and cuts flight ends) | inspection only (Modules 02-03 v1) |
| Preprocessed | the authors' own derived features | no |

AEROVIGIL uses **360 flights**: 144 normal and 216 attacked (the Merged set; 36 extra
attacked Raw folders are not used). Flights per mission: curved 108, random 144,
straight 108; rows: curved 57%, random 34%, straight 9% (straight flights are short,
32-55 s; curved flights run 218-299 s).

## 4. How AEROVIGIL builds its base table (Module 03 v2)

Every stream is aligned to the `gps.csv` timeline by nearest timestamp (tolerance
10 ms), so no GPS row is lost; exact duplicate GPS rows are dropped (median 11,088 per
flight). One parquet file per flight in `data/processed/module03_v2/base/`.

| Columns | Unit and convention |
|---|---|
| `latitude_deg`, `longitude_deg`, `altitude` | the GPS fix the autopilot used (spoofed during an attack); degrees, metres |
| `velocity_east`, `velocity_north`, `velocity_up` | GPS velocity, m/s, East-North-Up |
| `raw_latitude_deg` ... `raw_velocity_up` | `raw_gps`: the same receiver before the spoofing was applied |
| `gt_*` | Gazebo ground truth (position, velocity, attitude) |
| `pressure_altitude` (+ temperature, absolute pressure) | barometer, metres |
| `linear_acceleration_x/y/z` | accelerometer, m/s^2, body frame, specific force (includes gravity) |
| `angular_velocity_x/y/z` | gyroscope, rad/s (the dataset README says deg/s) |
| `orientation_x/y/z/w` | attitude quaternion in that order, body-to-world, East-North-Up (verified: horizontal acceleration R^2 0.963) |
| `magnetic_field_x/y/z` | gauss (the README says tesla), **in PX4 body axes (forward, right, down)**: flip y and z to use it with the quaternion (then the field keeps one direction to 1.7 deg and dips 63 deg, as at the simulator's location) |
| `t_s` | seconds from the flight's first row, from the log timestamps |

## 5. Sampling

- Nominal 250 Hz; measured: some flights at about 250 Hz, others at about 125 Hz, on
  average about **172 rows/s**, with irregular spacing. Never compute time from a row
  count: use `t_s`.
- The IMU, barometer, magnetometer and ground-truth streams **drop out** about twice per
  flight (up to 32 s) while GPS continues: about 8% of rows. These rows are kept with
  NaN sensors; `aligned_all == 1` marks complete rows.
- **IMU start offset:** real logs open with about 18 dropout rows, so a calculation that
  drops them and restarts its clock runs 0.10-0.12 s (typically 0.116 s) behind `t_s`.
  Every detector here works on the `t_s` clock.

## 6. Labels

| Column | Meaning |
|---|---|
| `attack_enabled` / `attack_label` | the dataset's own label: 1 while the attack is switched on (kept unchanged) |
| `offset_m` | measured horizontal distance between `gps` and `raw_gps` |
| `spoof_truth` | 1 from the first labelled row to the last row with `offset_m` >= 1 m |
| `phase` | 0 normal flight, 1 pre-attack, 2 ramp, 3 plateau, 4 post-attack (measured from `offset_m`) |
| `seconds_since_onset` | seconds from the first labelled row |

Real attacks start 1-26 s into the flight (median 5.4 s), ramp for about 4 s and then
hold a plateau of **7.5-11.1 m** (median 9.8 m) until the end of the label.

## 7. Split

Split by flight (seed 42): **252 train, 54 validation, 54 test** flights
(`data/interim/module03/flight_splits.csv`). Synthetic attacks (Module 04) are copies of
normal flights with an offset added, and always stay in their source flight's split.
For the LSTMs, every second normal training flight is held back as calibration
(`role = calib`). `tests/evaluation/test_no_leakage.py` checks all of this on the real
files.

## 8. Known quirks and limits

- Simulation only: no real radio signal, no real drone. One vehicle model.
- The real attacks are mild (plateau about 10 m); harder attacks come from Module 04.
- `raw_gps` (used as ground truth) sits about 2 m from `gt_*` (median; 3.6 m at the 99th
  percentile): ordinary GPS error.
- Multirotor attitude does not follow the direction of travel (the drone crabs), so a
  course-against-heading check is noisy at low speed.
- 140,137 Merged rows are out of time order (sorted, none removed); 92 dropouts of about
  16 s in 83 Merged flights.
- Sensor frames and units are not documented by the dataset; the conventions in
  section 4 were measured on the data.

## 9. Integrity

- The downloaded files are never modified; all processing is scripted
  (`make data`) and writes to `data/interim` and `data/processed`.
- Flight identifiers: `<Mission>__<Attacked|Normal>__<log name>`.
- Raw data, base tables and synthetic flights are excluded from Git; only small result
  tables and the split file are committed.

## 10. AEROVIGIL usage

GPS-spoofing detection, sensor-consistency analysis, machine learning experiments,
adaptive-attack evaluation, dead-reckoning analysis and the offline demo. The dataset
and paper are cited in the final project report.
