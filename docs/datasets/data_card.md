# AEROVIGIL - Dataset Data Card

## 1. Dataset Identification

**Dataset Name:** UAV GPS-Spoofing Dataset (PX4 Simulation)
**Project:** AEROVIGIL
**Dataset Type:** Simulated UAV flight telemetry
**Author:** Anthony Finn
**Primary Source:**
https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset
**Original Simulation Environment:** PX4 SITL and Gazebo Classic (collected in 2023, quadcopter)
**Archive used:** PX4_Simulation_Data.zip (SHA-256: 9E932C713AB2A880C167E6C9073CACB1B6C99561896B7C331E9AE3D98F8B668C)

## 2. Dataset Description

The dataset contains simulated UAV flight measurements collected using a quadcopter in a PX4 SITL and Gazebo Classic environment.

It includes normal and GPS-spoofed flight scenarios. The attack modifies the reported GPS position incrementally, so the reported position drifts away from the actual simulated position.

## 3. Available Sensor Categories

- Global Positioning System (GPS)
- Inertial Measurement Unit (IMU)
- Barometer
- Magnetometer
- Ground-truth flight measurements (Raw version only)
- Raw GPS measurements (Raw version only)

## 4. Flight Scenarios

The dataset contains three flight-plan categories:

- Straight
- Curved
- Random

Each category includes normal and attacked flight scenarios.

## 5. Sampling Frequency

Measured on the merged files (Modules 02-03). The nominal 250 Hz is not uniform:

- 298 flights run at about 250 Hz and 62 flights at about 125 Hz.
- Spacing is irregular inside flights.
- 83 flights contain 92 dropouts of about 16 s.
- 140,137 rows are stored out of time order. Module 03 sorts them; no rows are removed.

## 6. Data Labelling

`attack_enabled` (copied to `attack_label`) is 0 for normal operation and 1 while the attack is active.

Verified distribution over all rows:

- 0.0: 3,590,904 rows
- 1.0: 4,302,145 rows

144 flights contain no attack. 216 flights contain both 0 and 1, because the attack starts after take-off. Row labels are never inferred from the folder name.

## 7. Dataset Versions

The dataset provides:

- Raw data: individual sensor streams, including ground truth and raw GPS measurements. These streams are not guaranteed to be synchronized.
- Merged data: time-synchronized sensor measurements.
- Preprocessed data: data containing derived features used in the original research.

AEROVIGIL retains the raw data and uses the merged data for inspection and modelling.

## 8. Permission and Usage

The dataset author gave permission for non-commercial academic research use.

- Permission given: requested via LinkedIn, awaiting reply
- The author's reply is retained locally as project documentation and is not published in the repository.
- The original dataset is not redistributed through the AEROVIGIL GitHub repository.

## 9. Data Integrity

- Original downloaded files remain unchanged.
- Raw data is excluded from GitHub.
- Data processing is performed using reproducible Python scripts.
- Dataset inspection results are generated programmatically.
- Flight identifiers distinguish individual missions. Format: `<mission>__<Attacked|Normal>__<log name>`.
- The first build of the combined table had a defective `flight_id` column. It was repaired and the repair verified with `verify_flight_ids.py` (360 flights, 7,893,049 rows, per-flight row counts match the manifest).

## 10. Limitations

- The dataset is based on simulation rather than real-world drone recordings.
- Raw sensor streams are not guaranteed to be time-synchronized.
- The available flight scenarios may not represent every real UAV mission.
- The dataset does not provide real-world radio-frequency GPS signal measurements.
- The merged files hold `pressure_altitude` only (no absolute pressure or temperature).
- The IMU frame, magnetometer axes and units, and attitude convention are not documented in the merged files. Checked on one flight: the magnetic field magnitude is 0.48, so the unit is gauss (the README says Tesla); angular velocity is consistent with rad/s (the README says degrees/s) but not proven; quaternions are unit-norm.
- Sampling is non-uniform (see section 5) and straight flights are short, so they make up only about 10% of the rows.

## 11. AEROVIGIL Usage

The dataset is used for:

- UAV telemetry analysis.
- GPS spoofing detection.
- Sensor consistency analysis.
- Machine learning experiments.
- Adaptive attack evaluation.
- Detection performance measurement.
- Ground-truth-based evaluation of GPS position deviation.

## 12. References

Dataset repository:
https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset

Research paper:
Detecting Stealthy GPS Spoofing Attack Against UAVs Using Onboard Sensors.
https://par.nsf.gov/servlets/purl/10541394

The original dataset and research paper are cited in the final project report.

## 13. Verified Facts

| Item | Result |
|---|---|
| Merged files | 360 flights, 7,893,049 rows, 27 columns in the combined table (23 original + attack_label, flight_id, mission_type, timestamp) |
| Rows per mission | curved 4,362,481; random 2,741,656; straight 788,912 |
| Flights | 144 without attack, 216 with attack starting mid-flight |
| Missing values / duplicate rows | 0 / 0 |
| Sampling | 298 flights at about 250 Hz, 62 at about 125 Hz; 92 dropouts of about 16 s in 83 flights |
| Out-of-order rows | 140,137 (sorted in Module 03) |
| IMU convention | body_to_world, specific force, ENU (R^2 0.989, found empirically in Module 06) |
| Raw folder inventory | 396 flight folders x 6 files (barometer, gps, raw_gps, groundtruth, imu, magnetometer) = 2,376 CSVs; 144 normal and 252 attacked folders. Headers are identical within each file type and every file has time_usec. Merged contains 360 flights (144 normal, 216 attacked): 36 attacked raw flights are not in Merged (reason under investigation). Raw adds ground truth (lat/lon rad, velocity, attitude), eph/epv, temperature and absolute_pressure |
| Units check (magnetometer, gyro) | magnetic field magnitude 0.48 = gauss, not Tesla; angular velocity consistent with rad/s, not deg/s; quaternion norm 1.0 (one straight normal flight) |