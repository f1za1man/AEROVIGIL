# AEROVIGIL — Dataset Data Card

## 1. Dataset Identification

**Dataset Name:** UAV GPS-Spoofing Dataset (PX4 Simulation)

**Project:** AEROVIGIL

**Dataset Type:** Simulated UAV flight telemetry

**Primary Source:**  
https://github.com/anthony-finn/UAV-GPS-Spoofing-Dataset

**Original Simulation Environment:** PX4 SITL and Gazebo Classic

## 2. Dataset Description

The dataset contains simulated UAV flight measurements collected using a quadcopter in a PX4 SITL and Gazebo Classic environment.

It includes normal and GPS-spoofed flight scenarios. The attack modifies the reported GPS position incrementally, potentially causing the reported position to deviate from the actual simulated position.

## 3. Available Sensor Categories

- Global Positioning System (GPS)
- Inertial Measurement Unit (IMU)
- Barometer
- Magnetometer
- Ground-truth flight measurements
- Raw GPS measurements

## 4. Flight Scenarios

The dataset contains three flight-plan categories:

- Straight
- Curved
- Random

Each category includes normal and attacked flight scenarios.

## 5. Sampling Frequency

The dataset documentation reports an approximate sampling frequency of 250 Hz.

The actual sampling frequency of individual sensor streams will be verified during timestamp inspection and preprocessing.

## 6. Data Labelling

The dataset provides an attack indicator identifying whether GPS spoofing is enabled.

Expected interpretation:

- 0: Normal operation
- 1: Attack enabled

The exact values and their distribution will be verified against the downloaded CSV files.

## 7. Dataset Versions

The dataset provides:

- Raw data: Individual sensor streams, including ground truth and raw GPS measurements.
- Merged data: Time-synchronized sensor measurements.
- Preprocessed data: Data containing derived features used in the original research.

AEROVIGIL retains the raw data and uses the merged data for initial dataset inspection.

## 8. Permission and Usage

The dataset author has confirmed permission for non-commercial academic research use.

The original dataset will not be redistributed through the AEROVIGIL GitHub repository.

The author's confirmation will be retained as project documentation.

## 9. Data Integrity

- Original downloaded files remain unchanged.
- Raw data is excluded from GitHub.
- Data processing is performed using reproducible Python scripts.
- Dataset inspection results are generated programmatically.
- Flight identifiers are maintained to distinguish individual missions.

## 10. Limitations

- The dataset is based on simulation rather than real-world drone recordings.
- Raw sensor streams are not guaranteed to be time-synchronized.
- The available flight scenarios may not represent every real UAV mission.
- The dataset does not provide real-world radio-frequency GPS signal measurements.
- Sensor sampling rates and label meanings require verification against the actual files.

## 11. AEROVIGIL Usage

The dataset will be used for:

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

The original dataset and research paper will be cited in the final project report.