# AEROVIGIL Module 03 Report: Preprocessing and Sensor Consistency Features

## Introduction

Module 03 prepares the acquired UAV telemetry for GPS spoofing detection. It verifies the dataset structure, examines timestamps and sampling behaviour, handles data-quality issues, and derives sensor-consistency features for the machine-learning and statistical detection modules that follow.

## Dataset

- 360 flights, 7,893,049 rows, 27 columns (combined CSV, about 2.3 GB).
- Mission types: curved (4,362,481 rows), random (2,741,656), straight (788,912).
- Attack labels: 3,590,904 normal rows and 4,302,145 attack rows.
- A defect in Module 02's output was found and fixed during this module: the flight_id column contained a Python method representation instead of real identifiers. After regeneration, 360 distinct identifiers matched the manifest and all per-flight row counts.

## Pipeline

1. Profile the schema and verify the dataset (verify_dataset.py, profile_dataset.py).
2. Verify timestamps and sampling behaviour (inspect_timestamps.py, timestamp_report.csv).
3. Clean flights independently and write one Parquet file per flight (clean_flights.py).
4. Build sensor-consistency features (build_features.py).
5. Split by flight into train, validation and test (split_flights.py).
6. Fit normalization on training flights only and apply it to all flights (normalize_features.py).
7. Generate 5 s windows with 50% overlap (make_windows.py).
8. Explore the feature space with PCA fitted on training windows (pca_analysis.py).
9. Run the automated test suite (python -m pytest -v): 19 tests passed.

## Key results

- 22 engineered features per sample; 16,485 windows of 625 samples x 22 features.
- Windows per split: train 11,579 (252 flights), validation 2,451 (54 flights), test 2,455 (54 flights).
- Attack content per split (mean attack fraction): train 0.541, validation 0.533, test 0.550.
- Windows containing an attack onset: train 213, validation 47, test 44.
- PCA: PC1 explains 44.2% of variance and is dominated by the horizontal-velocity residual features; PC2 explains 13.1% and is dominated by GPS-derived ground speed.

## Data leakage prevention

Complete flights are assigned to training, validation and testing partitions before windows are generated. The split is stratified by mission type and by whether the flight contains an attack, with seed 42. Normalization parameters and the PCA model are fitted on training data only and reused for the other partitions. Attack labels are carried alongside the data for evaluation and are never part of the feature matrix; this is enforced by automated tests.

## Limitations

- The dataset is based on simulated UAV flights.
- The GPS/IMU integrated-velocity residual and the GPS course/magnetometer heading residual were not generated, because the attitude, gravity and magnetometer conventions are not documented. They are marked unavailable instead of being fabricated.
- The altitude and vertical-velocity residual features showed no difference between attack and normal rows. They are physically valid but may not help detection.
- The 92 large timestamp gaps (about 16 s each) look like a logging artifact. They were recorded and excluded from windows, not repaired.
- PCA separation is exploratory and does not show that a detector will generalize. Later modules must test unseen flights and attack patterns.
