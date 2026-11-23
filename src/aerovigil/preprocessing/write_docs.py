from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "module03"
META = ROOT / "data" / "processed" / "module03" / "feature_metadata.csv"
DOCS.mkdir(parents=True, exist_ok=True)


def write(name, text):
    (DOCS / name).write_text(text.strip() + "\n", encoding="utf-8")
    print("Wrote", DOCS / name)


REPORT = """
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
"""

METHODOLOGY = """
# Preprocessing Methodology

## Principles

- The original CSV is never modified. Every transformation is reproducible from scripts and configs/preprocessing/preprocessing.yaml.
- Each flight is processed independently.
- Attack labels are preserved and never used to calculate a feature.
- Anomalies are not silently removed.

## Timestamp findings

- time_usec (microseconds) and timestamp are identical for all rows; time_usec is used.
- 298 flights run at about 250 Hz and 62 at about 125 Hz. Sampling rate is spread across all mission types and has a similar attack fraction in both groups, so it does not act as a shortcut.
- All 360 flights contain locally out-of-order rows (140,137 rows, about 1.8%). Backward jumps are typically about 0.2 s. Rows are sorted by time within each flight; no rows are removed. There are no duplicate timestamps.
- 92 large gaps (more than 10 times the median interval) occur in 83 flights, typically about 16 s and between 48% and 94% of the way through a flight. The label before and after every gap is the same, so gaps are not label boundaries.

## Cleaning

Rows are sorted by time_usec within each flight, numeric columns are converted, and missing values, duplicates and gaps are counted. Result: 7,893,049 rows in and out, 4,302,145 attack rows in and out, no missing values, and no non-numeric values. Output: cleaned/ (one Parquet per flight) and cleaning_report.csv.

## Resampling

Resampling of the dataset is disabled (resampling_enabled: false). Interpolating across 16 s dropouts would invent data. Windows are the only place where values are interpolated, and only over normal sampling jitter (see Windowing).

## Split

Flights are split 70/15/15 (252/54/54), stratified by mission type and by whether the flight contains an attack, using seed 42. Every mission type and both flight classes appear in every partition.

## Normalization

Standardization z = (x - mean) / std, fitted on the 252 training flights only (NaN ignored, population standard deviation). Parameters and feature order are in normalization_parameters.json. Physical-unit features remain available in features/.

## Windowing

- Window duration 5 s, step 2.5 s (50% overlap), defined in time because sampling is irregular and mixed.
- Each window is interpolated linearly onto a regular 125 Hz grid (625 samples).
- A window is dropped if it contains a gap larger than 0.05 s, lacks data near either edge, or contains a NaN feature.
- Of 17,668 candidate windows, 756 were dropped for gaps or coverage and 427 for NaN features; 16,485 remain.
- Each window records its flight, start and end time, attack fraction, any-attack flag, whether an attack starts inside the window, the onset offset, and seconds since the flight's first attack sample.

## PCA

Each window is summarised by the mean of each normalized feature. PCA is fitted on training windows only and applied to validation and test windows. Points are coloured by attack content for inspection only.
"""

QUALITY = """
# Data Quality Report

| Check | Result |
|---|---|
| Flights | 360 |
| Rows | 7,893,049 |
| Columns | 27 |
| Missing values (raw columns) | 0 |
| Non-numeric values | 0 |
| Duplicate timestamps | 0 |
| Out-of-order rows | 140,137 (all 360 flights), corrected by sorting |
| Large gaps | 92 in 83 flights, longest 22.8 s, recorded and not repaired |
| time_usec vs timestamp | identical for all rows |
| Attack labels | 4,302,145 attack / 3,590,904 normal rows, preserved |
| Flights with no attack | 144 |
| Flights with attack | 216 (one transition: 153, two transitions: 63) |
| Sampling rate | 298 flights at 250 Hz, 62 flights at 125 Hz |
| attack_enabled vs attack_label | same distribution; attack_enabled is excluded from features |

## Engineered feature missing values

About 1% of rows have NaN in the lag-based features (the first second of each flight and the second after each gap) and about 2% in the rate features. Windows containing NaN are excluded.

## Physical validity of the substitute velocity features

On normal rows the median horizontal-velocity residual is 0.05 m/s, the median vertical-velocity residual is 0.005 m/s, and the median GPS/barometer altitude difference is 0.82 m (95th percentile 0.97 m). These are small and stable, so the units and reference frames are compatible.

## Observations

- Altitude and vertical-velocity residuals are the same for attack and normal rows to four decimals.
- The horizontal-velocity residual differs (95th percentile 0.10 for normal rows, 1.04 for attack rows).
- res_altitude and res_altitude_abs are identical because the offset is always positive.
- Validation and test normalized means stay near 0 and standard deviations near 1; the horizontal-velocity features have standard deviation about 1.06 there.
"""


def dictionary():
    meta = pd.read_csv(META).fillna("")

    def esc(x):
        return str(x).replace("|", "\\|")

    lines = [
        "# Feature Dictionary",
        "",
        "Generated from feature_metadata.csv. Features marked unavailable were "
        "not generated.",
        "",
        "Parameters: lag L = 1 s, rolling window = 2 s (trailing, causal).",
        "",
    ]
    for _, r in meta.iterrows():
        lines += [
            f"## {esc(r['feature'])}",
            "",
            f"- Formula: {esc(r['formula'])}",
            f"- Source columns: {esc(r['source_columns'])}",
            f"- Units: {esc(r['units'])}",
            f"- Assumptions: {esc(r['assumptions'])}",
            f"- Missing-value behaviour: {esc(r['missing_value_behaviour'])}",
            f"- Available for all flights: {esc(r['available_for_all_flights'])}",
            "",
        ]
    return "\n".join(lines)


def main():
    write("module03_report.md", REPORT)
    write("preprocessing_methodology.md", METHODOLOGY)
    write("data_quality_report.md", QUALITY)
    write("feature_dictionary.md", dictionary())


if __name__ == "__main__":
    main()
