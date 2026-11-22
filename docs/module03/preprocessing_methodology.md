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
