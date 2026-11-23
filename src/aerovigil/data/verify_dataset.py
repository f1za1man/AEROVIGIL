from pathlib import Path
import pandas as pd
import json


# --------------------------------------------------
# AEROVIGIL - DATASET VERIFICATION
# Memory-efficient verification using CSV chunks
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "aerovigil_flights.csv"

MANIFEST_FILE = PROJECT_ROOT / "data" / "interim" / "flight_manifest.csv"

SUMMARY_FILE = PROJECT_ROOT / "data" / "interim" / "inspection_summary.json"

CHUNK_SIZE = 200_000

REQUIRED_COLUMNS = ["flight_id", "timestamp", "mission_type", "attack_label"]


def main():

    print("=" * 65)
    print("AEROVIGIL - DATASET VERIFICATION")
    print("=" * 65)

    # --------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------

    for file_path in [DATASET_FILE, MANIFEST_FILE, SUMMARY_FILE]:

        if not file_path.exists():

            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("\nInput files found.")

    # --------------------------------------------------
    # LOAD MANIFEST AND SUMMARY
    # --------------------------------------------------

    manifest = pd.read_csv(MANIFEST_FILE)

    with open(SUMMARY_FILE, "r", encoding="utf-8") as file:

        summary = json.load(file)

    print("\nManifest flights:", len(manifest))

    print("Manifest rows:", int(manifest["rows"].sum()))

    # --------------------------------------------------
    # VALIDATE MANIFEST
    # --------------------------------------------------

    if manifest["flight_id"].isna().any():
        raise ValueError("Manifest contains missing flight IDs.")

    if manifest["flight_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate flight IDs.")

    if manifest["file_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate source paths.")

    expected_flight_rows = {
        str(row.flight_id): int(row.rows) for row in manifest.itertuples()
    }

    expected_flight_ids = set(expected_flight_rows)

    if len(expected_flight_ids) != 360:
        raise ValueError(f"Expected 360 flights, found {len(expected_flight_ids)}.")

    # --------------------------------------------------
    # INSPECT CSV HEADER
    # --------------------------------------------------

    header = pd.read_csv(DATASET_FILE, nrows=0).columns.tolist()

    print("\nDataset columns:", len(header))

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]

    if missing_columns:

        raise ValueError(f"Required columns missing: {missing_columns}")

    print("Required columns present.")

    # --------------------------------------------------
    # INITIALIZE COUNTERS
    # --------------------------------------------------

    total_rows = 0

    flight_row_counts = {}

    observed_flight_ids = set()

    mission_counts = {}

    attack_label_counts = {}

    missing_counts = {column: 0 for column in REQUIRED_COLUMNS}

    timestamp_min = None
    timestamp_max = None

    # --------------------------------------------------
    # PROCESS DATASET IN CHUNKS
    # --------------------------------------------------

    print("\nReading dataset in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(DATASET_FILE, usecols=REQUIRED_COLUMNS, chunksize=CHUNK_SIZE),
        start=1,
    ):

        total_rows += len(chunk)

        print(f"Processed chunk {chunk_number}: " f"{total_rows:,} rows")

        # Missing-value checks.

        for column in REQUIRED_COLUMNS:

            missing_counts[column] += int(chunk[column].isna().sum())

        # Flight identifiers.

        ids = chunk["flight_id"].astype("string")

        observed_flight_ids.update(ids.dropna().unique().tolist())

        # Count rows belonging to each flight.

        counts = ids.value_counts(dropna=True)

        for flight_id, count in counts.items():

            flight_row_counts[str(flight_id)] = flight_row_counts.get(
                str(flight_id), 0
            ) + int(count)

        # Mission types.

        for mission, count in chunk["mission_type"].value_counts(dropna=False).items():

            key = str(mission)

            mission_counts[key] = mission_counts.get(key, 0) + int(count)

        # Attack labels.

        for label, count in chunk["attack_label"].value_counts(dropna=False).items():

            key = str(label)

            attack_label_counts[key] = attack_label_counts.get(key, 0) + int(count)

        # Timestamp range.

        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").dropna()

        if not timestamps.empty:

            current_min = timestamps.min()
            current_max = timestamps.max()

            timestamp_min = (
                current_min
                if timestamp_min is None
                else min(timestamp_min, current_min)
            )

            timestamp_max = (
                current_max
                if timestamp_max is None
                else max(timestamp_max, current_max)
            )

    # --------------------------------------------------
    # VALIDATE TOTAL ROW COUNT
    # --------------------------------------------------

    expected_rows = int(manifest["rows"].sum())

    print("\n" + "-" * 65)

    print("TOTAL ROW VALIDATION")

    print("CSV rows:", f"{total_rows:,}")

    print("Manifest rows:", f"{expected_rows:,}")

    if total_rows != expected_rows:

        raise ValueError("CSV row count does not match manifest.")

    # --------------------------------------------------
    # VALIDATE FLIGHT IDENTIFIERS
    # --------------------------------------------------

    print("\nFLIGHT ID VALIDATION")

    print("Unique CSV flight IDs:", len(observed_flight_ids))

    print("Unique manifest flight IDs:", len(expected_flight_ids))

    missing_ids = expected_flight_ids - observed_flight_ids

    unexpected_ids = observed_flight_ids - expected_flight_ids

    if missing_ids:
        raise ValueError(f"Flight IDs missing from CSV: {sorted(missing_ids)[:10]}")

    if unexpected_ids:
        raise ValueError(f"Unexpected flight IDs in CSV: {sorted(unexpected_ids)[:10]}")

    # --------------------------------------------------
    # VALIDATE EACH FLIGHT ROW COUNT
    # --------------------------------------------------

    print("\nPER-FLIGHT ROW VALIDATION")

    row_mismatches = []

    for flight_id, expected_count in expected_flight_rows.items():

        actual_count = flight_row_counts.get(flight_id, 0)

        if actual_count != expected_count:

            row_mismatches.append(
                {
                    "flight_id": flight_id,
                    "expected": expected_count,
                    "actual": actual_count,
                }
            )

    if row_mismatches:

        raise ValueError("Per-flight row count mismatches: " + str(row_mismatches[:10]))

    print("All 360 flight row counts match the manifest.")

    # --------------------------------------------------
    # MISSING VALUE VALIDATION
    # --------------------------------------------------

    print("\nMISSING VALUES")

    for column, count in missing_counts.items():

        print(f"{column}: {count:,}")

    if missing_counts["flight_id"] > 0:

        raise ValueError("Missing flight identifiers detected.")

    if missing_counts["timestamp"] > 0:

        raise ValueError("Missing timestamps detected.")

    if missing_counts["mission_type"] > 0:

        raise ValueError("Missing mission types detected.")

    # Attack labels are checked and reported, not silently filled.

    # --------------------------------------------------
    # REPORT DATA DISTRIBUTION
    # --------------------------------------------------

    print("\nMISSION TYPE DISTRIBUTION")

    for mission, count in sorted(mission_counts.items()):

        print(f"{mission}: {count:,}")

    print("\nATTACK LABEL DISTRIBUTION")

    for label, count in sorted(attack_label_counts.items()):

        print(f"{label}: {count:,}")

    print("\nTIMESTAMP RANGE")

    print("Minimum:", timestamp_min)

    print("Maximum:", timestamp_max)

    # --------------------------------------------------
    # CHECK SUMMARY CONSISTENCY
    # --------------------------------------------------

    if summary["number_of_flights"] != len(manifest):

        raise ValueError("Inspection summary flight count mismatch.")

    if summary["total_rows"] != total_rows:

        raise ValueError("Inspection summary row count mismatch.")

    if summary["total_columns"] != len(header):

        raise ValueError("Inspection summary column count mismatch.")

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 65)

    print("DATASET VERIFICATION PASSED")

    print("=" * 65)

    print("Flights:", len(observed_flight_ids))

    print("Rows:", f"{total_rows:,}")

    print("Columns:", len(header))

    print("Flight ID consistency: PASSED")

    print("Per-flight row consistency: PASSED")

    print("Manifest consistency: PASSED")

    print("Summary consistency: PASSED")

    print("\nAEROVIGIL DATASET IS READY FOR MODULE 03.")


if __name__ == "__main__":

    main()
