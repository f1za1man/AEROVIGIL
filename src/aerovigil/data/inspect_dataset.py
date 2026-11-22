
from pathlib import Path
import pandas as pd
import json
import os


# --------------------------------------------------
# AEROVIGIL MODULE 02
# Dataset Inspection and Acquisition
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uav_gps_spoofing"
)

MERGED_ROOT = (
    DATASET_ROOT
    / "PX4_Simulation_Data"
    / "Merged"
)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = INTERIM_DIR / "flight_manifest.csv"
SUMMARY_PATH = INTERIM_DIR / "inspection_summary.json"

COMBINED_PATH = (
    PROCESSED_DIR / "aerovigil_flights.csv"
)

# Temporary output files.
# Original output files remain untouched until processing succeeds.

TEMP_COMBINED_PATH = (
    PROCESSED_DIR / "aerovigil_flights.tmp.csv"
)

TEMP_MANIFEST_PATH = (
    INTERIM_DIR / "flight_manifest.tmp.csv"
)

TEMP_SUMMARY_PATH = (
    INTERIM_DIR / "inspection_summary.tmp.json"
)


# --------------------------------------------------
# IDENTIFY MISSION SCENARIO
# --------------------------------------------------

def identify_scenario(file_path):

    parts = [
        part.lower()
        for part in file_path.parts
    ]

    scenario = "unknown"

    for name in ["straight", "curved", "random"]:

        if name in parts:
            scenario = name
            break

    return scenario


# --------------------------------------------------
# IDENTIFY FOLDER LABEL
# --------------------------------------------------

def identify_folder_label(file_path):

    parts = [
        part.lower()
        for part in file_path.parts
    ]

    if "attacked" in parts:
        return 1

    if "normal" in parts:
        return 0

    return None


# --------------------------------------------------
# IDENTIFY TIMESTAMP COLUMN
# --------------------------------------------------

def identify_timestamp_column(df):

    possible_names = [
        "time",
        "timestamp",
        "time_usec",
        "time_us",
        "time_ms"
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name in lookup:
            return lookup[name]

    return None


# --------------------------------------------------
# NORMALIZE ATTACK LABEL
# --------------------------------------------------

def normalize_attack_label(df, folder_label):

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    possible_labels = [
        "attack_label",
        "attack_enabled",
        "attack enabled",
        "attack"
    ]

    for name in possible_labels:

        if name in lookup:

            source_column = lookup[name]

            df["attack_label"] = pd.to_numeric(
                df[source_column],
                errors="coerce"
            )

            return

    # Use the folder label only when no attack
    # indicator column exists in the source file.

    if folder_label is not None:

        df["attack_label"] = folder_label

    else:

        df["attack_label"] = pd.NA


# --------------------------------------------------
# DISCOVER ALL OUTPUT COLUMNS
# --------------------------------------------------

def discover_output_columns(csv_files):

    print("\nDiscovering dataset columns...")

    output_columns = []

    for file_path in csv_files:

        source_columns = pd.read_csv(
            file_path,
            nrows=0
        ).columns.tolist()

        for column in source_columns:

            if column not in output_columns:
                output_columns.append(column)

    # Add standardized metadata columns if missing.

    required_metadata = [
        "attack_label",
        "flight_id",
        "mission_type",
        "timestamp"
    ]

    for column in required_metadata:

        if column not in output_columns:
            output_columns.append(column)

    print(f"Total output columns identified: {len(output_columns)}")

    return output_columns


# --------------------------------------------------
# MAIN DATASET INSPECTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("AEROVIGIL - MODULE 02 DATASET INSPECTION")
    print("=" * 65)

    if not MERGED_ROOT.exists():

        raise FileNotFoundError(
            f"Merged dataset directory not found: {MERGED_ROOT}"
        )

    csv_files = sorted(
        MERGED_ROOT.rglob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV files found inside the Merged directory."
        )

    print(f"\nTotal CSV files found: {len(csv_files)}")

    # --------------------------------------------------
    # CREATE UNIQUE FLIGHT IDENTIFIERS
    # --------------------------------------------------

    flight_ids = []

    for file_path in csv_files:

        flight_id = "__".join(
            file_path.relative_to(MERGED_ROOT)
            .with_suffix("")
            .parts
        )

        flight_ids.append(flight_id)

    if len(flight_ids) != len(set(flight_ids)):

        raise ValueError(
            "Duplicate flight identifiers detected. "
            "Dataset processing stopped."
        )

    print(f"Unique flight identifiers: {len(set(flight_ids))}")

    # --------------------------------------------------
    # DETERMINE COMPLETE OUTPUT SCHEMA
    # --------------------------------------------------

    output_columns = discover_output_columns(csv_files)

    # --------------------------------------------------
    # INITIALIZE PROCESSING
    # --------------------------------------------------

    manifest = []

    total_rows = 0

    try:

        # Remove an old temporary output, if one exists.

        if TEMP_COMBINED_PATH.exists():
            TEMP_COMBINED_PATH.unlink()

        if TEMP_MANIFEST_PATH.exists():
            TEMP_MANIFEST_PATH.unlink()

        if TEMP_SUMMARY_PATH.exists():
            TEMP_SUMMARY_PATH.unlink()

        # --------------------------------------------------
        # PROCESS FLIGHTS ONE AT A TIME
        # --------------------------------------------------

        for index, file_path in enumerate(csv_files, start=1):

            print("\n" + "-" * 65)

            print(
                f"Processing flight {index}/{len(csv_files)}: "
                f"{file_path.name}"
            )

            df = pd.read_csv(file_path)

            scenario = identify_scenario(file_path)

            folder_label = identify_folder_label(file_path)

            # IMPORTANT:
            # Use the relative folder structure and filename
            # to create a unique string identifier.
            #
            # Example:
            # Curved__Attacked__log_21_04_2023_07_21_03

            flight_id = "__".join(
                file_path.relative_to(MERGED_ROOT)
                .with_suffix("")
                .parts
            )

            timestamp_column = identify_timestamp_column(df)

            normalize_attack_label(
                df,
                folder_label
            )

            # Add standardized metadata.

            df["flight_id"] = flight_id

            df["mission_type"] = scenario

            # Preserve an existing timestamp column.
            # Otherwise, copy the identified timestamp source.

            if "timestamp" not in df.columns:

                if timestamp_column is not None:

                    df["timestamp"] = df[timestamp_column]

                else:

                    df["timestamp"] = pd.NA

            # Collect attack label information.

            if df["attack_label"].notna().any():

                label_values = sorted(
                    df["attack_label"]
                    .dropna()
                    .unique()
                    .tolist()
                )

            else:

                label_values = []

            # Record flight information in manifest.

            manifest.append({

                "flight_id": flight_id,

                "file_path": str(
                    file_path.relative_to(PROJECT_ROOT)
                ),

                "mission_type": scenario,

                "folder_label": folder_label,

                "rows": len(df),

                "columns": len(df.columns),

                "timestamp_column": (
                    str(timestamp_column)
                    if timestamp_column is not None
                    else "Not identified"
                ),

                "attack_label_values": json.dumps(
                    label_values
                ),

                "column_names": json.dumps(
                    [str(column) for column in df.columns]
                )

            })

            # Align this flight with the complete dataset schema.
            # Missing columns in an individual flight become NaN.

            df = df.reindex(
                columns=output_columns
            )

            # Write this flight directly to the temporary CSV.
            # This avoids keeping all flight DataFrames in RAM.

            df.to_csv(

                TEMP_COMBINED_PATH,

                mode="w" if index == 1 else "a",

                header=(index == 1),

                index=False

            )

            total_rows += len(df)

            print(f"Rows: {len(df)}")

            print(f"Columns: {len(df.columns)}")

            print(f"Flight ID: {flight_id}")

            print(f"Mission type: {scenario}")

            print(f"Attack labels: {label_values}")

            print(f"Total rows processed: {total_rows:,}")

            # Release the current flight DataFrame.

            del df

        # --------------------------------------------------
        # GENERATE AND VALIDATE MANIFEST
        # --------------------------------------------------

        manifest_df = pd.DataFrame(manifest)

        if len(manifest_df) != len(csv_files):

            raise ValueError(
                "Manifest flight count does not match "
                "the number of source files."
            )

        if manifest_df["flight_id"].nunique() != len(csv_files):

            raise ValueError(
                "Manifest contains duplicate flight identifiers."
            )

        manifest_rows = int(
            manifest_df["rows"].sum()
        )

        if manifest_rows != total_rows:

            raise ValueError(
                f"Row count mismatch: manifest={manifest_rows}, "
                f"processed={total_rows}"
            )

        manifest_df.to_csv(
            TEMP_MANIFEST_PATH,
            index=False
        )

        # --------------------------------------------------
        # GENERATE INSPECTION SUMMARY
        # --------------------------------------------------

        summary = {

            "dataset_name": "UAV GPS-Spoofing Dataset",

            "project": "AEROVIGIL",

            "number_of_flights": len(manifest_df),

            "total_rows": total_rows,

            "total_columns": len(output_columns),

            "unique_flight_ids": int(
                manifest_df["flight_id"].nunique()
            ),

            "mission_types": sorted(
                manifest_df["mission_type"]
                .unique()
                .tolist()
            ),

            "output_file": str(COMBINED_PATH)

        }

        with open(
            TEMP_SUMMARY_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

        # --------------------------------------------------
        # FINAL OUTPUT REPLACEMENT
        # --------------------------------------------------
        # Existing files are replaced only after all processing
        # and manifest checks have completed successfully.

        os.replace(
            TEMP_COMBINED_PATH,
            COMBINED_PATH
        )

        os.replace(
            TEMP_MANIFEST_PATH,
            MANIFEST_PATH
        )

        os.replace(
            TEMP_SUMMARY_PATH,
            SUMMARY_PATH
        )

    except Exception:

        # Remove incomplete temporary outputs.
        # Existing final outputs remain available if processing
        # fails before the replacement stage.

        for temporary_file in [
            TEMP_COMBINED_PATH,
            TEMP_MANIFEST_PATH,
            TEMP_SUMMARY_PATH
        ]:

            if temporary_file.exists():
                temporary_file.unlink()

        raise

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n" + "=" * 65)

    print("DATASET INSPECTION COMPLETED SUCCESSFULLY")

    print("=" * 65)

    print(f"Flights processed: {len(manifest_df)}")

    print(f"Unique flight IDs: {manifest_df['flight_id'].nunique()}")

    print(f"Total rows: {total_rows:,}")

    print(f"Total columns: {len(output_columns)}")

    print("\nGenerated files:")

    print(MANIFEST_PATH)

    print(COMBINED_PATH)

    print(SUMMARY_PATH)

    print("\nAEROVIGIL MODULE 02 PROCESSING PASSED.")


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    main()