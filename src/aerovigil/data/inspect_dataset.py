from pathlib import Path
import pandas as pd
import json
import re


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

MERGED_ROOT = DATASET_ROOT / "PX4_Simulation_Data" / "Merged"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def identify_scenario(file_path):
    parts = [part.lower() for part in file_path.parts]

    scenario = "unknown"

    for name in ["straight", "curved", "random"]:
        if name in parts:
            scenario = name
            break

    return scenario


def identify_folder_label(file_path):
    parts = [part.lower() for part in file_path.parts]

    if "attacked" in parts:
        return 1

    if "normal" in parts:
        return 0

    return None


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


def normalize_attack_label(df, folder_label):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    possible_labels = [
        "attack",
        "attack enabled",
        "attack_enabled"
    ]

    for name in possible_labels:
        if name in lookup:
            source_column = lookup[name]

            df["attack_label"] = pd.to_numeric(
                df[source_column],
                errors="coerce"
            )

            return

    if folder_label is not None:
        df["attack_label"] = folder_label
    else:
        df["attack_label"] = pd.NA


def main():

    print("=" * 60)
    print("AEROVIGIL - MODULE 02 DATASET INSPECTION")
    print("=" * 60)

    if not MERGED_ROOT.exists():
        raise FileNotFoundError(
            f"Merged dataset directory not found: {MERGED_ROOT}"
        )

    csv_files = sorted(MERGED_ROOT.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found inside the Merged directory."
        )

    print(f"\nTotal CSV files found: {len(csv_files)}")

    manifest = []
    flight_tables = []

    for index, file_path in enumerate(csv_files, start=1):

        print(f"\nInspecting flight {index}: {file_path.name}")

        df = pd.read_csv(file_path)

        scenario = identify_scenario(file_path)
        folder_label = identify_folder_label(file_path)

        flight_id = "__".join
        (
            file_path.relative_to(MERGED_ROOT)
            .with_suffix("")
            .parts
        )

        timestamp_column = identify_timestamp_column(df)

        normalize_attack_label(df, folder_label)

        df["flight_id"] = flight_id
        df["mission_type"] = scenario

        if "timestamp" not in df.columns:
            if timestamp_column is not None:
                df["timestamp"] = df[timestamp_column]
            else:
                df["timestamp"] = pd.NA

        if df["attack_label"].notna().any():
            label_values = sorted(
                df["attack_label"].dropna().unique().tolist()
            )
        else:
            label_values = []

        manifest.append({
            "flight_id": flight_id,
            "file_path": str(file_path.relative_to(PROJECT_ROOT)),
            "mission_type": scenario,
            "folder_label": folder_label,
            "rows": len(df),
            "columns": len(df.columns),
            "timestamp_column": (
                str(timestamp_column)
                if timestamp_column is not None
                else "Not identified"
            ),
            "attack_label_values": json.dumps(label_values),
            "column_names": json.dumps(
                [str(column) for column in df.columns]
            )
        })

        flight_tables.append(df)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Scenario: {scenario}")
        print(f"Attack labels: {label_values}")

    # Create flight manifest
    manifest_df = pd.DataFrame(manifest)

    manifest_path = INTERIM_DIR / "flight_manifest.csv"

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    # Combine all flight tables
    combined_df = pd.concat(
        flight_tables,
        ignore_index=True,
        sort=False
    )

    # Preserve the original columns and add standard metadata.
    combined_path = (
        PROCESSED_DIR / "aerovigil_flights.csv"
    )

    combined_df.to_csv(
        combined_path,
        index=False
    )

    # Generate inspection summary
    summary = {
        "dataset_name": "UAV GPS-Spoofing Dataset",
        "project": "AEROVIGIL",
        "number_of_flights": len(manifest_df),
        "total_rows": len(combined_df),
        "total_columns": len(combined_df.columns),
        "mission_types": sorted(
            manifest_df["mission_type"].unique().tolist()
        ),
        "output_file": str(combined_path)
    }

    summary_path = INTERIM_DIR / "inspection_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 60)

    print(f"Flights processed: {len(manifest_df)}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")

    print("\nGenerated files:")
    print(manifest_path)
    print(combined_path)
    print(summary_path)


if __name__ == "__main__":
    main()