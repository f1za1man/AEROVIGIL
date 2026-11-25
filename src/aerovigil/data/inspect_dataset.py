"""AEROVIGIL Module 02: inspect the merged dataset and build the combined table.

Reads every CSV under data/raw/uav_gps_spoofing/PX4_Simulation_Data/Merged, adds the
columns attack_label, flight_id, mission_type and timestamp, and writes:
    data/interim/flight_manifest.csv
    data/interim/inspection_summary.json
    data/processed/aerovigil_flights.csv

Usage:
    python -m aerovigil.data.inspect_dataset
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing"
MERGED_ROOT = DATASET_ROOT / "PX4_Simulation_Data" / "Merged"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SCENARIOS = ["straight", "curved", "random"]
TIMESTAMP_NAMES = ["time", "timestamp", "time_usec", "time_us", "time_ms"]
LABEL_NAMES = ["attack_label", "attack_enabled", "attack enabled", "attack"]


def folder_parts(file_path):
    """Lower-case folder names below the Merged root (never the project path)."""
    return [part.lower() for part in file_path.relative_to(MERGED_ROOT).parts]


def identify_scenario(file_path):
    parts = folder_parts(file_path)
    for name in SCENARIOS:
        if name in parts:
            return name
    return "unknown"


def identify_folder_label(file_path):
    parts = folder_parts(file_path)
    if "attacked" in parts:
        return 1
    if "normal" in parts:
        return 0
    return None


def identify_timestamp_column(df):
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for name in TIMESTAMP_NAMES:
        if name in lookup:
            return lookup[name]
    return None


def normalize_attack_label(df, file_path):
    """Create attack_label from the file's own attack column.

    The folder name is deliberately NOT used as a fallback: in an attacked flight the
    attack starts after take-off, so labelling every row as an attack would be wrong.
    """
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for name in LABEL_NAMES:
        if name in lookup:
            df["attack_label"] = pd.to_numeric(df[lookup[name]], errors="coerce")
            return
    raise ValueError(
        f"No attack column found in {file_path.name}. Refusing to label every row "
        "of an attacked flight as an attack, because attacks start after take-off."
    )


def make_flight_id(file_path):
    """<mission>__<Attacked|Normal>__<log name>, unique across the dataset."""
    relative = file_path.relative_to(MERGED_ROOT).with_suffix("")
    return "__".join(relative.parts)


def main():
    print("=" * 60)
    print("AEROVIGIL - MODULE 02 DATASET INSPECTION")
    print("=" * 60)
    if not MERGED_ROOT.exists():
        raise FileNotFoundError(f"Merged dataset directory not found: {MERGED_ROOT}")
    csv_files = sorted(MERGED_ROOT.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found inside the Merged directory.")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTotal CSV files found: {len(csv_files)}")

    manifest = []
    flight_tables = []
    for index, file_path in enumerate(csv_files, start=1):
        print(f"\nInspecting flight {index}: {file_path.name}")
        df = pd.read_csv(file_path)
        scenario = identify_scenario(file_path)
        folder_label = identify_folder_label(file_path)
        flight_id = make_flight_id(file_path)
        timestamp_column = identify_timestamp_column(df)
        normalize_attack_label(df, file_path)
        df["flight_id"] = flight_id
        df["mission_type"] = scenario
        if "timestamp" not in df.columns:
            if timestamp_column is None:
                raise ValueError(f"No timestamp column found in {file_path.name}.")
            df["timestamp"] = df[timestamp_column]

        label_values = sorted(df["attack_label"].dropna().unique().tolist())
        manifest.append(
            {
                "flight_id": flight_id,
                "file_path": str(file_path.relative_to(PROJECT_ROOT)),
                "mission_type": scenario,
                "folder_label": folder_label,
                "rows": len(df),
                "columns": len(df.columns),
                "timestamp_column": str(timestamp_column),
                "attack_label_values": json.dumps(label_values),
                "column_names": json.dumps([str(column) for column in df.columns]),
            }
        )
        flight_tables.append(df)
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Scenario: {scenario}")
        print(f"Attack labels: {label_values}")

    manifest_df = pd.DataFrame(manifest)
    # Guard against the defect found in Module 03 (every flight_id the same text).
    ids = manifest_df["flight_id"]
    if not ids.is_unique:
        raise ValueError("Flight identifiers are not unique.")
    if ids.str.contains("built-in method| at 0x", regex=True).any():
        raise ValueError("Flight identifiers contain a Python object representation.")
    manifest_path = INTERIM_DIR / "flight_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    combined_df = pd.concat(flight_tables, ignore_index=True, sort=False)
    combined_path = PROCESSED_DIR / "aerovigil_flights.csv"
    combined_df.to_csv(combined_path, index=False)

    summary = {
        "dataset_name": "UAV GPS-Spoofing Dataset",
        "project": "AEROVIGIL",
        "number_of_flights": len(manifest_df),
        "total_rows": len(combined_df),
        "total_columns": len(combined_df.columns),
        "mission_types": sorted(manifest_df["mission_type"].unique().tolist()),
        "output_file": str(combined_path),
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
