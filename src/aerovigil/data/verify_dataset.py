from pathlib import Path
import pandas as pd
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aerovigil_flights.csv"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "flight_manifest.csv"
)


def main():

    print("=" * 60)
    print("AEROVIGIL - DATASET VERIFICATION")
    print("=" * 60)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            "Combined dataset does not exist."
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "Flight manifest does not exist."
        )

    df = pd.read_csv(DATASET_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)

    print("\nDataset dimensions:")
    print(df.shape)

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFlight count:")
    print(df["flight_id"].nunique())

    print("\nMission types:")
    print(df["mission_type"].value_counts())

    print("\nAttack label distribution:")
    print(df["attack_label"].value_counts(dropna=False))

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(15))

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nManifest records:")
    print(len(manifest))

    required_columns = [
        "flight_id",
        "timestamp",
        "mission_type",
        "attack_label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["flight_id"].isna().any():
        raise ValueError("Missing flight identifiers detected.")

    print("\nDATASET VERIFICATION PASSED.")


if __name__ == "__main__":
    main()