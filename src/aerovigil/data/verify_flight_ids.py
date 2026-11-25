from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "aerovigil_flights.csv"
MANIFEST_FILE = PROJECT_ROOT / "data" / "interim" / "flight_manifest.csv"
CHUNK_SIZE = 500_000


def main():
    print("=" * 60)
    print("AEROVIGIL - FLIGHT IDENTIFIER VERIFICATION")
    print("=" * 60)
    manifest = pd.read_csv(MANIFEST_FILE)
    expected = manifest.set_index("flight_id")["rows"]
    problems = []

    if not manifest["flight_id"].is_unique:
        problems.append("Duplicate flight IDs in the manifest.")

    # Only the flight_id column is read, in chunks, to keep memory low.
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(DATASET_FILE, usecols=["flight_id"], chunksize=CHUNK_SIZE):
        counts = counts.add(chunk["flight_id"].value_counts(), fill_value=0)
    counts = counts.astype("int64")

    print(f"Manifest flights : {len(expected)}")
    print(f"CSV flight IDs   : {len(counts)}")
    print(f"Manifest rows    : {int(expected.sum()):,}")
    print(f"CSV rows         : {int(counts.sum()):,}")

    if set(counts.index) != set(expected.index):
        extra = sorted(set(counts.index) - set(expected.index))[:3]
        missing = sorted(set(expected.index) - set(counts.index))[:3]
        problems.append(
            f"IDs differ from the manifest. Extra: {extra} Missing: {missing}"
        )
    if any("built-in method" in str(i) or " at 0x" in str(i) for i in counts.index):
        problems.append("Flight IDs contain a Python object representation.")
    if len(counts) != len(expected):
        problems.append("The number of unique flight IDs does not match the manifest.")
    mismatch = (counts.reindex(expected.index).fillna(-1) != expected).sum()
    if mismatch:
        problems.append(
            f"{int(mismatch)} flights have a row count different from the manifest."
        )

    if problems:
        print("\nFLIGHT ID VERIFICATION FAILED")
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    print("\nAll flight row counts match the manifest.")
    print("FLIGHT ID VERIFICATION PASSED.")


if __name__ == "__main__":
    main()
