from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = ROOT / "data" / "processed" / "aerovigil_flights.csv"
OUTPUT_DIR = ROOT / "data" / "interim" / "module03"

CHUNK_SIZE = 200_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_timestamp_column(name):
    name = name.lower()
    terms = ("time", "timestamp", "stamp", "usec", "msec", "epoch")
    return any(term in name for term in terms)


def is_label_column(name):
    name = name.lower()
    terms = ("label", "attack", "spoof", "class", "target")
    return any(term in name for term in terms)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Combined dataset not found: {INPUT_FILE}")

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

    total_rows = 0
    column_types = {}
    null_counts = {}
    numeric_stats = {}
    categorical_values = {}
    first_values = {}
    last_values = {}

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        for column in chunk.columns:
            series = chunk[column]

            column_types.setdefault(column, set()).add(str(series.dtype))

            null_counts[column] = null_counts.get(column, 0) + int(series.isna().sum())

            if column not in first_values:
                first_values[column] = series.dropna().head(5).astype(str).tolist()

            if not series.dropna().empty:
                last_values[column] = series.dropna().tail(5).astype(str).tolist()

            if pd.api.types.is_numeric_dtype(series):
                valid = pd.to_numeric(series, errors="coerce").dropna()

                if not valid.empty:
                    stats = numeric_stats.setdefault(
                        column,
                        {
                            "min": float("inf"),
                            "max": float("-inf"),
                            "sum": 0.0,
                            "count": 0,
                        },
                    )

                    stats["min"] = min(stats["min"], float(valid.min()))
                    stats["max"] = max(stats["max"], float(valid.max()))
                    stats["sum"] += float(valid.sum())
                    stats["count"] += int(valid.count())

            if is_label_column(column):
                values = series.dropna().astype(str).value_counts().to_dict()

                counts = categorical_values.setdefault(column, {})

                for value, count in values.items():
                    counts[value] = counts.get(value, 0) + int(count)

        print(f"Processed chunk {chunk_number}; " f"rows so far: {total_rows:,}")

    for column, stats in numeric_stats.items():
        stats["mean"] = stats["sum"] / stats["count"] if stats["count"] else None
        del stats["sum"]

    report = {
        "input_file": str(INPUT_FILE),
        "total_rows": total_rows,
        "columns": list(column_types.keys()),
        "column_types": {key: sorted(value) for key, value in column_types.items()},
        "null_counts": null_counts,
        "numeric_statistics": numeric_stats,
        "possible_label_values": categorical_values,
        "timestamp_columns": [c for c in column_types if is_timestamp_column(c)],
        "possible_label_columns": [c for c in column_types if is_label_column(c)],
        "first_values": first_values,
        "last_values": last_values,
    }

    json_path = OUTPUT_DIR / "schema_profile.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    column_table = []

    for column in column_types:
        column_table.append(
            {
                "column": column,
                "dtype": ", ".join(sorted(column_types[column])),
                "missing_count": null_counts.get(column, 0),
                "missing_percentage": (
                    null_counts.get(column, 0) / total_rows * 100 if total_rows else 0
                ),
                "first_values": str(first_values.get(column, [])),
            }
        )

    pd.DataFrame(column_table).to_csv(OUTPUT_DIR / "column_profile.csv", index=False)

    print("\nDATASET PROFILE COMPLETED")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(column_types)}")
    print(f"Report: {json_path}")


if __name__ == "__main__":
    main()
