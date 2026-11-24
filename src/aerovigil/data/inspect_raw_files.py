from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "uav_gps_spoofing" / "PX4_Simulation_Data" / "Raw"
)
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

EXPECTED_FILES = [
    "barometer",
    "gps",
    "raw_gps",
    "groundtruth",
    "imu",
    "magnetometer",
]


def identify(parts, names):
    lowered = [part.lower() for part in parts]
    for name in names:
        if name in lowered:
            return name
    return "unknown"


def main():
    print("=" * 60)
    print("AEROVIGIL - RAW SENSOR FILE INVENTORY (read only)")
    print("=" * 60)
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {RAW_ROOT}")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_ROOT.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found inside the Raw directory.")
    print(f"Total raw CSV files: {len(csv_files)}")

    folders = {}
    for file_path in csv_files:
        folders.setdefault(file_path.parent, []).append(file_path)

    rows, headers, variants = [], {}, {}
    for folder, files in sorted(folders.items()):
        relative = folder.relative_to(RAW_ROOT)
        stems = {path.stem.lower(): path for path in files}
        row = {
            "scenario": identify(relative.parts, ["straight", "curved", "random"]),
            "condition": identify(relative.parts, ["attacked", "normal"]),
            "folder": relative.as_posix(),
            "csv_files": len(files),
        }
        for name in EXPECTED_FILES:
            row[name] = name in stems
        row["other_files"] = ";".join(sorted(set(stems) - set(EXPECTED_FILES)))
        rows.append(row)

        # Read only the header line of every file and compare it with the
        # first file of the same type.
        for stem, path in stems.items():
            cols = [str(column) for column in pd.read_csv(path, nrows=0).columns]
            headers.setdefault(stem, cols)
            if cols != headers[stem]:
                variants.setdefault(stem, []).append(str(path))

    inventory = pd.DataFrame(rows)
    inventory.to_csv(INTERIM_DIR / "raw_file_inventory.csv", index=False)
    with open(INTERIM_DIR / "raw_columns.json", "w", encoding="utf-8") as file:
        json.dump(headers, file, indent=4)

    print("\nFlight folders per scenario and condition:")
    print(inventory.groupby(["scenario", "condition"]).size().to_string())

    print("\nFolders containing each expected file:")
    print(inventory[EXPECTED_FILES].sum().astype(int).to_string())

    print("\nColumns of each file type (first file found):")
    for stem, columns in sorted(headers.items()):
        flag = "time_usec present" if "time_usec" in columns else "no time_usec"
        print(f"- {stem}: {len(columns)} columns, {flag}")
        print(f"    {', '.join(columns)}")

    print("\nFiles whose header differs from the first file of the same type:")
    for stem, paths in sorted(variants.items()):
        print(f"- {stem}: {len(paths)} files, first: {paths[0]}")
    if not variants:
        print("none")

    print("\nGenerated files:")
    print(INTERIM_DIR / "raw_file_inventory.csv")
    print(INTERIM_DIR / "raw_columns.json")


if __name__ == "__main__":
    main()
