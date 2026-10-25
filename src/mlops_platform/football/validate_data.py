
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "football" / "raw"

FILES = [
    "epl_1516.csv",
    "epl_1617.csv",
    "epl_1718.csv",
    "epl_1819.csv",
    "epl_1920.csv",
    "epl_2021.csv",
    "epl_2122.csv",
    "epl_2223.csv",
    "epl_2324.csv",
    "epl_2024_25.csv",
]

REQUIRED = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"
]


def main():
    reports = []
    all_matches = []

    for filename in FILES:
        path = RAW_DIR / filename

        if not path.exists():
            print(f"MISSING FILE: {filename}")
            continue

        df = pd.read_csv(path)
        missing = sorted(set(REQUIRED) - set(df.columns))

        if missing:
            print(f"INVALID SCHEMA: {filename}: {missing}")
            continue

        df = df[REQUIRED].copy()
        df["Date"] = pd.to_datetime(
            df["Date"], dayfirst=True, errors="coerce"
        )
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

        invalid = (
            df[REQUIRED].isna().any(axis=1)
            | ~df["FTR"].isin(["H", "D", "A"])
        )
        duplicates = df.duplicated(
            ["Date", "HomeTeam", "AwayTeam"]
        ).sum()

        valid = df.loc[~invalid].copy()
        valid["source_file"] = filename
        all_matches.append(valid)

        reports.append({
            "file": filename,
            "rows": len(df),
            "valid_rows": len(valid),
            "invalid_rows": int(invalid.sum()),
            "duplicate_fixtures": int(duplicates),
            "teams": df["HomeTeam"].nunique(),
            "first_date": df["Date"].min(),
            "last_date": df["Date"].max(),
        })

    print("\nSEASON VALIDATION")
    print(pd.DataFrame(reports).to_string(index=False))

    if not all_matches:
        raise RuntimeError("No valid match data found.")

    combined = pd.concat(all_matches, ignore_index=True)

    duplicate_keys = ["Date", "HomeTeam", "AwayTeam"]
    duplicates = combined.duplicated(duplicate_keys, keep=False)

    print("\nCOMBINED DATA")
    print(f"Total valid rows: {len(combined)}")
    print(f"Duplicate fixture rows: {int(duplicates.sum())}")
    print(f"Unique teams: {combined['HomeTeam'].nunique()}")
    print(f"Date range: {combined['Date'].min()} to "
          f"{combined['Date'].max()}")

    print("\nRESULT DISTRIBUTION")
    print(combined["FTR"].value_counts())

    print("\nTEAM NAMES TO REVIEW")
    print(sorted(set(combined["HomeTeam"]) | set(combined["AwayTeam"])))

    if duplicates.any():
        print("\nWARNING: duplicate fixture rows need investigation.")
    if combined["Date"].isna().any():
        raise RuntimeError("Unexpected invalid dates remain.")

    output = ROOT / "data" / "football" / "processed"
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "validation_report.csv"
    pd.DataFrame(reports).to_csv(report_path, index=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
