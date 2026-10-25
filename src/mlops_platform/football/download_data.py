
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "football" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
]

# Season code: 1516 = 2015-16, 2324 = 2023-24.
SEASONS = [
    ("1516", "2015-16"),
    ("1617", "2016-17"),
    ("1718", "2017-18"),
    ("1819", "2018-19"),
    ("1920", "2019-20"),
    ("2021", "2020-21"),
    ("2122", "2021-22"),
    ("2223", "2022-23"),
    ("2324", "2023-24"),
]


def download_season(code, season_name):
    filename = f"epl_{code}.csv"
    output_path = RAW_DIR / filename
    url = f"{BASE_URL}/{code}/E0.csv"

    print(f"\nSeason {season_name}")

    # Preserve an existing file rather than silently overwriting it.
    if output_path.exists():
        print(f"Already exists; skipping: {output_path.name}")
        return {
            "season": season_name,
            "status": "skipped_existing",
            "matches": None,
        }

    request = Request(
        url,
        headers={"User-Agent": "MLOpsFootballResearch/1.0"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()

    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Download failed: {exc}")
        return {
            "season": season_name,
            "status": "download_failed",
            "matches": 0,
        }

    # Parse and validate before writing the file.
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        print(f"CSV parsing failed: {exc}")
        return {
            "season": season_name,
            "status": "invalid_csv",
            "matches": 0,
        }

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    if missing:
        print(f"Missing required columns: {missing}")
        return {
            "season": season_name,
            "status": "invalid_schema",
            "matches": 0,
        }

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    if df.empty:
        print("No valid matches found.")
        return {
            "season": season_name,
            "status": "empty",
            "matches": 0,
        }

    if not df["FTR"].isin(["H", "D", "A"]).all():
        print("Unexpected result labels found.")
        return {
            "season": season_name,
            "status": "invalid_results",
            "matches": 0,
        }

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])

    duplicate_count = df.duplicated(
        subset=["Date", "HomeTeam", "AwayTeam"]
    ).sum()

    if duplicate_count:
        print(f"Warning: {duplicate_count} duplicate fixtures detected.")

    # Write only after the basic checks have passed.
    df.to_csv(output_path, index=False)

    print(f"Downloaded matches: {len(df)}")
    print(f"Saved: {output_path}")

    return {
        "season": season_name,
        "status": "downloaded",
        "matches": len(df),
    }


def main():
    print("=" * 60)
    print("EPL HISTORICAL DATA DOWNLOADER")
    print("=" * 60)
    print(f"Destination: {RAW_DIR}")

    results = []

    for code, season_name in SEASONS:
        results.append(download_season(code, season_name))
        time.sleep(1)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    successful = summary["status"].isin(
        ["downloaded", "skipped_existing"]
    ).sum()

    print(f"\nSeasons processed: {len(summary)}")
    print(f"Successful or already present: {successful}")

    failed = summary[
        ~summary["status"].isin(["downloaded", "skipped_existing"])
    ]

    if not failed.empty:
        print("\nSome seasons need attention. Review the errors above.")


if __name__ == "__main__":
    main()
