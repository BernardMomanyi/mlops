from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "football" / "raw"
OUTPUT_FILE = ROOT / "data" / "football" / "processed" / "epl_features.csv"

SEASON_FILES = [
    ("1516", "2015-16"),
    ("1617", "2016-17"),
    ("1718", "2017-18"),
    ("1819", "2018-19"),
    ("1920", "2019-20"),
    ("2021", "2020-21"),
    ("2122", "2021-22"),
    ("2223", "2022-23"),
    ("2324", "2023-24"),
    ("2024_25", "2024-25"),
]

WINDOW = 5

REQUIRED = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"
]

TEAM_ALIASES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
}


def parse_dates(series):
    """Parse dates consistently across historical CSV formats."""
    dates = pd.to_datetime(
        series, format="%d/%m/%Y", errors="coerce"
    )

    # Handle any rows that use a different recognizable format.
    missing = dates.isna()
    if missing.any():
        dates.loc[missing] = pd.to_datetime(
            series.loc[missing], dayfirst=True, errors="coerce"
        )

    return dates


def load_matches():
    frames = []

    for code, season in SEASON_FILES:
        path = RAW_DIR / f"epl_{code}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing season file: {path}")

        df = pd.read_csv(path)

        missing_columns = set(REQUIRED) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} missing columns: {sorted(missing_columns)}"
            )

        df = df[REQUIRED].copy()
        df["season"] = season
        df["source_file"] = path.name

        df["Date"] = parse_dates(df["Date"])
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

        for column in ["HomeTeam", "AwayTeam", "FTR"]:
            df[column] = df[column].astype("string").str.strip()

        df["HomeTeam"] = df["HomeTeam"].replace(TEAM_ALIASES)
        df["AwayTeam"] = df["AwayTeam"].replace(TEAM_ALIASES)
        df["FTR"] = df["FTR"].str.upper()

        if df[REQUIRED].isna().any().any():
            bad_rows = df[df[REQUIRED].isna().any(axis=1)]
            raise ValueError(
                f"{path.name} contains missing or invalid required values:\n"
                f"{bad_rows.head(10).to_string(index=False)}"
            )

        if not df["FTR"].isin(["H", "D", "A"]).all():
            raise ValueError(f"Unexpected result labels in {path.name}")

        if (
            (df[["FTHG", "FTAG"]] < 0).any().any()
            or (df[["FTHG", "FTAG"]] % 1 != 0).any().any()
        ):
            raise ValueError(
                f"Invalid goal values in {path.name}"
            )

        if df.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
            raise ValueError(
                f"Duplicate fixtures found in {path.name}"
            )

        # Verify that full-time result labels agree with the scores.
        expected = pd.Series(
            [
                "H" if home > away
                else "A" if home < away
                else "D"
                for home, away in zip(df["FTHG"], df["FTAG"])
            ],
            index=df.index,
        )

        actual = df["FTR"].astype(str).str.strip()
        mismatch = expected != actual

        if mismatch.any():
            print(f"\nResult mismatches in {path.name}:")
            print(
                df.loc[
                    mismatch,
                    ["Date", "HomeTeam", "AwayTeam",
                     "FTHG", "FTAG", "FTR"],
                ].head(20).to_string(index=False)
            )
            raise ValueError(
                f"{int(mismatch.sum())} result mismatches in {path.name}"
            )

        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)

    matches = matches.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    duplicate_mask = matches.duplicated(
        ["Date", "HomeTeam", "AwayTeam"], keep=False
    )

    if duplicate_mask.any():
        print(matches.loc[duplicate_mask].to_string(index=False))
        raise ValueError("Duplicate fixtures found across seasons.")

    return matches


def summarize(history):
    """Summarize a team's last five completed league matches."""
    if not history:
        return {
            "points_avg": 1.0,
            "goals_for_avg": 1.0,
            "goals_against_avg": 1.0,
            "matches_prior": 0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
        }

    history = list(history)
    count = len(history)

    return {
        "points_avg": sum(x[0] for x in history) / count,
        "goals_for_avg": sum(x[1] for x in history) / count,
        "goals_against_avg": sum(x[2] for x in history) / count,
        "matches_prior": count,
        "win_rate": sum(x[0] == 3 for x in history) / count,
        "draw_rate": sum(x[0] == 1 for x in history) / count,
        "loss_rate": sum(x[0] == 0 for x in history) / count,
    }


def prepare_data():
    matches = load_matches()

    histories = defaultdict(lambda: deque(maxlen=WINDOW))
    rows = []

    # Compute all features for a date before adding any results
    # from that date. This avoids same-day result leakage.
    for match_date, day in matches.groupby("Date", sort=True):
        pending_updates = []

        for _, match in day.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            home_stats = summarize(histories[home])
            away_stats = summarize(histories[away])

            rows.append({
                "Date": match_date,
                "season": match["season"],
                "HomeTeam": home,
                "AwayTeam": away,

                "home_points_avg_5": home_stats["points_avg"],
                "home_goals_for_avg_5": home_stats["goals_for_avg"],
                "home_goals_against_avg_5": home_stats["goals_against_avg"],
                "home_matches_prior": home_stats["matches_prior"],
                "home_win_rate_5": home_stats["win_rate"],
                "home_draw_rate_5": home_stats["draw_rate"],
                "home_loss_rate_5": home_stats["loss_rate"],

                "away_points_avg_5": away_stats["points_avg"],
                "away_goals_for_avg_5": away_stats["goals_for_avg"],
                "away_goals_against_avg_5": away_stats["goals_against_avg"],
                "away_matches_prior": away_stats["matches_prior"],
                "away_win_rate_5": away_stats["win_rate"],
                "away_draw_rate_5": away_stats["draw_rate"],
                "away_loss_rate_5": away_stats["loss_rate"],

                # Target and scores are outcomes, not input features.
                "target": match["FTR"],
                "home_goals": int(match["FTHG"]),
                "away_goals": int(match["FTAG"]),
            })

            home_goals = int(match["FTHG"])
            away_goals = int(match["FTAG"])

            if home_goals > away_goals:
                home_points, away_points = 3, 0
            elif home_goals < away_goals:
                home_points, away_points = 0, 3
            else:
                home_points, away_points = 1, 1

            pending_updates.extend([
                (home, (home_points, home_goals, away_goals)),
                (away, (away_points, away_goals, home_goals)),
            ])

        # Update histories only after every fixture on this date
        # has received its pre-match features.
        for team, result in pending_updates:
            histories[team].append(result)

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["Date", "HomeTeam", "AwayTeam"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("MULTI-SEASON EPL FEATURE PREPARATION")
    print("=" * 60)
    print(f"Matches loaded: {len(matches)}")
    print(f"Feature rows: {len(output)}")
    print(f"Seasons: {output['season'].nunique()}")
    print(
        f"Date range: {output['Date'].min().date()} "
        f"to {output['Date'].max().date()}"
    )
    print(
        "Unique teams:",
        len(set(output["HomeTeam"]) | set(output["AwayTeam"]))
    )
    print(f"Missing values: {int(output.isna().sum().sum())}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nMatches per season:")
    print(output.groupby("season").size().to_string())

    print("\nResult distribution:")
    print(output["target"].value_counts().to_string())

    print("\nPreview:")
    print(output.head(5).to_string(index=False))


if __name__ == "__main__":
    prepare_data()
