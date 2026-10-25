import pandas as pd
from pathlib import Path

raw_dir = Path("data/football/raw")

season_codes = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2024_25"
]

columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

for code in season_codes:
    path = raw_dir / f"epl_{code}.csv"
    df = pd.read_csv(path)[columns].copy()

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["FTR"] = df["FTR"].astype("string").str.strip()

    expected = pd.Series(
        [
            "H" if h > a else "A" if h < a else "D"
            for h, a in zip(df["FTHG"], df["FTAG"])
        ],
        index=df.index,
    )

    bad = df.loc[expected != df["FTR"]]

    if not bad.empty:
        print(f"\n{path.name}: {len(bad)} mismatches")
        print(bad.head(15).to_string(index=False))

print("\nInspection complete.")