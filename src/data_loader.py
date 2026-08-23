import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEASONS = {
    "2023_24": DATA_DIR / "premier_league_2023_24.csv",
    "2024_25": DATA_DIR / "premier_league_2024_25.csv",
    "2025_26": DATA_DIR / "premier_league_2025_26.csv",
}

PREDICTION_DATE = pd.Timestamp("2026-08-20")

HALF_LIFE_DAYS = 365


def calculate_recency_weight(match_date):
    age_days = (PREDICTION_DATE - match_date).days

    return np.exp(
        -np.log(2) * age_days / HALF_LIFE_DAYS
    )


def load_premier_league_data():
    all_matches = []

    for season, file_path in SEASONS.items():
        df = pd.read_csv(file_path)

        df = df[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
            ]
        ].copy()

        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d/%m/%Y",
        )

        df["Season"] = season

        df["Weight"] = df["Date"].apply(
            calculate_recency_weight
        )

        all_matches.append(df)

    return pd.concat(all_matches, ignore_index=True)

def load_championship_data():
    file_path = DATA_DIR / "championship_2025_26.csv"

    df = pd.read_csv(file_path)

    df = df[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
    )

    return df

if __name__ == "__main__":
    matches = load_premier_league_data()

    print(
        matches[
            ["Date", "HomeTeam", "AwayTeam", "Season", "Weight"]
        ].head()
    )

    print()

    print(
        matches.groupby("Season")["Weight"].agg(
            ["min", "max", "mean"]
        )
    )