import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "player_data"

PLAYER_FILE = DATA_DIR / "players_data_2025_26.csv"


def load_player_data():
    df = pd.read_csv(PLAYER_FILE)
    return df


if __name__ == "__main__":
    df = load_player_data()

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    print("COLUMN NAMES")
    print(df.columns.tolist())
    print()

    print("FIRST 5 ROWS")
    print(df.head())
    print()

    print("MISSING VALUES - TOP 30")
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(30)
        * 100
    )

    print(missing.round(1))