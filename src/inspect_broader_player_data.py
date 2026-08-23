import pandas as pd
from pathlib import Path


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "player_data"
)

STATS_FILE = DATA_DIR / "all_player_stats.csv"
PROFILES_FILE = DATA_DIR / "all_player_profiles.csv"


def inspect_file(file_path, name):
    df = pd.read_csv(file_path)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()

    print("COLUMN NAMES")
    print(df.columns.tolist())

    print()
    print("FIRST 5 ROWS")
    print(df.head().to_string())

    print()


if __name__ == "__main__":
    print("Script started")

    inspect_file(
        STATS_FILE,
        "ALL PLAYER STATS",
    )

    inspect_file(
        PROFILES_FILE,
        "ALL PLAYER PROFILES",
    )