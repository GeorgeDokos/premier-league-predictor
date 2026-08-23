import pandas as pd
from pathlib import Path


DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "player_data"
)

STATS_FILE = DATA_DIR / "all_player_stats.csv"
PROFILES_FILE = DATA_DIR / "all_player_profiles.csv"


def load_broader_player_data():
    stats = pd.read_csv(STATS_FILE)
    profiles = pd.read_csv(PROFILES_FILE)

    players = profiles.merge(
        stats,
        on=["player_id", "league"],
        how="inner",
    )

    return players


if __name__ == "__main__":
    df = load_broader_player_data()

    print("Rows after merge:", len(df))
    print()

    print("LEAGUES")
    print(
        df["league"]
        .value_counts()
        .to_string()
    )

    print()
    print("DUPLICATE PLAYER NAMES")

    duplicate_names = (
        df[df["name"].duplicated(keep=False)]
        .sort_values("name")
    )

    print(
        duplicate_names[
            [
                "name",
                "league",
                "position",
                "minutes_played",
            ]
        ].head(30).to_string(index=False)
    )