import pandas as pd
import requests
from io import StringIO


URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_English_football_transfers_summer_2026"
)


PREMIER_LEAGUE_TEAMS = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
}


def load_transfer_table():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(
        URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    return tables[0]


def get_premier_league_transfers():
    df = load_transfer_table().copy()

    mask = (
        df["Moving from"].isin(PREMIER_LEAGUE_TEAMS)
        | df["Moving to"].isin(PREMIER_LEAGUE_TEAMS)
    )

    transfers = df[mask].copy()

    transfers = transfers[
        [
            "Date",
            "Player",
            "Moving from",
            "Moving to",
            "Fee",
        ]
    ]

    return transfers


if __name__ == "__main__":
    transfers = get_premier_league_transfers()

    print("Premier League-related transfers:", len(transfers))
    print()

    print(
        transfers.to_string(
            index=False
        )
    )
    from pathlib import Path

    OUTPUT_DIR = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "transfer_data"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "premier_league_transfers_2026.csv"
    )

    transfers.to_csv(
        output_file,
        index=False,
    )

    print()
    print("Saved to:", output_file)