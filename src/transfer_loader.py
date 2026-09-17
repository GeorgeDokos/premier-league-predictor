import pandas as pd
import requests
from io import StringIO
from pathlib import Path

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
    MANUAL_TRANSFERS = [
        {
            "Date": "24 August 2026",
            "Player": "Savinho",
            "Moving from": "Manchester City",
            "Moving to": "Tottenham Hotspur",
            "Fee": "£85m",
        },
        {
            "Date": "24 August 2026",
            "Player": "Omar Marmoush",
            "Moving from": "Manchester City",
            "Moving to": "Tottenham Hotspur",
            "Fee": "Loan",
        },
    ]

    manual_df = pd.DataFrame(MANUAL_TRANSFERS)

    transfers = pd.concat(
        [transfers, manual_df],
        ignore_index=True,
    )

    return transfers

CUTOFF_DATE = pd.Timestamp("2026-08-24")

TRANSFER_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "transfer_data"
    / "premier_league_transfers_2026_08_24.csv"
)


def freeze_transfers():
    transfers = get_premier_league_transfers().copy()

    transfers["Date_parsed"] = pd.to_datetime(
        transfers["Date"],
        errors="coerce",
        dayfirst=True,
    )

    transfers = transfers[
        transfers["Date_parsed"].isna()
        | (transfers["Date_parsed"] <= CUTOFF_DATE)
    ].copy()

    transfers = transfers.drop(
        columns=["Date_parsed"]
    )

    TRANSFER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transfers.to_csv(
        TRANSFER_FILE,
        index=False,
    )

    print(f"Frozen transfers: {len(transfers)}")
    print(f"Cutoff date: {CUTOFF_DATE.date()}")
    print(f"Saved to: {TRANSFER_FILE}")

    return transfers


def load_saved_transfers():
    if not TRANSFER_FILE.exists():
        raise FileNotFoundError(
            f"Frozen transfer file not found: {TRANSFER_FILE}"
        )

    return pd.read_csv(TRANSFER_FILE)

if __name__ == "__main__":
    freeze_transfers()
