import numpy as np
import pandas as pd

from transfer_player_matcher import (
    match_transfers_to_players,
    parse_fee,
)

print("transfer_impact.py started")
def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )

def normalize_position(position):
    if pd.isna(position):
        return "UNKNOWN"

    position = str(position).upper().strip()

    if position in ["F", "FW"]:
        return "FW"

    if position in ["M", "MF"]:
        return "MF"

    if position in ["D", "DF"]:
        return "DF"

    if position in ["G", "GK"]:
        return "GK"

    return "UNKNOWN"

def calculate_role_importance(minutes):
    minutes = safe_numeric(minutes)

    role = np.sqrt(
        minutes.clip(lower=0) / 2700
    )

    return role.clip(
        lower=0,
        upper=1,
    )


def build_transfer_impact_table():
    matches, _, _ = (
        match_transfers_to_players()
    )

    df = matches.copy()

    df["fee_millions"] = (
        df["Fee"]
        .apply(parse_fee)
    )

    numeric_columns = [
        "stats_minutes",
        "stats_goals",
        "stats_assists",
        "stats_xg",
        "stats_xa",
        "stats_rating",
        "stats_market_value",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = safe_numeric(
                df[column]
            )

    df["position_group"] = (
        df["stats_position"]
        .apply(normalize_position)
    )

    df["role_importance"] = (
        calculate_role_importance(
            df["stats_minutes"]
        )
    )

    return df


if __name__ == "__main__":
    print("main block started")

    df = build_transfer_impact_table()

    print("Rows:", len(df))

    matched = df[
        df["matched_name_key"].notna()
    ].copy()

    print("Matched rows:", len(matched))

    columns = [
        "Player",
        "Moving from",
        "Moving to",
        "stats_position",
        "position_group",
        "stats_minutes",
        "role_importance",
        "stats_rating",
        "stats_market_value",
        "fee_millions",
    ]

    print(
        matched[columns]
        .sort_values(
            "role_importance",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )