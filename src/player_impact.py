import numpy as np
import pandas as pd

from player_data_loader import load_player_data


def position_group(pos):
    if pd.isna(pos):
        return "OTHER"

    if "GK" in pos:
        return "GK"

    if "FW" in pos:
        return "FW"

    if "MF" in pos:
        return "MF"

    if "DF" in pos:
        return "DF"

    return "OTHER"


def safe_per90(value, nineties):
    if nineties <= 0:
        return 0

    return value / nineties


def prepare_player_data():
    df = load_player_data().copy()

    df["PositionGroup"] = df["Pos"].apply(
        position_group
    )

    numeric_columns = [
        "Min",
        "90s",
        "Gls",
        "Ast",
        "Sh",
        "SoT",
        "TklW",
        "Int",
        "Crs",
        "PPM",
        "+/-90",
        "Save%",
        "GA90",
        "CS%",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["Gls90"] = (
        df["Gls"] / df["90s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Ast90"] = (
        df["Ast"] / df["90s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["TklW90"] = (
        df["TklW"] / df["90s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Int90"] = (
        df["Int"] / df["90s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["Crs90"] = (
        df["Crs"] / df["90s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def percentile(series):
    return series.rank(
        pct=True,
        method="average",
    )

def calculate_quality_scores(df):
    df = df.copy()

    df["QualityScore"] = np.nan

    for (competition, position), group in df.groupby(
        ["Comp", "PositionGroup"]
    ):
        idx = group.index

        if position == "FW":
            score = (
                0.45 * percentile(group["Gls90"])
                + 0.25 * percentile(group["Ast90"])
                + 0.15 * percentile(group["Sh/90"])
                + 0.10 * percentile(group["SoT/90"])
                + 0.05 * percentile(group["PPM"])
            )

        elif position == "MF":
            score = (
                0.20 * percentile(group["Gls90"])
                + 0.25 * percentile(group["Ast90"])
                + 0.15 * percentile(group["TklW90"])
                + 0.15 * percentile(group["Int90"])
                + 0.15 * percentile(group["Crs90"])
                + 0.10 * percentile(group["PPM"])
            )

        elif position == "DF":
            score = (
                0.30 * percentile(group["TklW90"])
                + 0.30 * percentile(group["Int90"])
                + 0.10 * percentile(group["Crs90"])
                + 0.10 * percentile(group["Gls90"])
                + 0.10 * percentile(group["PPM"])
                + 0.10 * percentile(group["+/-90"])
            )

        elif position == "GK":
            score = (
                0.45 * percentile(group["Save%"])
                + 0.30 * (1 - percentile(group["GA90"]))
                + 0.15 * percentile(group["CS%"])
                + 0.10 * percentile(group["PPM"])
            )

        else:
            continue

        df.loc[idx, "QualityScore"] = score

    return df

def calculate_player_impact(df):
    df = calculate_quality_scores(df)

    max_minutes = (
        df.groupby("Comp")["Min"]
        .transform("max")
    )

    df["RoleImportance"] = (
        df["Min"] / max_minutes
    ).clip(0, 1)

    df["PlayerImpact"] = (
        df["QualityScore"]
        * np.sqrt(df["RoleImportance"])
    )

    return df

if __name__ == "__main__":
    df = prepare_player_data()
    df = calculate_player_impact(df)

    premier_league = df[
        df["Comp"].str.contains(
            "Premier League",
            case=False,
            na=False,
        )
    ]

    premier_league = premier_league[
        premier_league["Min"] >= 900
    ]

    top = premier_league.sort_values(
        "PlayerImpact",
        ascending=False,
    )

    print(
        top[
            [
                "Player",
                "Squad",
                "Pos",
                "Min",
                "QualityScore",
                "RoleImportance",
                "PlayerImpact",
            ]
        ].head(30).to_string(index=False)
    )