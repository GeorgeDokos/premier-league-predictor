import numpy as np
import pandas as pd
from broader_player_loader import load_broader_player_data

from transfer_player_matcher import (
    match_transfers_to_players,
    parse_fee,
)

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

def build_reference_population():
    players = load_broader_player_data().copy()

    players["position_group"] = (
        players["position"]
        .apply(normalize_position)
    )

    numeric_columns = [
        "minutes_played",
        "goals",
        "assists",
        "expected_goals",
        "expected_assists",
        "rating",
        "market_value",
    ]

    for column in numeric_columns:
        players[column] = pd.to_numeric(
            players[column],
            errors="coerce",
        )

    safe_minutes = players[
        "minutes_played"
    ].replace(0, np.nan)

    players["goals_per90"] = (
        players["goals"]
        / safe_minutes
        * 90
    )

    players["assists_per90"] = (
        players["assists"]
        / safe_minutes
        * 90
    )

    players["xg_per90"] = (
        players["expected_goals"]
        / safe_minutes
        * 90
    )

    players["xa_per90"] = (
        players["expected_assists"]
        / safe_minutes
        * 90
    )
    
    players["tackles_per90"] = (
        players["tackles"]
        / safe_minutes
        * 90
    )

    players["interceptions_per90"] = (
        players["interceptions"]
        / safe_minutes
        * 90
    )

    players["saves_per90"] = (
        players["saves"]
        / safe_minutes
        * 90
    )

    return players

def percentile(series):
    return series.rank(
        pct=True,
        method="average",
    )

def calculate_quality_scores(df):
    df = df.copy()

    reference = build_reference_population()

    numeric_columns = [
        "stats_goals",
        "stats_assists",
        "stats_xg",
        "stats_xa",
        "stats_rating",
        "stats_market_value",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    safe_minutes = df[
        "stats_minutes"
    ].replace(0, np.nan)

    df["goals_per90"] = (
        df["stats_goals"]
        / safe_minutes
        * 90
    )

    df["assists_per90"] = (
        df["stats_assists"]
        / safe_minutes
        * 90
    )

    df["xg_per90"] = (
        df["stats_xg"]
        / safe_minutes
        * 90
    )

    df["xa_per90"] = (
        df["stats_xa"]
        / safe_minutes
        * 90
    )
    
    df["tackles_per90"] = (
        df["stats_tackles"]
        / safe_minutes
        * 90
    )

    df["interceptions_per90"] = (
        df["stats_interceptions"]
        / safe_minutes
        * 90
    )

    df["saves_per90"] = (
        df["stats_saves"]
        / safe_minutes
        * 90
    )

    df["quality_score"] = np.nan

    for position in ["FW", "MF", "DF", "GK"]:
        transfer_mask = (
            (df["position_group"] == position)
            & df["matched_name_key"].notna()
        )

        transfer_group = df.loc[
            transfer_mask
        ].copy()

        reference_group = reference[
            reference["position_group"] == position
        ].copy()

        if (
            transfer_group.empty
            or reference_group.empty
        ):
            continue

        def reference_percentile(
            values,
            reference_values,
        ):
            reference_values = (
                reference_values
                .dropna()
                .sort_values()
                .to_numpy()
            )

            if len(reference_values) == 0:
                return pd.Series(
                    0.5,
                    index=values.index,
                )

            result = []

            for value in values:
                if pd.isna(value):
                    result.append(np.nan)
                else:
                    percentile_value = (
                        np.searchsorted(
                            reference_values,
                            value,
                            side="right",
                        )
                        / len(reference_values)
                    )

                    result.append(
                        percentile_value
                    )

            return pd.Series(
                result,
                index=values.index,
            )

        rating_pct = reference_percentile(
            transfer_group["stats_rating"],
            reference_group["rating"],
        )

        market_pct = reference_percentile(
            transfer_group["stats_market_value"],
            reference_group["market_value"],
        )

        goals_pct = reference_percentile(
            transfer_group["goals_per90"],
            reference_group["goals_per90"],
        )

        assists_pct = reference_percentile(
            transfer_group["assists_per90"],
            reference_group["assists_per90"],
        )

        xg_pct = reference_percentile(
            transfer_group["xg_per90"],
            reference_group["xg_per90"],
        )

        xa_pct = reference_percentile(
            transfer_group["xa_per90"],
            reference_group["xa_per90"],
        )

        # Missing advanced stats:
        # fall back to related observed statistic.
        xg_pct = xg_pct.fillna(
            goals_pct
        )

        xa_pct = xa_pct.fillna(
            assists_pct
        )
        
        tackles_pct = reference_percentile(
            transfer_group["tackles_per90"],
            reference_group["tackles_per90"],
        ).fillna(0.5)

        interceptions_pct = reference_percentile(
            transfer_group["interceptions_per90"],
            reference_group["interceptions_per90"],
        ).fillna(0.5)

        saves_pct = reference_percentile(
            transfer_group["saves_per90"],
            reference_group["saves_per90"],
        ).fillna(0.5)

        # Remaining missing values get
        # neutral percentile rather than 0.
        rating_pct = rating_pct.fillna(0.5)
        market_pct = market_pct.fillna(0.5)
        goals_pct = goals_pct.fillna(0.5)
        assists_pct = assists_pct.fillna(0.5)
        xg_pct = xg_pct.fillna(0.5)
        xa_pct = xa_pct.fillna(0.5)

        if position == "FW":
            score = (
                0.30 * rating_pct
                + 0.20 * goals_pct
                + 0.20 * xg_pct
                + 0.10 * assists_pct
                + 0.10 * xa_pct
                + 0.10 * market_pct
            )

        elif position == "MF":
            score = (
                0.35 * rating_pct
                + 0.15 * goals_pct
                + 0.15 * assists_pct
                + 0.10 * xg_pct
                + 0.10 * xa_pct
                + 0.15 * market_pct
            )

        elif position == "DF":
            score = (
                0.35 * rating_pct
                + 0.20 * tackles_pct
                + 0.20 * interceptions_pct
                + 0.10 * goals_pct
                + 0.05 * assists_pct
                + 0.10 * market_pct
            )

        else:  # GK
            score = (
                0.55 * rating_pct
                + 0.30 * saves_pct
                + 0.15 * market_pct
            )

        df.loc[
            transfer_group.index,
            "quality_score",
        ] = score

    return df

def calculate_role_importance(minutes):
    minutes = safe_numeric(minutes)

    role = np.sqrt(
        minutes.clip(lower=0) / 2700
    )

    return role.clip(0, 1)
def calculate_starter_multiplier(minutes):
    minutes = safe_numeric(minutes)

    if pd.isna(minutes):
        return 1.0

    if minutes >= 2700:
        return 1.10

    if minutes >= 1800:
        return 1.05

    return 1.0
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
        "stats_tackles",
        "stats_interceptions",
        "stats_saves",
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
    missing_position_mask = (
        df["position_group"].isna()
        | (df["position_group"] == "UNKNOWN")
    )

    df.loc[
        missing_position_mask,
        "position_group",
    ] = df.loc[
        missing_position_mask,
        "Player",
    ].map(
        MANUAL_POSITION_MAP
    )

    df = calculate_quality_scores(df)
 

    df["role_importance"] = (
        calculate_role_importance(
            df["stats_minutes"]
        )
    )

    high_quality_mask = (
        (df["quality_score"] >= 0.70)
        & (df["stats_minutes"] >= 900)
    )

    df.loc[
        high_quality_mask,
        "role_importance",
    ] = df.loc[
        high_quality_mask,
        "role_importance",
    ].clip(lower=0.80)

    df["starter_multiplier"] = (
        df["stats_minutes"].apply(
            calculate_starter_multiplier
        )
    )

    df["player_impact"] = (
        df["quality_score"]
        * df["role_importance"]
        * df["starter_multiplier"]
    ).clip(upper=1.0)
    fee_model = fit_fee_fallback_model(df)

    df["fallback_impact"] = np.nan

    unmatched_with_fee = (
        df["matched_name_key"].isna()
        & (df["fee_millions"] > 0)
    )

    df.loc[
        unmatched_with_fee,
        "fallback_impact",
    ] = df.loc[
        unmatched_with_fee,
        "fee_millions",
    ].apply(
        lambda fee: estimate_impact_from_fee(
            fee,
            fee_model,
        )
    )

    df["effective_player_impact"] = (
        df["player_impact"]
        .fillna(
            df["fallback_impact"]
        )
    )
    manual_mask = (
        df["effective_player_impact"].isna()
        & df["Player"].isin(
            MANUAL_FREE_UNDISCLOSED_FALLBACK
        )
    )

    df.loc[
        manual_mask,
        "effective_player_impact",
    ] = df.loc[
        manual_mask,
        "Player",
    ].map(
        MANUAL_FREE_UNDISCLOSED_FALLBACK
    )
    df["effective_player_impact"] = (
        df["effective_player_impact"]
        .fillna(0.0)
    )

    return df

TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Coventry City": "Coventry",
    "Ipswich Town": "Ipswich",
    "Hull City": "Hull",
}

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

def normalize_team_name(team):
    return TEAM_NAME_MAP.get(
        team,
        team,
    )

def calculate_team_transfer_impacts(df):
    records = []

    matched = df[
        df["effective_player_impact"].notna()
    ].copy()

    for _, row in matched.iterrows():

        moving_from = row["Moving from"]
        moving_to = row["Moving to"]
        impact = row["effective_player_impact"]
        position = row["position_group"]

        attack_impact, defence_impact = (
            split_player_impact(
                position,
                impact,
            )
        )

        # Player leaves a Premier League team
        if moving_from in PREMIER_LEAGUE_TEAMS:
            records.append(
                {
                    "team": normalize_team_name(
                        moving_from
                    ),
                    "Player": row["Player"],
                    "direction": "OUT",
                    "position_group": position,
                    "impact": -impact,
                    "attack_impact": -attack_impact,
                    "defence_impact": -defence_impact,
                }
            )

        # Player joins a Premier League team
        if moving_to in PREMIER_LEAGUE_TEAMS:
            records.append(
                {
                    "team": normalize_team_name(
                        moving_to
                    ),
                    "Player": row["Player"],
                    "direction": "IN",
                    "position_group": position,
                    "impact": impact,
                    "attack_impact": attack_impact,
                    "defence_impact": defence_impact,
                }
            )

    movements = pd.DataFrame(records)

    movements["incoming_impact"] = (
        movements["impact"].where(
            movements["direction"] == "IN",
            0.0,
        )
    )

    movements["outgoing_impact"] = (
        -movements["impact"].where(
            movements["direction"] == "OUT",
            0.0,
        )
    )

    movements["incoming_attack"] = (
        movements["attack_impact"].where(
            movements["direction"] == "IN",
            0.0,
        )
    )

    movements["outgoing_attack"] = (
        -movements["attack_impact"].where(
            movements["direction"] == "OUT",
            0.0,
        )
    )

    movements["incoming_defence"] = (
        movements["defence_impact"].where(
            movements["direction"] == "IN",
            0.0,
        )
    )

    movements["outgoing_defence"] = (
        -movements["defence_impact"].where(
            movements["direction"] == "OUT",
            0.0,
        )
    )

    # -----------------------------------------
    # POSITION-AWARE NET TRANSFER IMPACT
    # -----------------------------------------

    position_summary = (
        movements
        .groupby(
            ["team", "position_group"]
        )
        .agg(
            incoming=(
                "incoming_impact",
                "sum",
            ),
            outgoing=(
                "outgoing_impact",
                "sum",
            ),
        )
        .reset_index()
    )

    position_summary["net_position_impact"] = (
        position_summary["incoming"]
        - position_summary["outgoing"]
    )

    # An unfilled positional loss hurts more
    position_summary[
        "adjusted_position_impact"
    ] = position_summary[
        "net_position_impact"
    ]

    negative_mask = (
        position_summary[
            "adjusted_position_impact"
        ] < 0
    )

    position_summary.loc[
        negative_mask,
        "adjusted_position_impact",
    ] *= POSITION_DEFICIT_MULTIPLIER

    # Convert each positional net impact
    # into attacking / defensive contribution
    position_summary[
        "adjusted_attack"
    ] = 0.0

    position_summary[
        "adjusted_defence"
    ] = 0.0

    for idx, row in (
        position_summary.iterrows()
    ):
        attack, defence = (
            split_player_impact(
                row["position_group"],
                row[
                    "adjusted_position_impact"
                ],
            )
        )

        position_summary.loc[
            idx,
            "adjusted_attack",
        ] = attack

        position_summary.loc[
            idx,
            "adjusted_defence",
        ] = defence

    position_team_summary = (
        position_summary
        .groupby("team")
        .agg(
            net_attack_impact=(
                "adjusted_attack",
                "sum",
            ),
            net_defence_impact=(
                "adjusted_defence",
                "sum",
            ),
        )
        .reset_index()
    )

    # -----------------------------------------
    # NORMAL TEAM SUMMARY
    # -----------------------------------------

    summary = (
        movements
        .groupby("team")
        .agg(
            incoming_impact=(
                "incoming_impact",
                "sum",
            ),
            outgoing_impact=(
                "outgoing_impact",
                "sum",
            ),
            net_transfer_impact=(
                "impact",
                "sum",
            ),
            incoming_attack=(
                "incoming_attack",
                "sum",
            ),
            outgoing_attack=(
                "outgoing_attack",
                "sum",
            ),
            incoming_defence=(
                "incoming_defence",
                "sum",
            ),
            outgoing_defence=(
                "outgoing_defence",
                "sum",
            ),
            total_movements=(
                "Player",
                "count",
            ),
        )
        .reset_index()
    )

    # Replace the old simple net with
    # the position-aware version
    summary = summary.merge(
        position_team_summary,
        on="team",
        how="left",
    )

    summary["attack_adjustment"] = (
        summary["net_attack_impact"]
        .apply(scale_transfer_impact)
    )

    summary["defence_adjustment"] = (
        summary["net_defence_impact"]
        .apply(scale_transfer_impact)
    )

    summary = summary.sort_values(
        "net_transfer_impact",
        ascending=False,
    )
    debug_positions = position_summary[
    position_summary["team"].isin(
        ["Newcastle", "Tottenham"]
    )
]

    print()
    print("POSITION BREAKDOWN")
    print(
        debug_positions[
            [
                "team",
                "position_group",
                "incoming",
                "outgoing",
                "net_position_impact",
                "adjusted_position_impact",
                "adjusted_attack",
                "adjusted_defence",
            ]
        ].to_string(index=False)
    )
    audit_teams = [
        "Arsenal",
        "Man City",
        "Tottenham",
        "Newcastle",
    ]

    audit = movements[
        movements["team"].isin(audit_teams)
    ].copy()

    audit["abs_impact"] = audit["impact"].abs()

    audit = audit.sort_values(
        ["team", "abs_impact"],
        ascending=[True, False],
    )

    print()
    print("TOP TRANSFER IMPACT AUDIT")

    for team in audit_teams:
        print()
        print("=" * 60)
        print(team)
        print("=" * 60)

        team_rows = audit[
            audit["team"] == team
        ].head(15)

        print(
            team_rows[
                [
                    "Player",
                    "direction",
                    "position_group",
                    "impact",
                    "attack_impact",
                    "defence_impact",
                ]
            ].to_string(index=False)
        )

    return movements, summary

def fit_fee_fallback_model(df):
    training = df[
        df["player_impact"].notna()
        & (df["fee_millions"] > 0)
    ].copy()

    if len(training) < 5:
        return None

    x = np.log1p(
        training["fee_millions"].to_numpy()
    )

    y = training[
        "player_impact"
    ].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "n": len(training),
    }

def estimate_impact_from_fee(
    fee_millions,
    model,
):
    if (
        model is None
        or pd.isna(fee_millions)
        or fee_millions <= 0
    ):
        return np.nan

    estimate = (
        model["intercept"]
        + model["slope"]
        * np.log1p(fee_millions)
    )

    return float(
        np.clip(
            estimate,
            0.05,
            0.85,
        )
    )
    
MANUAL_FREE_UNDISCLOSED_FALLBACK = {
    "Chuba Akpom": 0.40,
    "Jack Butland": 0.42,
    "Illan Meslier": 0.48,
    "Matt Targett": 0.38,
    "Kjell Scherpen": 0.42,
    "Gustavo Hamer": 0.50,
    "Joe Gelhardt": 0.30,
}
MANUAL_POSITION_MAP = {
    "Chuba Akpom": "FW",
    "Modou Kéba Cissé": "DF",
    "Costinha": "DF",
    "Ewen Jaouen": "GK",
    "Jannik Schuster": "DF",
    "Zadok Yohanna": "FW",
    "Jimmy-Jay Morgan": "FW",
    "Ivor Pandur": "GK",
    "Aidon Shehu": "DF",
    "Nazariy Rusyn": "FW",
    "Jack Butland": "GK",
    "Hayden Hackney": "MF",
    "Michael Svoboda": "DF",
    "Illan Meslier": "GK",
    "Jeremy Monga": "FW",
    "Lewis Dobbin": "FW",
    "Reigan Heskey": "FW",
    "Tynan Thompson": "FW",
    "Abdul Fatawu": "FW",
    "Will Lankshear": "FW",
    "Matt Targett": "DF",
    "Christos Tzolis": "FW",
    "Daizen Maeda": "FW",
    "Kjell Scherpen": "GK",
    "Carl Rushworth": "GK",
    "Konstantinos Tzolakis": "GK",
    "Caleb Yirenkyi": "MF",
    "Shea Charles": "MF",
    "Gustavo Hamer": "MF",
    "Dastan Satpayev": "FW",
    "Joe Gelhardt": "FW",
    "Anan Khalaili": "FW",
    "Zavier Gozo": "FW",
}

def split_player_impact(position, impact):
    if pd.isna(impact):
        return 0.0, 0.0

    if position == "FW":
        attack = impact * 0.85
        defence = impact * 0.15

    elif position == "MF":
        attack = impact * 0.60
        defence = impact * 0.40

    elif position == "DF":
        attack = impact * 0.20
        defence = impact * 0.80

    elif position == "GK":
        attack = 0.0
        defence = impact

    else:
        attack = impact * 0.50
        defence = impact * 0.50

    return attack, defence
MAX_TRANSFER_ADJUSTMENT = 0.18
IMPACT_SCALE = 2.0
POSITION_DEFICIT_MULTIPLIER = 1.25
def scale_transfer_impact(raw_impact):
    if pd.isna(raw_impact):
        return 0.0

    return (
        MAX_TRANSFER_ADJUSTMENT
        * np.tanh(
            raw_impact / IMPACT_SCALE
        )
    )
def get_team_transfer_adjustments():
    df = build_transfer_impact_table()

    _, summary = (
        calculate_team_transfer_impacts(df)
    )

    adjustments = {}

    for _, row in summary.iterrows():
        adjustments[row["team"]] = {
            "attack_adjustment": row[
                "attack_adjustment"
            ],
            "defence_adjustment": row[
                "defence_adjustment"
            ],
        }

    return adjustments
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
        "position_group",
        "stats_minutes",
        "quality_score",
        "role_importance",
        "player_impact",
        "fee_millions",
    ]

    print(
        matched[columns]
        .sort_values(
            "player_impact",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )
    
    reference = build_reference_population()

    print()
    print("REFERENCE POPULATION")
    print("Total:", len(reference))

    print()
    print(
        reference["position_group"]
        .value_counts()
        .to_string()
    )
    movements, summary = (
        calculate_team_transfer_impacts(df)
    )
    print()
    print("TEAM TRANSFER IMPACTS")
    print()

    display_columns = [
        "team",
        "net_attack_impact",
        "attack_adjustment",
        "net_defence_impact",
        "defence_adjustment",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )
    
    fee_model = fit_fee_fallback_model(df)

    print()
    print("FEE FALLBACK MODEL")
    print(fee_model)

    print()
    print("UNMATCHED WITH FEE FALLBACK")
    print()

    fallback_rows = df[
        df["matched_name_key"].isna()
        & df["fallback_impact"].notna()
    ].copy()

    print(
        fallback_rows[
            [
                "Player",
                "Moving from",
                "Moving to",
                "fee_millions",
                "fallback_impact",
            ]
        ]
        .sort_values(
            "fallback_impact",
            ascending=False,
        )
        .to_string(index=False)
    )
    
    print()
    print("UNMATCHED FREE / UNDISCLOSED INCOMING")
    print()

    free_unknown_incoming = df[
        df["matched_name_key"].isna()
        & (df["fee_millions"] == 0)
        & df["Moving to"].isin(PREMIER_LEAGUE_TEAMS)
    ].copy()

    print(
        free_unknown_incoming[
            [
                "Player",
                "Moving from",
                "Moving to",
                "Fee",
            ]
        ].to_string(index=False)
    )
    print()
    print("IMPACT PLAYERS WITH UNKNOWN POSITION")
    print()

    unknown_position = df[
        (df["effective_player_impact"] > 0)
        & (
            df["position_group"].isna()
            | (df["position_group"] == "UNKNOWN")
        )
    ].copy()

    print("Count:", len(unknown_position))

    print(
        unknown_position[
            [
                "Player",
                "Moving from",
                "Moving to",
                "fee_millions",
                "effective_player_impact",
            ]
        ].to_string(index=False)
    )
    print()
    print("TEAM ADJUSTMENT DICTIONARY")
    print()

    adjustments = (
        get_team_transfer_adjustments()
    )

    for team, values in adjustments.items():
        print(
            team,
            values,
        )
        
