import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


SEASON_PAIRS = [
    (
        DATA_DIR / "championship_2020_21.csv",
        DATA_DIR / "premier_league_2021_22.csv",
    ),
    (
        DATA_DIR / "championship_2021_22.csv",
        DATA_DIR / "premier_league_2022_23.csv",
    ),
    (
        DATA_DIR / "championship_2022_23.csv",
        DATA_DIR / "premier_league_2023_24.csv",
    ),
    (
        DATA_DIR / "championship_2023_24.csv",
        DATA_DIR / "premier_league_2024_25.csv",
    ),
    (
        DATA_DIR / "championship_2024_25.csv",
        DATA_DIR / "premier_league_2025_26.csv",
    ),
]

def load_season(file_path):
    df = pd.read_csv(file_path)

    return df[
        [
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    ].copy()

def calculate_ratings(matches):
    home_goal_avg = matches["FTHG"].mean()
    away_goal_avg = matches["FTAG"].mean()

    teams = sorted(
        set(matches["HomeTeam"]).union(
            set(matches["AwayTeam"])
        )
    )

    ratings = {}

    for team in teams:
        home_matches = matches[
            matches["HomeTeam"] == team
        ]

        away_matches = matches[
            matches["AwayTeam"] == team
        ]

        home_scored = home_matches["FTHG"].mean()
        home_conceded = home_matches["FTAG"].mean()

        away_scored = away_matches["FTAG"].mean()
        away_conceded = away_matches["FTHG"].mean()

        ratings[team] = {
            "home_attack": (
                home_scored / home_goal_avg
            ),
            "home_defence": (
                home_conceded / away_goal_avg
            ),
            "away_attack": (
                away_scored / away_goal_avg
            ),
            "away_defence": (
                away_conceded / home_goal_avg
            ),
        }

    return ratings

def find_promoted_teams(
    championship_matches,
    premier_matches,
):
    championship_teams = (
        set(championship_matches["HomeTeam"])
        | set(championship_matches["AwayTeam"])
    )

    premier_teams = (
        set(premier_matches["HomeTeam"])
        | set(premier_matches["AwayTeam"])
    )

    return championship_teams & premier_teams

def calculate_conversion_factors():
    records = []

    for championship_file, premier_file in SEASON_PAIRS:
        championship_matches = load_season(
            championship_file
        )

        premier_matches = load_season(
            premier_file
        )

        champ_ratings = calculate_ratings(
            championship_matches
        )

        prem_ratings = calculate_ratings(
            premier_matches
        )

        promoted_teams = find_promoted_teams(
            championship_matches,
            premier_matches,
        )

        for team in promoted_teams:
            champ = champ_ratings[team]
            prem = prem_ratings[team]

            records.append(
                {
                    "team": team,

                    "ch_home_attack": champ["home_attack"],
                    "pl_home_attack": prem["home_attack"],

                    "ch_away_attack": champ["away_attack"],
                    "pl_away_attack": prem["away_attack"],

                    "ch_home_defence": champ["home_defence"],
                    "pl_home_defence": prem["home_defence"],

                    "ch_away_defence": champ["away_defence"],
                    "pl_away_defence": prem["away_defence"],
                }
            )

    return pd.DataFrame(records)

def fit_regressions(df):
    mappings = [
        ("home_attack", "ch_home_attack", "pl_home_attack"),
        ("away_attack", "ch_away_attack", "pl_away_attack"),
        ("home_defence", "ch_home_defence", "pl_home_defence"),
        ("away_defence", "ch_away_defence", "pl_away_defence"),
    ]

    models = {}

    for name, x_col, y_col in mappings:
        x = df[[x_col]]
        y = df[y_col]

        model = LinearRegression()
        model.fit(x, y)

        models[name] = {
            "intercept": model.intercept_,
            "slope": model.coef_[0],
            "r2": model.score(x, y),
        }

    return models

if __name__ == "__main__":
    factors = calculate_conversion_factors()

    print()
    print("PROMOTED TEAM DATA")
    print()

    print(factors.to_string(index=False))

    models = fit_regressions(factors)

    print()
    print("REGRESSION MODELS")
    print()

    for name, values in models.items():
        print(name)
        print(
            f"PL = {values['intercept']:.3f} "
            f"+ {values['slope']:.3f} * CH"
        )
        print(f"R² = {values['r2']:.3f}")
        print()