import random
from collections import defaultdict
import pandas as pd
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
from data_loader import (
    load_premier_league_data,
    load_championship_data,
)
from ratings import (
    calculate_team_ratings,
    calculate_league_averages,
    calculate_championship_ratings,
    convert_promoted_rating,
)
from poisson import expected_goals


PREMIER_LEAGUE_2026_27 = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull",
    "Ipswich",
    "Leeds",
    "Liverpool",
    "Man City",
    "Man United",
    "Newcastle",
    "Nott'm Forest",
    "Sunderland",
    "Tottenham",
}

PROMOTED_TEAMS = {
    "Coventry",
    "Ipswich",
    "Hull",
}


def build_2026_27_ratings():
    premier_matches = load_premier_league_data()

    historical_ratings = calculate_team_ratings(
        premier_matches
    )

    ratings = {}

    for team in PREMIER_LEAGUE_2026_27:
        if team in historical_ratings:
            ratings[team] = historical_ratings[team]

    championship_matches = load_championship_data()

    championship_ratings = calculate_championship_ratings(
        championship_matches
    )

    for team in PROMOTED_TEAMS:
        ratings[team] = convert_promoted_rating(
            championship_ratings[team]
        )

    return ratings, premier_matches

def simulate_score(home_xg, away_xg):
    import numpy as np

    home_goals = np.random.poisson(home_xg)
    away_goals = np.random.poisson(away_xg)

    return home_goals, away_goals

def create_table(teams):
    table = {}

    for team in teams:
        table[team] = {
            "points": 0,
            "gf": 0,
            "ga": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }

    return table

def simulate_season(ratings, home_goal_avg, away_goal_avg):
    teams = list(ratings.keys())

    table = create_table(teams)

    for home_team in teams:
        for away_team in teams:

            if home_team == away_team:
                continue

            home_xg, away_xg = expected_goals(
                home_team,
                away_team,
                ratings,
                home_goal_avg,
                away_goal_avg,
            )

            home_goals, away_goals = simulate_score(
                home_xg,
                away_xg,
            )

            table[home_team]["gf"] += home_goals
            table[home_team]["ga"] += away_goals

            table[away_team]["gf"] += away_goals
            table[away_team]["ga"] += home_goals

            if home_goals > away_goals:
                table[home_team]["points"] += 3
                table[home_team]["wins"] += 1
                table[away_team]["losses"] += 1

            elif home_goals < away_goals:
                table[away_team]["points"] += 3
                table[away_team]["wins"] += 1
                table[home_team]["losses"] += 1

            else:
                table[home_team]["points"] += 1
                table[away_team]["points"] += 1

                table[home_team]["draws"] += 1
                table[away_team]["draws"] += 1

    standings = sorted(
        teams,
        key=lambda team: (
            table[team]["points"],
            table[team]["gf"] - table[team]["ga"],
            table[team]["gf"],
        ),
        reverse=True,
    )

    return standings, table

def run_simulations(
    ratings,
    home_goal_avg,
    away_goal_avg,
    simulations=10000,
):
    teams = list(ratings.keys())

    summary = {
        team: {
            "positions": 0,
            "points": 0,
            "titles": 0,
            "top4": 0,
            "relegations": 0,
        }
        for team in teams
    }

    for _ in range(simulations):
        standings, table = simulate_season(
            ratings,
            home_goal_avg,
            away_goal_avg,
        )

        for position, team in enumerate(
            standings,
            start=1,
        ):
            summary[team]["positions"] += position
            summary[team]["points"] += table[team]["points"]

            if position == 1:
                summary[team]["titles"] += 1

            if position <= 4:
                summary[team]["top4"] += 1

            if position >= 18:
                summary[team]["relegations"] += 1

    results = []

    for team in teams:
        results.append(
            {
                "team": team,
                "avg_position": (
                    summary[team]["positions"]
                    / simulations
                ),
                "avg_points": (
                    summary[team]["points"]
                    / simulations
                ),
                "title_pct": (
                    summary[team]["titles"]
                    / simulations
                    * 100
                ),
                "top4_pct": (
                    summary[team]["top4"]
                    / simulations
                    * 100
                ),
                "relegation_pct": (
                    summary[team]["relegations"]
                    / simulations
                    * 100
                ),
            }
        )

    results.sort(
        key=lambda x: x["avg_position"]
    )

    return results

def save_results(results, filename):
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.DataFrame(results)

    output_path = RESULTS_DIR / filename
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved to: {output_path}")
if __name__ == "__main__":
    ratings, matches = build_2026_27_ratings()

    home_goal_avg, away_goal_avg = (
        calculate_league_averages(matches)
    )

    print("Teams:", len(ratings))
    print("Running simulations...")
    print()

    results = run_simulations(
        ratings,
        home_goal_avg,
        away_goal_avg,
        simulations=50000,
    )

    print("BASELINE 2026/27 PREDICTION")
    print("Historical results only")
    print()

    for position, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{position:2}. "
            f"{result['team']:18} "
            f"Pos {result['avg_position']:5.2f} | "
            f"Pts {result['avg_points']:5.1f} | "
            f"Title {result['title_pct']:5.1f}% | "
            f"Top4 {result['top4_pct']:5.1f}% | "
            f"Rel {result['relegation_pct']:5.1f}%"
        )
    save_results(
    results,
    "baseline_results_only_2026_27.csv",
)