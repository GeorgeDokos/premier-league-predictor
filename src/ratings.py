from data_loader import load_premier_league_data


def weighted_average(values, weights):
    return (values * weights).sum() / weights.sum()


def calculate_league_averages(matches):
    home_goals = weighted_average(matches["FTHG"], matches["Weight"])
    away_goals = weighted_average(matches["FTAG"], matches["Weight"])

    return home_goals, away_goals


def calculate_team_ratings(matches):
    home_goal_avg, away_goal_avg = calculate_league_averages(matches)

    teams = sorted(
        set(matches["HomeTeam"]).union(set(matches["AwayTeam"]))
    )

    ratings = {}

    for team in teams:
        home_matches = matches[matches["HomeTeam"] == team]
        away_matches = matches[matches["AwayTeam"] == team]

        if home_matches.empty or away_matches.empty:
            continue

        home_scored = weighted_average(
            home_matches["FTHG"],
            home_matches["Weight"],
        )

        home_conceded = weighted_average(
            home_matches["FTAG"],
            home_matches["Weight"],
        )

        away_scored = weighted_average(
            away_matches["FTAG"],
            away_matches["Weight"],
        )

        away_conceded = weighted_average(
            away_matches["FTHG"],
            away_matches["Weight"],
        )

        ratings[team] = {
            "home_attack": home_scored / home_goal_avg,
            "home_defence": home_conceded / away_goal_avg,
            "away_attack": away_scored / away_goal_avg,
            "away_defence": away_conceded / home_goal_avg,
        }

    return ratings

def calculate_championship_ratings(matches):
    home_goal_avg = matches["FTHG"].mean()
    away_goal_avg = matches["FTAG"].mean()

    teams = sorted(
        set(matches["HomeTeam"]).union(matches["AwayTeam"])
    )

    ratings = {}

    for team in teams:
        home_matches = matches[matches["HomeTeam"] == team]
        away_matches = matches[matches["AwayTeam"] == team]

        home_scored = home_matches["FTHG"].mean()
        home_conceded = home_matches["FTAG"].mean()

        away_scored = away_matches["FTAG"].mean()
        away_conceded = away_matches["FTHG"].mean()

        ratings[team] = {
            "home_attack": home_scored / home_goal_avg,
            "home_defence": home_conceded / away_goal_avg,
            "away_attack": away_scored / away_goal_avg,
            "away_defence": away_conceded / home_goal_avg,
        }

    return ratings

def convert_promoted_rating(rating):
    promoted_baseline = {
        "home_attack": 0.632,
        "away_attack": 0.759,
        "home_defence": 1.256,
        "away_defence": 1.365,
    }

    championship_weight = 0.25
    baseline_weight = 0.75

    return {
        "home_attack": (
            baseline_weight * promoted_baseline["home_attack"]
            + championship_weight
            * promoted_baseline["home_attack"]
            * rating["home_attack"]
        ),

        "away_attack": (
            baseline_weight * promoted_baseline["away_attack"]
            + championship_weight
            * promoted_baseline["away_attack"]
            * rating["away_attack"]
        ),

        "home_defence": (
            baseline_weight * promoted_baseline["home_defence"]
            + championship_weight
            * promoted_baseline["home_defence"]
            * rating["home_defence"]
        ),

        "away_defence": (
            baseline_weight * promoted_baseline["away_defence"]
            + championship_weight
            * promoted_baseline["away_defence"]
            * rating["away_defence"]
        ),
    }

if __name__ == "__main__":
    matches = load_premier_league_data()

    ratings = calculate_team_ratings(matches)

    for team in ["Arsenal", "Liverpool", "Man City"]:
        print(team)
        print(ratings[team])
        print()