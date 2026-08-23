from scipy.stats import poisson

from data_loader import load_premier_league_data
from ratings import calculate_team_ratings, calculate_league_averages


def expected_goals(home_team, away_team, ratings, home_goal_avg, away_goal_avg):
    home_rating = ratings[home_team]
    away_rating = ratings[away_team]

    home_xg = (
        home_goal_avg
        * home_rating["home_attack"]
        * away_rating["away_defence"]
    )

    away_xg = (
        away_goal_avg
        * away_rating["away_attack"]
        * home_rating["home_defence"]
    )

    return home_xg, away_xg


def predict_match(home_team, away_team, ratings, home_goal_avg, away_goal_avg):
    home_xg, away_xg = expected_goals(
        home_team,
        away_team,
        ratings,
        home_goal_avg,
        away_goal_avg,
    )

    home_win = 0
    draw = 0
    away_win = 0

    max_goals = 10

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):

            probability = (
                poisson.pmf(home_goals, home_xg)
                * poisson.pmf(away_goals, away_xg)
            )

            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
    }


if __name__ == "__main__":
    matches = load_premier_league_data()

    ratings = calculate_team_ratings(matches)
    home_goal_avg, away_goal_avg = calculate_league_averages(matches)

    test_matches = [
        ("Arsenal", "Liverpool"),
        ("Liverpool", "Arsenal"),
        ("Man City", "Arsenal"),
        ("Arsenal", "Man City"),
        ("Liverpool", "Man City"),
        ("Man City", "Liverpool"),
    ]

    for home_team, away_team in test_matches:
        result = predict_match(
            home_team,
            away_team,
            ratings,
            home_goal_avg,
            away_goal_avg,
        )

        print(f"{home_team} vs {away_team}")
        print(
            f"xG: {result['home_xg']:.2f} - "
            f"{result['away_xg']:.2f}"
        )
        print(
            f"{home_team}: {result['home_win'] * 100:.1f}% | "
            f"Draw: {result['draw'] * 100:.1f}% | "
            f"{away_team}: {result['away_win'] * 100:.1f}%"
        )
        print()