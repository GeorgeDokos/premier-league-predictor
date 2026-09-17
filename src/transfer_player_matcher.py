import pandas as pd
import unicodedata
import re

from player_data_loader import load_player_data
from transfer_loader import load_saved_transfers
from rapidfuzz import process, fuzz
from broader_player_loader import load_broader_player_data

def prepare_broader_players():
    players = load_broader_player_data().copy()

    players = players.rename(
        columns={
            "name": "Player"
        }
    )

    players["name_key"] = (
        players["Player"]
        .apply(normalize_name)
    )

    players["data_source"] = "broader"

    return players


def prepare_original_players():
    players = load_player_data().copy()

    players["name_key"] = (
        players["Player"]
        .apply(normalize_name)
    )

    players["data_source"] = "original"

    return players

def normalize_league_name(league):
    if pd.isna(league):
        return None

    league = str(league).strip()

    prefixes = [
        "eng ",
        "it ",
        "es ",
        "de ",
        "fr ",
    ]

    for prefix in prefixes:
        if league.lower().startswith(prefix):
            league = league[len(prefix):]

    return league.strip()

def normalize_name(name):
    if pd.isna(name):
        return ""

    name = unicodedata.normalize(
        "NFKD",
        str(name),
    )

    name = "".join(
        char
        for char in name
        if not unicodedata.combining(char)
    )

    name = name.lower()

    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    return name

def find_fuzzy_match(name_key, player_names, threshold=88):
    if not name_key:
        return None

    result = process.extractOne(
        name_key,
        player_names,
        scorer=fuzz.token_sort_ratio,
    )

    if result is None:
        return None

    matched_name, score, _ = result

    if score >= threshold:
        return matched_name

    return None

def match_transfers_to_players():
    transfers = load_saved_transfers().copy()

    transfers["name_key"] = (
        transfers["Player"]
        .apply(normalize_name)
    )

    broader = prepare_broader_players()
    original = prepare_original_players()

    broader_names = (
        broader["name_key"]
        .dropna()
        .unique()
        .tolist()
    )

    original_names = (
        original["name_key"]
        .dropna()
        .unique()
        .tolist()
    )

    broader_exact = set(broader_names)
    original_exact = set(original_names)

    results = []

    for _, transfer in transfers.iterrows():
        key = transfer["name_key"]

        matched_key = None
        source = None

        # 1. Exact broader match
        if key in broader_exact:
            matched_key = key
            source = "broader"

        # 2. Exact original match
        elif key in original_exact:
            matched_key = key
            source = "original"

        # 3. Fuzzy broader match
        else:
            fuzzy = find_fuzzy_match(
                key,
                broader_names,
            )

            if fuzzy is not None:
                matched_key = fuzzy
                source = "broader"

        # 4. Fuzzy original match
        if matched_key is None:
            fuzzy = find_fuzzy_match(
                key,
                original_names,
            )

            if fuzzy is not None:
                matched_key = fuzzy
                source = "original"

        row = transfer.to_dict()

        row["matched_name_key"] = matched_key
        row["data_source"] = source

        results.append(row)

    matches = pd.DataFrame(results)

    broader_lookup = broader.drop_duplicates(
        subset="name_key",
        keep="first",
    ).set_index("name_key")

    original_lookup = original.drop_duplicates(
        subset="name_key",
        keep="first",
    ).set_index("name_key")

    enriched_rows = []

    for _, row in matches.iterrows():
        result = row.to_dict()

        matched_key = row["matched_name_key"]
        source = row["data_source"]

        if pd.notna(matched_key):

            if source == "broader":
                player_row = broader_lookup.loc[matched_key]

            elif source == "original":
                player_row = original_lookup.loc[matched_key]

            else:
                player_row = None

            if player_row is not None:

                if source == "broader":
                    result["stats_name"] = player_row.get("Player")
                    result["stats_league"] = player_row.get("league")
                    result["stats_position"] = player_row.get("position")
                    result["stats_minutes"] = player_row.get("minutes_played")
                    result["stats_goals"] = player_row.get("goals")
                    result["stats_assists"] = player_row.get("assists")
                    result["stats_xg"] = player_row.get("expected_goals")
                    result["stats_xa"] = player_row.get("expected_assists")
                    result["stats_rating"] = player_row.get("rating")
                    result["stats_market_value"] = player_row.get("market_value")
                    result["stats_tackles"] = player_row.get("tackles")
                    result["stats_interceptions"] = player_row.get("interceptions")
                    result["stats_saves"] = player_row.get("saves")

                elif source == "original":
                    result["stats_name"] = player_row.get("Player")
                    result["stats_league"] = normalize_league_name(
                        player_row.get("Comp"))
                    result["stats_position"] = player_row.get("Pos")
                    result["stats_minutes"] = player_row.get("Min")
                    result["stats_goals"] = player_row.get("Gls")
                    result["stats_assists"] = player_row.get("Ast")

                    result["stats_xg"] = None
                    result["stats_xa"] = None
                    result["stats_rating"] = None
                    result["stats_market_value"] = None
                    result["stats_tackles"] = player_row.get("TklW")
                    result["stats_interceptions"] = player_row.get("Int")
                    result["stats_saves"] = player_row.get("Saves")

        enriched_rows.append(result)

    matches = pd.DataFrame(enriched_rows)

    return matches, broader, original

def parse_fee(fee):
    if pd.isna(fee):
        return 0.0

    fee = str(fee).lower()

    if "free" in fee or "undisclosed" in fee:
        return 0.0

    fee = re.sub(r"\[\d+\]", "", fee)

    match = re.search(r"£([\d.]+)m", fee)

    if match:
        return float(match.group(1))

    match = re.search(r"£([\d,]+)", fee)

    if match:
        return float(
            match.group(1).replace(",", "")
        ) / 1_000_000

    return 0.0

if __name__ == "__main__":
    matches, broader, original = (
        match_transfers_to_players()
    )

    found = matches[
        matches["matched_name_key"].notna()
    ]

    missing = matches[
        matches["matched_name_key"].isna()
    ]
    missing = missing.copy()

    missing["fee_millions"] = (
        missing["Fee"]
        .apply(parse_fee)
    )

    important_missing = missing[
        missing["fee_millions"] >= 5
    ].sort_values(
        "fee_millions",
        ascending=False,
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


    missing["incoming_pl"] = (
        missing["Moving to"]
        .isin(PREMIER_LEAGUE_TEAMS)
    )

    fallback_candidates = missing[
        (
            missing["fee_millions"] >= 5
        )
        |
        (
            missing["incoming_pl"]
        )
    ].copy()

    fallback_candidates = fallback_candidates.sort_values(
        ["incoming_pl", "fee_millions"],
        ascending=[False, False],
    )

    print()
    print("FALLBACK CANDIDATES")
    print()

    print(
        fallback_candidates[
            [
                "Player",
                "Moving from",
                "Moving to",
                "Fee",
                "fee_millions",
                "incoming_pl",
            ]
        ].to_string(index=False)
    )

    print()
    print("IMPORTANT UNMATCHED TRANSFERS")
    print()

    print(
        important_missing[
            [
                "Player",
                "Moving from",
                "Moving to",
                "Fee",
                "fee_millions",
            ]
        ].to_string(index=False)
    )

    print("Total transfers:", len(matches))
    print("Matched players:", len(found))
    print("Unmatched players:", len(missing))

    print()
    print("MATCH RATE")
    print(
        f"{len(found) / len(matches) * 100:.1f}%"
    )

    print()
    print("MATCHES BY SOURCE")

    print(
        found["data_source"]
        .value_counts()
        .to_string()
    )

    print()
    print("UNMATCHED PLAYERS")

    print(
        missing[
            [
                "Player",
                "Moving from",
                "Moving to",
            ]
        ].to_string(index=False)
    )
    print()
    print("ENRICHED MATCH SAMPLE")
    print()

    sample = matches[
        matches["matched_name_key"].notna()
    ].head(10)

    columns_to_show = [
        "Player",
        "Moving from",
        "Moving to",
        "data_source",
        "matched_name_key",
    ]

    extra_columns = [
        "stats_name",
        "stats_league",
        "stats_position",
        "stats_minutes",
        "stats_goals",
        "stats_assists",
        "stats_xg",
        "stats_xa",
        "stats_rating",
        "stats_market_value",
    ]

    print(
        sample[
            columns_to_show + extra_columns
        ].to_string(index=False)
    )