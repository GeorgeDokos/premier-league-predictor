import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}


def api_get(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("errors"):
        raise RuntimeError(data["errors"])

    remaining = response.headers.get(
        "x-ratelimit-requests-remaining"
    )

    print(f"API requests remaining today: {remaining}")

    return data

if __name__ == "__main__":
    data = api_get(
        "leagues",
        {
            "country": "England",
            "season": 2025,
        },
    )

    for item in data["response"]:
        print(
            item["league"]["id"],
            item["league"]["name"],
        )