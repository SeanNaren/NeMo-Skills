import argparse
import requests
import json
import time
import os
from tqdm import tqdm

CONTEST_IDS = sorted({
    1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983,
    1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996,
    1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2013,
    2014, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
    2030, 2032, 2033, 2035, 2036,
})


def fetch_contest_data(contest_id):
    time.sleep(0.5)

    standings_url = f"https://codeforces.com/api/contest.standings?contestId={contest_id}&showUnofficial=false"
    ratings_url = f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}"

    standings_resp = requests.get(standings_url, timeout=15)
    ratings_resp = requests.get(ratings_url, timeout=15)

    standings_resp.raise_for_status()
    ratings_resp.raise_for_status()

    standings_data = standings_resp.json()
    ratings_data = ratings_resp.json()

    standings_rows = standings_data['result']['rows']
    rating_results = ratings_data['result']

    standings_handles = {row['party']['members'][0]['handle'] for row in standings_rows}
    ratings_handles = {change['handle'] for change in rating_results}
    common_handles = standings_handles.intersection(ratings_handles)

    filtered_standings = [row for row in standings_rows if row['party']['members'][0]['handle'] in common_handles]
    filtered_ratings = [change for change in rating_results if change['handle'] in common_handles]

    return {'rows': filtered_standings}, filtered_ratings


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Codeforces contest standings "
                    "and rating changes for simulating Elo rating."
    )
    parser.add_argument('--output_file', default="cf_ratings.json", type=str)
    args = parser.parse_args()

    all_contest_data = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            all_contest_data = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded data for {len(all_contest_data)} contests from {args.output_file}")

    processed_contest_ids = set(all_contest_data.keys())
    contests_to_fetch = [cid for cid in CONTEST_IDS if cid not in processed_contest_ids]

    if not contests_to_fetch:
        print("All specified contests have already been processed.")
    else:
        print(f"Found {len(processed_contest_ids)} already processed contests.")
        print(f"Processing {len(contests_to_fetch)} new contests.")

        for contest_id in tqdm(contests_to_fetch, desc="Fetching new contest data"):
            standings, ratings = fetch_contest_data(contest_id)
            all_contest_data[contest_id] = {"standings": standings, "rating_changes": ratings}

    with open(args.output_file, 'w') as f:
        json.dump(all_contest_data, f, separators=(',', ':'))
    print(f"\nData for {len(all_contest_data)} contests saved to {args.output_file}")


if __name__ == "__main__":
    main()
