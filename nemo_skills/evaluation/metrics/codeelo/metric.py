import json
import multiprocessing
import os
from collections import defaultdict
from functools import lru_cache

from nemo_skills.evaluation.metrics.base import BaseMetrics


def calc_elo_rating(contest_id, problem_status, cf_contest_cache):
    print(f"Calculating rating for contest_id: {contest_id}")

    cached_standings_result = cf_contest_cache["standings"]
    cached_rating_changes_result = cf_contest_cache["rating_changes"]
    # Mimic the structure returned by requests.get().json()
    standings = {"status": "OK", "result": cached_standings_result}
    rating_changes = {"status": "OK", "result": cached_rating_changes_result}

    try:
        handle_set = set([standings["result"]["rows"][i]["party"]["members"][0]["handle"] for i in
                          range(len(standings["result"]["rows"]))]) and \
                     set([rating_changes["result"][i]["handle"] for i in range(len(rating_changes["result"]))])
        standings["result"]["rows"] = [standings["result"]["rows"][i] for i in range(len(standings["result"]["rows"]))
                                       if standings["result"]["rows"][i]["party"]["members"][0]["handle"] in handle_set]
        rating_changes["result"] = [rating_changes["result"][i] for i in range(len(rating_changes["result"])) if
                                    rating_changes["result"][i]["handle"] in handle_set]
        assert len(standings["result"]["rows"]) == len(rating_changes["result"]) and len(
            standings["result"]["rows"]) > 200
        print(f"Number of handle: {len(handle_set)}")
    except Exception as e:
        print(e)

    contest_name = standings["result"]["contest"]["name"]

    print(f"Contest name: {contest_name}")
    # if "Div. 1" in contest_name and "Div. 2" not in contest_name:
    #     print("Pass Div. 1")
    #     return

    if "result" not in standings or "result" not in rating_changes or len(standings["result"]["rows"]) != len(
            rating_changes["result"]) or len(standings["result"]["rows"]) <= 200:
        print("No result")
        return
    max_rating = 0
    for i in range(len(rating_changes["result"])):
        max_rating = max(max_rating, rating_changes["result"][i]["oldRating"])
    score = 0
    penalty = 0
    for problem in standings["result"]["problems"]:
        prob = f"{problem['contestId']}{problem['index']}"
        if prob in problem_status.keys():
            for ith, status in enumerate(problem_status[prob]):
                if status == "AC":
                    print(f"AC at {prob} in {ith + 1}th submission, total submissions: {len(problem_status[prob])}")
                    if "points" in problem:
                        score += max(0, problem["points"] - 50 * ith)
                    else:
                        score += 1
                        penalty += ith * 10
                    break

    print(f"Score: {score}, Penalty: {penalty}")

    n = len(standings["result"]["rows"])
    print(f"Number of participants: {n}, {len(rating_changes['result'])}")
    rank = n
    for i in range(n):
        if standings["result"]["rows"][i]["points"] < score or (
                standings["result"]["rows"][i]["points"] == score and standings["result"]["rows"][i][
            "penalty"] > penalty):
            rank = i
            break
    print(f"Rank: {rank}")

    l, r = 0, max_rating + 100
    while r - l > 1:
        mid = int((l + r) / 2)
        new_seed = 1
        for i in range(n):
            new_seed += 1 / (1 + 10 ** ((mid - rating_changes["result"][i]["oldRating"]) / 400))
        if new_seed < rank:
            r = mid
        else:
            l = mid

    print(f"Rating: {l}")
    return l


def _process_contest_for_elo(args):
    """Helper function to process a single contest for multiprocessing."""
    contest_id, problems_in_contest, cf_cache = args
    statuses_for_calc = {}
    for index, statuses in problems_in_contest.items():
        problem_id = f"{contest_id}{index}"
        statuses_for_calc[problem_id] = statuses
    rating = calc_elo_rating(contest_id, statuses_for_calc, cf_cache)
    return float(rating)


@lru_cache()
def _load_cf_cache_from_file(path):
    with open(path) as f:
        cache_data = json.load(f)
        return cache_data


class CodeEloMetrics(BaseMetrics):
    def __init__(self):
        """Initializes the metrics object, loading shared resources."""
        self.cf_cache = None
        self.cf_cache = {}
        if "CF_RATINGS_PATH" in os.environ:
            self.cf_cache = _load_cf_cache_from_file(os.environ["CF_RATINGS_PATH"])
        else:
            raise ValueError(
                "You must provide CF_RATINGS_PATH as an environment variable, pointing to the cf_ratings.json file."
            )
        super().__init__()

    def reset(self):
        """Resets only the accumulated submission history for the instance."""
        self.submissions_by_contest = defaultdict(lambda: defaultdict(list))

    def update(self, predictions):
        """
        Accumulates submission statuses for each problem within its contest.
        Args:
            predictions (list[dict]): ... (rest of docstring) ...
        """
        if len(predictions) > 1:
            self.agg_mode = f"pass@{len(predictions)}"
        elif len(predictions) == 1:
            self.agg_mode = "greedy"

        for pred in predictions:
            if 'interactive problem' in pred['question'].lower():
                print('skipping interactive problem, by setting them correct', pred['cf_contest_id'], pred['cf_index'])
                continue
                # continue
            contest_id = int(pred['cf_contest_id'])
            index = str(pred['cf_index'])
            is_correct = bool(pred['is_correct'])
            status = "AC" if is_correct else "WA"
            self.submissions_by_contest[contest_id][index].append(status)

    def get_metrics(self):
        """
        Calculates the average Elo rating based on the accumulated submission history
        across all processed contests using multiprocessing.
        """
        num_contests = len(self.submissions_by_contest)

        print(f"Processing {num_contests} unique contests...")

        # Prepare list of arguments for each task
        tasks = []
        for contest_id, problems_in_contest in self.submissions_by_contest.items():
            tasks.append((str(contest_id), problems_in_contest, self.cf_cache[str(contest_id)]))

        num_workers = 32  # Set default number of workers

        with multiprocessing.Pool(processes=num_workers) as pool:
            # map applies the function to each item in tasks iterable across the pool
            elo_ratings = pool.map(_process_contest_for_elo, tasks)

        average_elo = sum(elo_ratings) / len(elo_ratings)
        print(f"\n--- Elo Calculation Summary ---")
        print(f"Elo ratings obtained for {len(elo_ratings)} contests: {[f'{r:.2f}' for r in elo_ratings]}")
        print(f"Average Elo: {average_elo:.2f}")

        return {self.agg_mode: {"elo rating": str(round(average_elo))}}
