import random
from datetime import datetime
from pathlib import Path

import brute_force
import ga
from knapsack import KnapsackProblem


RANDOM_SEED = 42  # seed to reproduce the results
random.seed(RANDOM_SEED)
N = 50  # number of simulations
OVERWEIGHT_PENALTY = 2

# selections to vary
SELECTION_STRATEGIES = {
  "roulette": {"strategy": "selRoulette"},
  "tourn_3": {"strategy": "selTournament", "args": {"tournsize": 3}},
  "tourn_9": {"strategy": "selTournament", "args": {"tournsize": 9}},
  "tourn_27": {"strategy": "selTournament", "args": {"tournsize": 27}},
  "tourn_81": {"strategy": "selTournament", "args": {"tournsize": 81}},
  "ranking": {"strategy": ga.selRanking},
}

ts = int(datetime.now().timestamp())
BASE_LOCATION = Path(f"output/{ts}_penalty_{OVERWEIGHT_PENALTY}")


problem = KnapsackProblem.from_yaml("src/problem_setup_data.yml")


# find best solution with brute force and save it
start_brute_force = datetime.now()
best_solution = brute_force.solve_knapsack(problem)[0]
best_solution\
  .include_execution_time(start_brute_force, datetime.now())\
  .save_to_json_file(BASE_LOCATION / "best_solution.json")

# for each selection strategy: repeat GA N times
for key, selection_config in SELECTION_STRATEGIES.items():
  for i in range(N):
    start_ga = datetime.now()
    
    algo_config = {"overweight_penalty": OVERWEIGHT_PENALTY, "selection": selection_config}
    result = ga.solve_knapsack(problem, algo_config)\
      .include_execution_time(start_ga, datetime.now())
    
    result.save_to_json_file(BASE_LOCATION / f"{key}_iter_{i+1}.json")