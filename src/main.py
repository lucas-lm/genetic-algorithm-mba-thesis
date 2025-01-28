import random
from datetime import datetime

import brute_force
import ga
from knapsack import KnapsackProblem


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
N = 50  # number of simulations
SELECTION_STRATEGIES = {
  "roulette": {"strategy": "selRoulette"},
  "tourn_3": {"strategy": "selTournament", "args": {"tournsize": 3}},
  "tourn_9": {"strategy": "selTournament", "args": {"tournsize": 9}},
  "tourn_27": {"strategy": "selTournament", "args": {"tournsize": 27}},
  "tourn_81": {"strategy": "selTournament", "args": {"tournsize": 81}},
}

ts = int(datetime.now().timestamp())

problem = KnapsackProblem.from_yaml("src/problem_setup_data.yml")

# sol generator: solutions = (KnapsackSolution(ind) for ind in pop)

# find best solution with brute force and save it
start_brute_force = datetime.now()
best_solution = brute_force.solve_knapsack(problem)[0]
best_solution\
  .include_execution_time(start_brute_force, datetime.now())\
  .save_to_json_file(f"output/{ts}/best_solution.json")

# for each selection strategy: repeat GA N times
for key, selection in SELECTION_STRATEGIES.items():
  for i in range(N):
    start_ga = datetime.now()
    result = ga.solve_knapsack(problem, {"selection": selection}).include_execution_time(start_ga, datetime.now())
    result.save_to_json_file(f"output/{ts}/{key}_iter_{i}.json")