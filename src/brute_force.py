from datetime import datetime
from knapsack import KnapsackProblem, KnapsackSolution


def binary_code_generator(binary_length: int):
  """
  Generator to yield all combinations of 0s and 1s for an array of length n.

  :param n: Number of elements in the array.
  """
  for i in range(2 ** binary_length):
    yield [int(bit) for bit in f"{i:0{binary_length}b}"]


# TODO: introduce arg 'rank_size' to get top N solutions
def solve_knapsack(problem: KnapsackProblem):

  high_score = 0
  best_solutions = []
  all_items_count = problem.get_number_of_items()

  for binary_code in binary_code_generator(all_items_count):

    if binary_code.count(1) == 1:
      print(f'Partial progress... {binary_code = }')

    new_solution = KnapsackSolution(problem, binary_code)

    if new_solution.is_overweight:  # skip invalid solutions
      continue

    new_solution_score = new_solution.compute_fitness()
    if new_solution_score > high_score:
      best_solutions = [new_solution]
      high_score = new_solution_score
    elif new_solution_score == high_score:
      best_solutions.append(new_solution)

  return tuple(best_solutions)


def main():
  problem = KnapsackProblem.from_yaml("src/problem_setup_data.yml")
  best_solutions = solve_knapsack(problem)

  print('#'*32)
  print("# BEST SOLUTIONS FOUND:")
  print("*"*32)
  for sol in best_solutions:
    print(f'Binary Code = {sol.binary_code}')
    print(f'Fitness Score = {sol.compute_fitness()}  (also total value)')
    print(f'Remaining capacity = {sol.remaining_capacity}')
    print('-'*32)

    ts = int(datetime.now().timestamp())
    sol.save_to_json_file(f"output/best_solution_{ts}.json")

if __name__ == "__main__":
  main()