import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, NamedTuple, Literal

import yaml


# logging setup
logger = logging.getLogger("knapsack")
logger.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

# definitions
class KnapsackItem(NamedTuple):
    name: str
    weight: int
    value: int

    def as_dict(self):
        return self._asdict()

    @classmethod
    def from_tuple(cls, item_tuple: Tuple[str, int, int]):
        return cls(*item_tuple)
    
    @classmethod
    def from_dict(cls, item_dict: dict):
        return cls(**item_dict)


class KnapsackProblem:
    """The knapsack problem."""

    def __init__(self, capacity: int, available_items: Tuple[KnapsackItem, ...]):
        self._capacity = capacity
        self._available_items = available_items

    @property
    def available_items(self):
        return self._available_items
    
    @property
    def capacity(self):
        return self._capacity
    
    def get_number_of_items(self):
        return len(self.available_items)

    def get_total_combinations(self):
        return 2**len(self.available_items)
    
    def as_dict(self):
        available_items = [item.as_dict() for item in self.available_items]
        return {
            "capacity": self.capacity,
            "available_items": available_items
        }
    
    def as_json(self):
        return json.dumps(self.as_dict())

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path) as yaml_file:
            data = yaml.safe_load(yaml_file)

        if not isinstance(data, dict):
            raise Exception(f'YAML data should be a dictionary containing the keys capacity and items. {data = }')
        
        capacity = data.get('capacity')
        if not isinstance(capacity, int):
            raise Exception(f'`capacity` should be a number.\n{ capacity = }')
        
        all_items = data.get('items')
        available_items =  tuple(KnapsackItem.from_dict(item) for item in all_items)

        problem = cls(capacity, available_items)

        logger.info(f"Knapsack problem loaded from yaml: {yaml_path = }.")
        logger.info(f"Knapsack problem setup: {capacity = } and {len(available_items)} allowed items.")
        logger.info(f"Items:\n{available_items}")

        return problem


class KnapsackSolution:

    DEFAULT_OVERWEIGHT_PENALTY = 2

    def __init__(self, problem: KnapsackProblem, binary_code: List[Literal[0, 1]], overweight_penalty=None):
        self._problem = problem
        self._binary_code = binary_code
        self._overweight_penalty = overweight_penalty if overweight_penalty is not None else self.DEFAULT_OVERWEIGHT_PENALTY
        self._start_timestamp: datetime = None
        self._end_timestamp: datetime = None

    @property
    def problem(self):
        return self._problem
    
    @property
    def binary_code(self):
        return self._binary_code
    
    @property
    def overweight_penalty(self):
        return self._overweight_penalty
    
    @classmethod
    def from_bit_string(cls, problem: KnapsackProblem, bit_string: str, overweight_penalty=None):
        for char in bit_string:
            if char not in '01':
                raise ValueError(f"The argument {bit_string=} should be a string of 0s and 1s.")

        binary_code = [int(bit) for bit in bit_string]

        return cls(problem, binary_code, overweight_penalty)
    
    def include_execution_time(self, start_timestamp: datetime, end_timestamp: datetime = None):
        self._start_timestamp = start_timestamp
        self._end_timestamp = end_timestamp or datetime.now()
        return self

    def get_bit_string(self):
        return ''.join([str(bit) for bit in self.binary_code])
    
    def get_items(self):
        filtered_items = tuple()
        problem = self._problem
        # assumption: binary_code has same size as available_items
        # TODO: handle scenarios where bin and items have different sizes
        for i in range(len(problem.available_items)):
            if self._binary_code[i] == 1:
                filtered_items += (problem.available_items[i],)

        return filtered_items
    
    def get_total_value(self):
        items = self.get_items()
        return sum(item.value for item in items)
    
    def get_total_weight(self):
        items = self.get_items()
        return sum(item.weight for item in items)
    
    def compute_fitness(self):
        fitness = self.get_total_value() - self.overweight_penalty*self.overweight
        return max(0, fitness)
    
    @property
    def remaining_capacity(self):
        total_weight = self.get_total_weight()
        return max(0, self.problem.capacity - total_weight)
    
    @property
    def overweight(self):
        total_weight = self.get_total_weight()
        return max(0, total_weight - self.problem.capacity)
    
    @property
    def is_overweight(self):
        total_weight = self.get_total_weight()
        capacity = self.problem.capacity
        return total_weight > capacity
    
    def as_dict(self, json_serializable = False):
        solution_as_dict = {
            "binary_code": self.get_bit_string(),
            "total_value": self.get_total_value(),
            "total_weight": self.get_total_weight(),
            "fitness_score": self.compute_fitness(),
            "remaining_capacity": self.remaining_capacity,
            "overweight": self.overweight,
            "is_overweight": self.is_overweight,
            "items": [item.as_dict() for item in self.get_items()],
        }

        execution_time = None
        if self._start_timestamp and self._end_timestamp:
            start_ts = self._start_timestamp
            end_ts = self._end_timestamp

            if json_serializable:
                start_ts = start_ts.timestamp()
                end_ts = end_ts.timestamp()

            execution_time = {
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_in_seconds": (self._end_timestamp - self._start_timestamp).total_seconds()
            }

        return {
            "execution_time": execution_time,
            "solution": solution_as_dict,
            "problem": self.problem.as_dict()
        }

    def save_to_json_file(self, json_path: str):

        if os.path.exists(json_path):
            raise FileExistsError(f"The file '{json_path}' already exists.")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w') as json_file:
            json.dump(self.as_dict(json_serializable=True), json_file, indent=2)

        return

    def __repr__(self):
        b = self.get_bit_string()
        f = self.compute_fitness()
        v = self.get_total_value()
        w = self.get_total_weight()
        p = self.overweight_penalty
        over = self.overweight
        return f"KnapsackSolution(bitstring = {b}, overweight_penalty = {p}, fitness = {f}, value = {v}, weight = {w}, overweight = {over})"

    def __str__(self):
        bitstring = self.get_bit_string()
        return bitstring

# test
def main():
    import random
    problem = KnapsackProblem.from_yaml('src/problem_setup_data.yml')
    print(getattr(problem, 'capacity'))

    solution = KnapsackSolution(problem, [random.randint(0, 1) for _ in range(22)])
    print(solution.binary_code)
    print(solution.problem.capacity)
    print(solution.get_items())
    print(solution.get_total_value())
    print(solution.get_total_weight())
    print(solution.remaining_capacity)
    print(solution.overweight)
    print(solution.is_overweight)

if __name__ == "__main__":
    main()