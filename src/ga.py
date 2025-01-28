import os
import random
import json
import yaml

from datetime import datetime

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


from knapsack import KnapsackSolution, KnapsackProblem


# Defaults:
POPULATION_SIZE = 100
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 10
P_CROSSOVER = 1  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual

# with reduce: from functools import reduce
# def get_nested(obj, *keys, default=None):
#     try:
#         return reduce(lambda d, key: d[key], keys, obj)
#     except (KeyError, TypeError):
#         return default

def _population_stringfy(population):  # population -> ([1,0, 1...], [1, 0, 0...]...)
    return tuple(''.join([str(i) for i in ind]) for ind in population)
    

def _dict_get_nested(obj: dict, *keys, default=None):
    k = keys[0]
    obj = obj.get(k, default)
    has_more_keys = len(keys) > 1
    
    if has_more_keys:
        if type(obj) is not dict:
            # obj is not a dict and has more keys to go through.
            # return default in this case and stop the recursion removing the keys
            return default
        
        # obj is a dict and there is more keys to go through
        return _dict_get_nested(obj, *keys[1:], default=default)
    # end of recursion returning the final value
    return obj

class ToolboxBuilder:
    def __init__(self):
        self._missing = ("individual", "evaluate", "select", "mate", "mutate")
        self._toolbox = base.Toolbox()

    @property
    def missing(self):
        return self._missing
    
    @property
    def is_incomplete(self):
        return len(self._missing) > 0
    
    def _remove_missings(self, *setup: str):
        return tuple(s for s in self._missing if s not in setup)

    def set_individual(self, length):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self._toolbox.register("zeroOrOne", random.randint, 0, 1)
        self._toolbox.register("individualCreator", tools.initRepeat, creator.Individual, self._toolbox.zeroOrOne, length)
        self._toolbox.register("populationCreator", tools.initRepeat, list, self._toolbox.individualCreator)
        self._missing = self._remove_missings("individual")
        return self

    def set_fitness_function(self, fn):
        self._toolbox.register("evaluate", fn)
        self._missing = self._remove_missings("evaluate")
        return self
    
    def set_selection_strategy(self, strategy: str = 'selRoulette', **args):
        selection_function = getattr(tools, strategy)
        self._toolbox.register("select", selection_function, **args)
        self._missing = self._remove_missings("select")
        return self
    
    def set_crossover_operator(self, rate=P_CROSSOVER):
        self._toolbox.register("mate", tools.cxTwoPoint)
        self._toolbox.register("get_crossover_rate", lambda: rate)
        self._missing = self._remove_missings("mate")
        return self
    
    def set_mutation_operator(self, rate=P_MUTATION, prob=0.1):
        self._toolbox.register("mutate", tools.mutFlipBit, indpb=prob)
        self._toolbox.register("get_mutation_rate", lambda: rate)
        self._missing = self._remove_missings("mutate")
        return self
    
    # unstable
    def load_from_yaml(self, yaml_path: str):
        with open(yaml_path) as yaml_file:
            toolbox_raw_data =  yaml.safe_load(yaml_file)

        if not isinstance(toolbox_raw_data, dict):
            raise Exception("Invalid YAML file.")
        
        # selection_strategy = _dict_get_nested(toolbox_raw_data, "selection", "strategy", default='selRoulette')
        # selection_arguments = _dict_get_nested(toolbox_raw_data, "selection", "strategy", default={})
        crossover_rate = _dict_get_nested(toolbox_raw_data, "crossover", "rate", default=P_CROSSOVER)
        mutation_rate = _dict_get_nested(toolbox_raw_data, "mutation", "rate", default=P_MUTATION)
        mutation_prob = _dict_get_nested(toolbox_raw_data, "mutation", "prob", default=0.1)

        return (self
            .set_selection_strategy()  # change here parameters
            .set_crossover_operator(rate=crossover_rate)
            .set_mutation_operator(rate=mutation_rate, prob=mutation_prob)
        )
    
    @classmethod
    def init_from_yaml(cls, yaml_path: str):
        new_builder = cls()
        return new_builder.load_from_yaml(yaml_path)
    
    def build(self):
        if self.is_incomplete:
            str_formatted_missing = ', '.join([f"'{m}'" for m in self.missing])
            raise ValueError(f"Missing setup: {str_formatted_missing}")
        return self._toolbox


class GenAlResult:

    def __init__(self, logbook: tools.Logbook, hall_of_fame = None):
        self._logbook = logbook
        self._hall_of_fame = hall_of_fame
        self._start_timestamp: datetime = None
        self._end_timestamp: datetime = None

    @property
    def hall_of_fame(self):
        return self._hall_of_fame

    def include_execution_time(self, start_timestamp: datetime, end_timestamp: datetime = None):
        self._start_timestamp = start_timestamp
        self._end_timestamp = end_timestamp or datetime.now()
        return self

    def get_generations(self):
        return self._logbook.select('population')

    def get_fittest(self):
        if self._hall_of_fame:
            return self._hall_of_fame[0]

    def as_dict(self, json_serializable=False):
        logbook = [record for record in self._logbook]
        
        hof = []
        for individual in self._hall_of_fame:
            stringfied_individual = ''.join([str(bit) for bit in individual])
            individual_fitness = individual.fitness.getValues()[0]
            hof.append((stringfied_individual, individual_fitness))
        
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
            "fittest": hof[0],
            "hall_of_fame": hof, 
            "logbook": logbook
        }

    def save_to_json_file(self, json_path: str):

        if os.path.exists(json_path):
            raise FileExistsError(f"The file '{json_path}' already exists.")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w') as json_file:
            json.dump(self.as_dict(json_serializable=True), json_file, indent=2)

        return

class GenAl:

    def __init__(self, toolbox, population_initial_size: int = POPULATION_SIZE, max_generations: int = MAX_GENERATIONS):    
        self._toolbox = toolbox
        self._population_initial_size = population_initial_size
        self._max_generations = max_generations

    @classmethod
    def from_yaml(cls, yaml_path: str): ...


    def _create_random_population(self, size):
        return self._toolbox.populationCreator(n=size)
    

    def _create_stats_object(self):
        stats = tools.Statistics()
        stats.register("population", _population_stringfy)
        return stats

    def run(self, verbose=False):
        initial_population = self._create_random_population(self._population_initial_size)
        stats = self._create_stats_object()
        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)  # enchament: use from parameter

        _, logbook = algorithms.eaSimple(
            initial_population, 
            self._toolbox, 
            cxpb=self._toolbox.get_crossover_rate(), 
            mutpb=self._toolbox.get_mutation_rate(),
            ngen=self._max_generations, 
            stats=stats, 
            halloffame=hof, 
            verbose=verbose
        )
        return GenAlResult(logbook, hof)

# fitness calculation
# class KnapsackGA(ga_spec)[.solve(problem), .generate_seeds, .repeat_solve(problem, n)]
def compute_knapsack_fitness(problem):
    def compute(individual):
        solution_instance = KnapsackSolution(problem, individual)
        fitness = solution_instance.compute_fitness(overweight_penalty_weight=2)
        return fitness,  # return a tuple
    return compute


def solve_knapsack(problem: KnapsackProblem, algo_config: dict = None) -> GenAlResult:
    raise NotImplementedError() 

# define args: problem, algoConfig (pop_size, max_gen, p_crossover, p_mutation, selection) -> toolbox...
def main():

    problem = KnapsackProblem.from_yaml("src/problem_setup_data.yml")
    individual_genes_count = problem.get_number_of_items()

    # selRoulette
    # selTournament (tournSize=3)
    # selBest -> selRanking
    toolbox = (
        ToolboxBuilder()
            .set_individual(length=individual_genes_count)
            .set_fitness_function(compute_knapsack_fitness(problem))
            .set_selection_strategy(strategy='selRoulette')  # change here parameters
            .set_crossover_operator()
            .set_mutation_operator(prob=1/individual_genes_count)
            .build()
        )
    
    genalgo = GenAl(toolbox)


    start = datetime.now()
    result = genalgo.run().include_execution_time(start, end_timestamp=datetime.now())

    ts = int(datetime.now().timestamp())
    result.save_to_json_file(f'output/simulation_{ts}.json')


if __name__ == "__main__":
    main()