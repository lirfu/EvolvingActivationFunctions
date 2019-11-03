from activation import Tree
from utils import Stopwatch, roulette_wheel_select
import crossovers as crx
import random


def print_done_with_stopwatch(s):
    dt = s.lap()
    print("===> Done! ({}h {}m {}s)".format(dt.hour, dt.minute, dt.second + dt.microsecond / 1e6))


def will_occur(prob):
    return random.random() < prob


class GA:
    def __init__(self, params, initializer, evaluator):
        self.params = params
        try:
            self.initializer = params['initializer']
            self.evaluator = params['evaluator']
            self.stop_condition = StopCondition(params)
        except KeyError:
            print('')
            return

        self.population = []
        self.crossovers = []
        self.mutators = []

    def __initialize_population(self):
        self.population = []
        for i in range(self.params['population_size']):
            g = Genotype()
            self.initializer.initialize(g)
            self.evaluator.evaluate(g)
            self.population.append(g)

    def __select_parents(self):
        return roulette_wheel_select(self.population, [g.fitness for g in self.population], 2)

    def __get_best(self):
        return self.population[max(self.population, key=lambda g: g.fitness)]

    def __crossover(self, parents):
        return roulette_wheel_select(self.crossovers, [c.weight for c in self.crossovers]).cross(parents)

    def __mutate(self, g):
        roulette_wheel_select(self.mutators, [m.weight for m in self.mutators]).mutate(g)

    def run(self):
        iteration = 0
        best_unit = None
        evaluations = 0
        elapsed_time = 0
        stopwatch = Stopwatch()
        stopwatch.start()

        print("===> Initializing population!")
        __initialize_population()
        best_unit = __get_best().clone()
        print_done_with_stopwatch(stopwatch)

        print("===> Starting algorithm with population of {} units!".format(len(self.population)))
        while not self.stop_condition.is_satisfied(iteration, best_unit, evaluations, elapsed_time):
            iteration += 1

            if self.params['elitism']:  # Save the queen.
                new_pop.append(__get_best().clone())

            for i in range(self.params['elitism'] - (self.params['elitism'] ? 1 : 0)):
                parents = __select_parents(self.population)
                children = __crossover(parents)
                for c in children:
                    if will_occur(self.params['mutation_prob']):
                        __mutate(c)
        print_done_with_stopwatch(stopwatch)


class Initializer:
    def initialize(self, g):
        g.tree = None
        raise Exception('Missing implementation!')

class Evaluator:
    def evaluate(self, g):
        g.fitness = None
        raise Exception('Missing implementation!')

class StopCondition:
    def __init__(self, params):
        self.max_iterations = params.get('max_iterations', None)
        self.max_evaluations = params.get('max_evaluations', None)
        self.max_time = params.get('max_time', None)
        self.min_fitness = params.get('min_fitness', None)
        self.max_fitness = params.get('max_fitness', None)

    def is_satisfied(self, iteration, best_unit, evaluations, elapsed_time):
        return (self.max_iterations and self.max_iterations <= iteration) or \
               (self.min_fitness and self.min_fitness >= best_unit.fitness) or \
               (self.max_fitness and self.max_fitness <= best_unit.fitness) or \
               (self.max_evaluations and self.max_evaluations <= evaluations) or \
               (self.max_time and self.max_time <= elapsed_time)

    def report(self, iteration, best_unit, evaluations, elapsed_time):
        s = ""
        if max_iterations and max_iterations <= iteration:
            s += "Max iterations achieved!\n"
        if min_fitness and min_fitness >= best_unit.fitness:
            s += "Min fitness achieved!\n"
        if max_fitness and max_fitness <= best_unit.fitness:
            s += "Max fitness achieved!\n"
        if max_evaluations and max_evaluations <= evaluations:
            s += "Max evaluations achieved!\n"
        if max_time and max_time <= elapsed_time:
            s += "Max time achieved!\n"
        if not s:
            return "No condition satisfied!"
        return s


class Genotype:
    def __init__(self):
        self.fitness = None
        self.tree = None

    def __init__(self, tree):
        self.fitness = None
        self.tree = tree

    def get(self, index):
        return self.tree.get(index)

    def set(self, index, value):
        n = self.tree.get(index)
        n.swap(value)

    def size(self):
        return self.tree.size()

    def clone(self):
        return Genotype(self.tree)
