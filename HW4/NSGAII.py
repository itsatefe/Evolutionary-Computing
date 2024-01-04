from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from DataProcessor import DataProcessor
import numpy as np
import math
from Chromosome import *
from Recombination import Recombination
import random
import numpy as np
from Chromosome import Chromosome
from Recombination import Recombination
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


class NSGAII:
    def __init__(self,X,y, population_size, LP, num_features, true_pareto_front, maxFEs, sigma=0.0001):
        self.population_size = population_size
        self.LP = LP
        self.mutation_probability= 1 / num_features
        self.num_features = num_features
        self.objective_functions =  [self.fitness_function_1, self.fitness_function_2]
        self.maxFEs = maxFEs
        self.sigma = sigma
        self.crossover_keys = ["onepoint_crossover", "twopoint_crossover", "uniform_crossover", "shuffle_crossover", "reduced_surrogate_crossover"]
        self.default_value = 0
        self.nReward = {key: self.default_value for key in self.crossover_keys}
        self.nPenalty = {key: self.default_value for key in self.crossover_keys}
        self.RD = []
        self.PN = []
        self.X = X
        self.y = y
        self.true_pareto_front = true_pareto_front
        
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            values = list(np.random.randint(2, size=self.num_features))
            chromosome = Chromosome(values)
            chromosome.objectives = [f(values) for f in self.objective_functions]
            population.append(chromosome)
        return population

    def environmental_selection(self, population):
        remaining_pop_size = self.population_size
        new_population = []
        front_0 = []
        fronts = self.fast_non_dominated_sort(population)
        for i, front in enumerate(fronts):
            front_size = len(front)
            if remaining_pop_size > front_size:
                new_population += front
                remaining_pop_size -= front_size
                
            else:
                self.crowding_distance(front)
                front.sort(key=lambda chromosome: chromosome.crowding_distance, reverse=True)
                new_population += front[:remaining_pop_size]
                
                break
        return new_population, fronts[0]

    def fast_non_dominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = set()

            for q in population:
                if p.dominate(q):
                    p.dominated_solutions.add(q)
                elif q.dominate(p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            fronts.append(next_front)
        return fronts

    def crowding_distance(self, front):
        if not front:
            return
        num_objectives = len(front[0].objectives)
        for solution in front:
            solution.crowding_distance = 0
        for i in range(num_objectives):
            front.sort(key=lambda solution: solution.objectives[i])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (front[j + 1].objectives[i] - front[j - 1].objectives[i])

    def fitness_function_1(self, chromosome):
        chromosome = np.array(chromosome)
        selected_features = self.X[:, chromosome == 1]
        classifier = KNeighborsClassifier(n_neighbors=3)
        accuracy_scores = cross_val_score(classifier, selected_features, self.y, cv=3)
        mean_error = 1 - np.mean(accuracy_scores)
        return mean_error

    def fitness_function_2(self, chromosome):
        return np.sum(chromosome)

    def init_OSP(self):
        return {key: 1/len(self.crossover_keys) for key in self.crossover_keys}

    def update_OSP(self):
        sum_crossover_RD = {key: sum(row[key] for row in self.RD) for key in self.crossover_keys}
        sum_crossover_PN = {key: sum(row[key] for row in self.PN) for key in self.crossover_keys}
        non_zero_sum = {key: sum_crossover_RD[key] if sum_crossover_RD[key] != 0 else sigma for key in self.crossover_keys}
        crossover_probabilities = {key: sum_crossover_RD[key] / (non_zero_sum[key] + sum_crossover_PN[key]) for key in sum_crossover_RD.keys()}
        denominator = sum(crossover_probabilities.values())
        normalized_probabilities = {key: crossover_probabilities[key] / denominator for key in crossover_probabilities.keys()}
        return normalized_probabilities

    def parent_selection(self, population):
        return np.random.choice(population,2)

    def credit_assignment(self, parents, offsprings, crossover):
        p_nd, p_d = self.dominance_comparison(parents)
        if p_d != None:
            for parent in p_nd:
                for offspring in offsprings:
                    if parent.dominate(offspring):
                        self.nPenalty[crossover.__name__] += 1
                    else: 
                        self.nReward[crossover.__name__] += 1
        else:
            for offspring in offsprings:
                if all(not parent.dominate(offspring) for parent in parents):
                    self.nReward[crossover.__name__] += 1
                else:
                    self.nPenalty[crossover.__name__] += 1    
  

    # if nobody dominates the solution it goes to non-dominated set
    def dominance_comparison(self, chromosomes):
        non_dominated_set = []
        dominated_set = []
        for i, sol1 in enumerate(chromosomes):
            is_dominated_by_others = any(sol2.dominate(sol1) for j, sol2 in enumerate(chromosomes) if i != j)
            if not is_dominated_by_others:
                non_dominated_set.append(sol1)
            else:
                dominated_set.append(sol1)
        return non_dominated_set, dominated_set

    def uniform_mutation(self, chromosome):
        mutated_values = [1 - val if np.random.rand() < self.mutation_probability else val for val in chromosome]
        return mutated_values

    def find_non_dominated_solution(self, population):
        fronts = self.fast_non_dominated_sort(population)
        return fronts[0]

    def calculate_hypervolume(self, population):
        objectives = np.array([chromosome.objectives for chromosome in population])
        reference_point =  np.array([1, self.num_features])
        ind = HV(ref_point=reference_point)
        return ind(objectives)
    
    def calculate_igd(self, population):
        objectives = np.array([chromosome.objectives for chromosome in population])
        pf = np.array(objectives)
        A = np.array(self.true_pareto_front)
        ind = IGD(A)
        return ind(pf)

    
    def nsga2(self):
        count_evaluation = 0
        k = 0
        population = self.initialize_population()
        crossover_probability = self.init_OSP()
        hypervolume_values = []
        igd_values = []
        while count_evaluation < self.maxFEs:
            self.default_value = 0
            self.nReward = {key: self.default_value for key in self.crossover_keys}
            self.nPenalty = {key: self.default_value for key in self.crossover_keys}
            new_population = []
            
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.parent_selection(population)
                rc = Recombination(parent1.values, parent2.values)
                crossover = rc.roulette_wheel_selection(crossover_probability)
                offspring1, offspring2 = rc.execute_crossover(crossover)
                offspring1 = self.uniform_mutation(offspring1)
                offspring2 = self.uniform_mutation(offspring2)
                
                offspring1, offspring2 = self.avoid_zero_offspring(offspring1, offspring2)   
             
                offspring1, offspring2 = Chromosome(offspring1), Chromosome(offspring2)
                offspring1.objectives = [f(offspring1.values) for f in self.objective_functions]
                offspring2.objectives = [f(offspring2.values) for f in self.objective_functions]
                count_evaluation += 2
                self.credit_assignment([parent1, parent2], [offspring1, offspring2], crossover)
                new_population += [offspring1, offspring2]

            k += 1
            self.RD.append(self.nReward)
            self.PN.append(self.nPenalty)

            if k == self.LP:
                crossover_probability = self.update_OSP()
                k = 0

            current_pool = new_population + population

            distinct_objects = []
            for obj in current_pool:
                if not any(existing_obj.values == obj.values for existing_obj in distinct_objects):
                    distinct_objects.append(obj)
                    
            population, current_solutions = self.environmental_selection(distinct_objects)
#             current_solutions = self.find_non_dominated_solution(population)
            hypervolume = self.calculate_hypervolume(current_solutions)
            hypervolume_values.append(hypervolume)
            igd = self.calculate_igd(current_solutions)
            igd_values.append(igd)
            
            if count_evaluation % 100000 == 0:
                print(f"Evaluations: {count_evaluation}, Hypervolume: {hypervolume}")

       
        return current_solutions, igd_values, hypervolume_values
    
    def avoid_zero_offspring(self, offspring1, offspring2):
        not_zero_offspring1 = offspring1.copy()
        not_zero_offspring2 = offspring2.copy()

        while not_zero_offspring1 == [0] * self.num_features:
            not_zero_offspring1 = self.uniform_mutation(offspring1)

        while not_zero_offspring2 == [0] * self.num_features:
            not_zero_offspring2 = self.uniform_mutation(offspring2)

        return not_zero_offspring1, not_zero_offspring2
            
