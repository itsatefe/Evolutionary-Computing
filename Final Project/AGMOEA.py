import numpy as np
import random
import itertools
from Recombination import Recombination
from Chromosome import Chromosome
from Subspace import Subspace

class AGMOEA:
    def __init__(self, NP, K, NGBA, NEXA, Tmax, FETmax, evaluator, crossover_parameters, N, M):
        self.NP = NP # number of population
        self.K = K # number of intervals on each dimension
        self.NGBA = NGBA # maximum capacity for each subspace
        self.NEXA = NEXA #maximum capacity for External Archive
        self.Tmax = Tmax # The maximum generations
        self.EXA = []  # External archive
        self.GEXA = {}
        self.GBA = {}
        self.current_generation = 0
        self.Pm = 1/N # mutation probability
        self.N = N # number of decision variables
        self.S = set()
        self.operators = ['blx_alpha', 'sbx', 'spx', 'pcx', 'de_rand_1']
        self.operator_usage = {operator: 0 for operator in self.operators}
        self.operator_probabilities = {operator: 1.0 / len(self.operators) for operator in self.operators}
        self.pmin = 0.1  # Minimum selection probability for each operator
        self.M = M # number of objectives
        self.evaluator = evaluator  # Problem evaluator
        self.FET = 0
        self.FETmax = FETmax
        self.crossover_parameters = crossover_parameters
        self.lower_bounds, self.upper_bounds = self.evaluator.get_bounds()
        self.Test = []
        self.S_EXA = set()
        self.ideal_point = evaluator.ideal_point()
        self.nadir_point = evaluator.nadir_point()
        
    def set_bounds(self):
        return self.evaluator.get_bounds()
    
    def initialize_population(self):
        population = []
        for i in range(self.NP):
            decision_variables = np.array([np.random.uniform(low, high) for low, high in zip(self.lower_bounds,self.upper_bounds)])
            chromosome = Chromosome(decision_variables)
            chromosome.objectives = self.evaluate_individual(chromosome)
            population.append(chromosome)
        return population

    def construct_subspaces(self, solutions):
        grid_intervals = (np.array(self.nadir_point) - np.array(self.ideal_point)) / self.K
        self.GBA = {tuple(i): Subspace(coordinates=i, ideal_point=self.ideal_point, grid_intervals=grid_intervals) for i in self.generate_grid_coordinates()}
        for solution in solutions:
            relative_position = np.array(solution.objectives) - np.array(self.ideal_point)
            grid_coordinates = np.floor(relative_position / grid_intervals).astype(int)
            grid_coordinates = np.clip(grid_coordinates, 0, self.K - 1)
            self.GBA[tuple(grid_coordinates)].solutions.append(solution)
            
                
    def generate_grid_coordinates(self):
        coordinate_ranges = [range(self.K) for _ in range(self.M)]
        all_combinations = list(itertools.product(*coordinate_ranges))
        return all_combinations

    def polynomial_mutation_chromosome(self, chromosome, mutation_rate=0.1, eta_m=20):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                gene = chromosome[i]
                delta_1 = (gene - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
                delta_2 = (self.upper_bounds[i] - gene) / (self.upper_bounds[i] - self.lower_bounds[i])
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow
                gene = gene + delta_q * (self.upper_bounds[i] - self.lower_bounds[i])
                chromosome[i] = min(max(gene, self.lower_bounds[i]), self.upper_bounds[i])
        return chromosome

    
    def improve_EXA(self):
        grid_intervals = (np.array(self.nadir_point) - np.array(self.ideal_point)) / self.K
        self.GEXA = {tuple(i): Subspace(coordinates=i, ideal_point=self.ideal_point, grid_intervals=grid_intervals) for i in self.generate_grid_coordinates()}
        for solution in self.EXA:
            relative_position = solution.objectives - self.ideal_point
            grid_coordinates = np.floor(relative_position / grid_intervals).astype(int)
            grid_coordinates = np.clip(grid_coordinates, 0, self.K - 1)
            self.GEXA[tuple(grid_coordinates)].solutions.append(solution)
    
    
    def select_subspace_EXA(self):
        epsilon = 1e-6
        G_minus_S = [subspace for subspace in self.GEXA if subspace not in self.S_EXA]
        probabilities = {k: (1 / (self.SR(self.GEXA[k]) + epsilon)) for k in G_minus_S}
        total = sum(probabilities.values())
        normalized_probabilities = {k: (v / total) for k, v in probabilities.items()}
        selected_subspace = random.choices(list(normalized_probabilities.keys()), weights=normalized_probabilities.values(), k=1)[0]
        for subspace in self.GEXA.values():
            if selected_subspace.strong_subspace_dominance(subspace):
                self.S_EXA.add(subspace)
        return self.GEXA[selected_subspace]

    
    def SR(self, subspace):
        return sum(subspace.coordinates)

    def select_subspace(self):
        epsilon = 1e-6
        G_minus_S = [subspace for subspace in self.GBA if subspace not in self.S]
        probabilities = {k: (1 / (self.SR(self.GBA[k]) + epsilon)) for k in G_minus_S}
        total = sum(probabilities.values())
        normalized_probabilities = {k: (v / total) for k, v in probabilities.items()}
        selected_subspace = random.choices(list(normalized_probabilities.keys()), weights=normalized_probabilities.values(), k=1)[0]
        self.update_degraded_subspaces(self.GBA[selected_subspace])
        return self.GBA[selected_subspace]

    def update_degraded_subspaces(self, selected_subspace):
        self.S.clear()
        for subspace in self.GBA.values():
            if selected_subspace.strong_subspace_dominance(subspace):
                self.S.add(subspace)

    def adaptive_selection_probability(self):
        pre = 0.8
        pRE = pre / (1.0 + np.exp(-20 * ((self.current_generation / self.Tmax) - 0.25)))
        return pRE + 0.1
    
    def parent_selection(self, selected_subspace):
        parent1 = None
        if random.random() < self.adaptive_selection_probability() or not selected_subspace.solutions :
            parent1 = selected_subspace.select_representative()
            if parent1 is None:
                parent1 = random.choice(self.EXA)
            parent1 = parent1.values
        else:
            parent1 = random.choice(selected_subspace.solutions).values
        parents = []
        if len(self.EXA) < 3:
            parents = [parent.values for parent in random.sample(self.EXA, len(self.EXA))]
        else:
            parents = [parent.values for parent in random.sample(self.EXA, 3)]
        parents.insert(0, parent1)
        return np.array(parents)

    def update_operator_probabilities(self):
        total_solutions = len(self.EXA)
        for operator in self.operators:
            self.operator_usage[operator] = 0
        for solution in self.EXA:
            if solution.crossover_type == None:
                continue
            self.operator_usage[solution.crossover_type] += 1
        if total_solutions > 0:
            for operator in self.operators:
                self.operator_probabilities[operator] = max(self.operator_usage[operator] / total_solutions, self.pmin)
            total_probability = sum(self.operator_probabilities.values())
            if total_probability > 1.0:
                for operator in self.operators:
                    self.operator_probabilities[operator] /= total_probability


    def generate_offspring(self, selected_subspace):
        parents = self.parent_selection(selected_subspace)
        recombination = Recombination(parents, self.crossover_parameters)
        selected_operator = recombination.select_crossover_operator(self.operator_probabilities.items())
        values = recombination.execute_crossover(selected_operator)
        
        
        offsprings = []
        for value in values:
            if np.random.rand() < 1:
                value = self.polynomial_mutation_chromosome(value)
            offspring = Chromosome(value)
            np.clip(value, self.lower_bounds, self.upper_bounds, out=value)
            offspring.crossover_type = selected_operator
            offspring.objectives = self.evaluate_individual(offspring)
            offsprings.append(offspring)
        return offsprings

  
    def evaluate_individual(self, chromosome):
        self.FET += 1
        return self.evaluator.evaluate(chromosome.values)
        

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
        for solution in front:
            solution.crowding_distance = 0
        for i in range(self.M):
            front.sort(key=lambda solution: solution.objectives[i])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (front[j + 1].objectives[i] - front[j - 1].objectives[i])

    def manage_exa_capacity(self):
        if len(self.EXA) > self.NEXA:
            self.crowding_distance(self.EXA)
            self.EXA.sort(key=lambda solution: solution.crowding_distance)
            return self.EXA[:self.NEXA]
        else:
            return self.EXA

    
    def agmoea_algorithm(self):
        P = self.initialize_population()
        non_dominated_solutions = self.fast_non_dominated_sort(P)[0]
        self.EXA.extend(non_dominated_solutions)
        self.EXA = self.manage_exa_capacity()
        self.update_operator_probabilities()
        while not self.termination_criterion():
            self.S.clear()

            self.construct_subspaces(P)
#             self.improve_EXA()
            TP = []

            for _ in range(self.NP):
                selected_subspace = self.select_subspace()
                offsprings = self.generate_offspring(selected_subspace)
                TP += offsprings

            non_dominated_solutions = self.fast_non_dominated_sort(TP)[0]
            self.Test = non_dominated_solutions

            self.EXA.extend(non_dominated_solutions)
            self.EXA = self.fast_non_dominated_sort(self.EXA)[0]
            self.EXA = self.manage_exa_capacity()
            self.update_operator_probabilities()
            P.extend(TP)
            fronts = self.fast_non_dominated_sort(P)
            flattened_fronts = [item for sublist in fronts for item in sublist]
            P = flattened_fronts[:self.NP]

    def termination_criterion(self):
        if self.FETmax <= self.FET:
            return True
        if self.FET % 100 == 0:
            print("so far: ",self.FET)
        return False




