import numpy as np
import random
import itertools

class AGMOEA:
    def __init__(self, NP, K, NGBA, NEXA, Tmax, N, M):
        self.NP = NP # number of population
        self.K = K # number of intervals on each dimension
        self.NGBA = NGBA # maximum capacity for each subspace
        self.NEXA = NEXA #maximum capacity for External Archive
        self.Tmax = Tmax # The maximum generations
        self.EXA = []  # External archive
        self.GEXA = {}
        self.GBA = {}
        self.current_generation = 0
        self.Pm = 1/n # mutation probability
        self.N = N # number of decision variables
        self.S = set()
        self.operators = ['BLX-α', 'SBX', 'SPX', 'PCX', 'DE/rand/1']
        self.operator_usage = {operator: 0 for operator in self.operators}
        self.operator_probabilities = {operator: 1.0 / len(self.operators) for operator in self.operators}
        self.pmin = 0.1  # Minimum selection probability for each operator
        self.M = M # number of objectives
        self.evaluator = evaluator  # Problem evaluator

    def initialize_population(self):
        population = []
        for i in range(self.NP):
            decision_variables = np.random.rand(self.N)
            objective_values = self.evaluator.evaluate(decision_variables)
            chromosome = Chromosome(decision_variables, objective_values)
            population.append(chromosome)
        return population

    def construct_subspaces(self, solutions, ideal_point, nadir_point):
        grid_intervals = (nadir_point - ideal_point) / self.K
        self.GBA = {tuple(i): Subspace(coordinates=i, ideal_point=ideal_point, grid_intervals=grid_intervals) for i in self.generate_grid_coordinates()}
        for solution in solutions:
            relative_position = solution.objective_values - ideal_point
            grid_coordinates = np.floor(relative_position / grid_intervals).astype(int)
            self.GBA[tuple(grid_coordinates)].solutions.append(solution)
                
    def generate_grid_coordinates(self):
        coordinate_ranges = [range(self.K) for _ in range(self.M)]
        all_combinations = list(itertools.product(*coordinate_ranges))
        return all_combinations

    def polynomial_mutation_chromosome(self, chromosome, mutation_rate=0.1, eta_m=20, lower_bound=0, upper_bound=1):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                gene = chromosome[i]
                delta_1 = (gene - lower_bound) / (upper_bound - lower_bound)
                delta_2 = (upper_bound - gene) / (upper_bound - lower_bound)
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

                gene = gene + delta_q * (upper_bound - lower_bound)
                chromosome[i] = min(max(gene, lower_bound), upper_bound)
        return chromosome

    
    def improve_EXA(self):
        ideal_point = 
        nadir_point =
        grid_intervals = (nadir_point - ideal_point) / self.K
        self.GEXA = {tuple(i): Subspace(coordinates=i, ideal_point=ideal_point, grid_intervals=grid_intervals) for i in self.generate_grid_coordinates()}
        for solution in self.EXA:
            relative_position = solution.objective_values - ideal_point
            grid_coordinates = np.floor(relative_position / grid_intervals).astype(int)
            self.GEXA[tuple(grid_coordinates)].solutions.append(solution)
            
            
        return

    def SR(self, subspace):
        return sum(subspace.grid_coordinates)

    def select_subspace(self):
        epsilon = 1e-6
        G_minus_S = [subspace for subspace in self.GBA if subspace not in self.S]
        probabilities = {k: (1 / (self.SR(k) + epsilon)) for k in G_minus_S}
        total = sum(probabilities.values())
        normalized_probabilities = {k: (v / total) for k, v in probabilities.items()}
        selected_subspace = random.choices(list(normalized_probabilities.keys()), weights=normalized_probabilities.values(), k=1)[0]
        self.update_degraded_subspaces(selected_subspace)
        return selected_subspace

    def update_degraded_subspaces(self, selected_subspace):
        for subspace in self.GBA:
            if selected_subspace.strong_subspace_dominance(subspace):
                self.S.add(subspace)

    def adaptive_selection_probability(self):
        pre = 0.8
        pRE = pre / (1.0 + np.exp(-20 * ((self.current_generation / self.Tmax) - 0.25)))
        return pRE + 0.1
    
    def parent_selection(self, selected_subspace):
        if random.random() < self.adaptive_selection_probability():
            parent1 = selected_subspace.select_representative()
            if parent1 is None:
                # select from EXA
#                 parent1 = random.choice(self.EXA)
        else:
            # Select a random individual from the solutions within the selected subspace
            parent1 = random.choice(selected_subspace.solutions, k=1)
        # select from EXA
#         parent2 = random.choice(self.EXA)
        # in case some operators need 3 parents
        # select from EXA
    
#         parent3 = random.choice(self.EXA)
        return parent1, parent2, parent3

    def update_operator_probabilities(self):
        total_solutions = len(self.EXA)
        if total_solutions > 0:
            for operator in self.operators:
                self.operator_probabilities[operator] = max(self.operator_usage[operator] / total_solutions, self.pmin)
            total_probability = sum(self.operator_probabilities.values())
            if total_probability > 1.0:
                for operator in self.operators:
                    self.operator_probabilities[operator] /= total_probability

    def select_crossover_operator(self):
        operators, probabilities = zip(*self.operator_probabilities.items())
        selected_operator = random.choices(operators, weights=probabilities, k=1)[0]
        return selected_operator

    def generate_offspring(self, parent1, parent2):
        # Select a crossover operator based on updated probabilities
        selected_operator = self.select_crossover_operator()
        # This is a placeholder for the actual crossover implementation
        # in each crossover when we want to make an object of chromosome becareful of crossover type just in case
        offspring = crossover(selected_operator, parent1, parent2)
        self.operator_usage[selected_operator] += 1
        return offspring

    # fix this one
    def evaluate_individual(self, individual):
        objective_values = self.evaluator.evaluate(decision_variables)
        pass

    def fast_non_dominated_sort(self, population):
        # Sort the population based on non-domination criteria
        pass
    def crowding_distance(self):
        pass

    def agmoea_algorithm(self):
        # Generate initial population P0

        P0 = self.initialize_population()
        # Evaluate individuals in P0
        for individual in P0:
            self.evaluate_individual(individual)

        # Store non-dominated solutions in P0 into EXA
        self.EXA.extend(self.fast_non_dominated_sort(P0))

        # Main loop
        while not self.termination_criterion():
            self.S.clear()
            
            # Construct subspaces
            self.construct_subspaces()
            
            # Improve EXA
            self.improve_EXA()

            # Set TP to be empty
            TP = []  # Temporary population

            # Generate offsprings and evaluate
            for _ in range(NP):
                # Select a subspace
                selected_subspace, S = self.select_subspace(G, S)
                
                # Generate an offspring
                offspring = self.generate_offspring(subspace)
                
                # Evaluate offspring and store in TP
                self.evaluate_individual(offspring)
                TP.append(offspring)

            # Update EXA with TP
            self.EXA.extend(self.fast_non_dominated_sort(TP))

            # something is wrong here
            # Update population P with TP
            P = TP
            # Set P to be the set of the best NP individuals in P based on fast non-dominated sorting
            P = self.fast_non_dominated_sort(P)[:NP]

    def termination_criterion(self):
        # Define the termination criterion for the algorithm
        pass



# In[ ]:




