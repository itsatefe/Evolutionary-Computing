import numpy as np
import random
import matplotlib.pyplot as plt

class Road_Alignment:

    def __init__(self, data, pop_size, max_generations=100, tournament_size=5, crossover_probability=0.8, uniform_crossover_probability=0.6, 
                 mutation_probability=0.1, uniform_mutation_probability=0.1, boltzmann_selection_temperature=1, elite_percentage=0.2, convergence_threshold=4e-1, no_improvement_threshold=10, type_crossover="uniform", type_mutation="uniform", type_parent_selection="tournament"):
        self.data = data
        self.shape = data.shape
        self.pop_size = pop_size
        self.population = None
        self.tournament_size = tournament_size
        self.boltzmann_selection_temperature = boltzmann_selection_temperature
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.uniform_mutation_probability = uniform_mutation_probability
        self.convergence_threshold = convergence_threshold
        self.no_improvement_threshold = no_improvement_threshold
        self.elite_percentage = elite_percentage
        self.max_generations = max_generations
        self.uniform_crossover_probability = uniform_crossover_probability
        self.type_crossover = type_crossover
        self.type_mutation = type_mutation
        self.type_parent_selection = type_parent_selection


    def initialize_population(self):
        self.population = []
        for i in range(self.pop_size):
            chromosome = [(i, random.randint(0, self.data.shape[0] - 1)) for i in range(0, self.data.shape[1])]
            self.population.append(chromosome)

    def calculate_fitness(self, chromosome):
        total_cost = 0.0
        for i in range(len(chromosome) - 1):
            x1, y1 = chromosome[i]
            x2, y2 = chromosome[i + 1]
            if abs(x1 - x2) == 1 and abs(y1 - y2) <= 1:
                road_cost = self.data[y1][x1]
                total_cost += road_cost
            else:
                distance_cost = (x2 - x1) ** 2 + (y2 - y1) ** 2
                total_cost += distance_cost
        return total_cost

    
    def parent_boltzmann_selection(self):
        fitness_values = np.array([self.calculate_fitness(individual) for individual in self.population])
        normalized_fitness = (fitness_values - np.min(fitness_values)) / (np.max(fitness_values) - np.min(fitness_values))
        probabilities = np.exp(-normalized_fitness / self.boltzmann_selection_temperature)
        probabilities /= np.sum(probabilities)
        selected_indices = np.random.choice(self.pop_size, size=2, p=probabilities)
        selected_parents = [self.population[i] for i in selected_indices]
        return selected_parents[0], selected_parents[1]

    def parent_tournament_selection(self):
        tournament_candidates = random.sample(self.population, self.tournament_size)
        winner = min(tournament_candidates, key=lambda ind: self.calculate_fitness(ind))
        tournament_candidates = random.sample(self.population, self.tournament_size)
        runner_up = min(tournament_candidates, key=lambda ind: self.calculate_fitness(ind))
        return winner, runner_up

    def uniform_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_probability:
            return parent1, parent2
        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)
        for i in range(len(parent1)):
            if np.random.rand() < self.uniform_crossover_probability:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
            else:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]
        return offspring1, offspring2

    def arithmetic_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_probability:
            return parent1, parent2
        alpha1 = random.uniform(0,1)
        alpha2 = 1.0 - alpha1
        offspring1 = []
        offspring2 = []
        for (x1,y1), (x2,y2) in zip(parent1, parent2):
            new_gene1 = x1, int(alpha1 * y1 + (1 - alpha1) * y2)
            new_gene2 = x2, int(alpha2 * y1 + (1 - alpha2) * y2)
            offspring1.append(new_gene1)
            offspring2.append(new_gene2)

        return offspring1, offspring2

    def one_point_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_probability:
            return parent1, parent2
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def uniform_mutate(self, chromosome):
        if np.random.rand() > self.mutation_probability:
            return chromosome
        mutated_chromosome = np.copy(chromosome)
        for i in range(len(chromosome)):
            if np.random.rand() < self.uniform_mutation_probability:
                x, y = mutated_chromosome[i]
                y = np.clip(y + np.random.randint(-2, 2), 0, self.data.shape[0] - 1)    
#                 if i != 0 and i != self.data.shape[1] - 1:
#                     x = np.clip(y + np.random.randint(0, 2), 0, self.data.shape[1] - 1)
                mutated_chromosome[i] = (x, y)
        offspring = list(mutated_chromosome)
        return offspring
    
    def inversion_mutation(self, chromosome):
        if np.random.rand() > self.mutation_probability:
            return chromosome
        point1, point2 = random.sample(range(1, len(chromosome)), 2)
        start, end = min(point1, point2), max(point1, point2)
        inverted_second_elements = [t[1] for t in chromosome[start:end+1]][::-1]
        mutated_chromosome = [(t[0], inverted_second_elements[i - start]) if start <= i <= end else t for i, t in enumerate(chromosome)]
        return mutated_chromosome

    
    def generate_offspring(self, num_offspring):
        offsprings = []
        while len(offsprings) < num_offspring:
            #-------------parent_selection-------------------
            if self.type_parent_selection == "tournament":
                parents = self.parent_tournament_selection()
                parent1, parent2 = parents[0], parents[1]
            else:
                parents = self.parent_boltzmann_selection()
                parent1, parent2 = parents[0], parents[1]
            #-------------crossover-------------------
            if self.type_crossover == "uniform":
                offspring1, offspring2 = self.uniform_crossover(parent1, parent2)
            elif self.type_crossover == "arithmetic":
                offspring1, offspring2 = self.arithmetic_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = self.one_point_crossover(parent1, parent2)
            #-------------mutation-------------------
            if self.type_mutation == "uniform":
                offspring1, offspring2 = self.uniform_mutate(offspring1), self.uniform_mutate(offspring2)
            else:
                offspring1, offspring2 = self.inversion_mutation(offspring1),self.inversion_mutation(offspring2)
            offsprings.extend([offspring1, offspring2])

        return offsprings[:num_offspring]


    def elitism_survival_selection(self, fitness_values, offsprings, num_elite):
        sorted_indices = np.argsort(fitness_values)
        elite_indices = sorted_indices[:num_elite]
        elite_individuals = [self.population[i] for i in elite_indices]
        new_generation = elite_individuals + offsprings
        return new_generation


    def terminate(self, best_fitness_values):
        if self.has_converged(best_fitness_values):
            return True

        if self.no_improvement(best_fitness_values):
            return True

        return False


    def has_converged(self, best_fitness_values):
        if len(best_fitness_values) > 1:
            recent_changes = [np.abs(best_fitness_values[i] - best_fitness_values[i - 1]) for i in range(1, len(best_fitness_values))]
            average_change = np.mean(recent_changes)
            return average_change / 100 < self.convergence_threshold
        return False

    def no_improvement(self, best_fitness_values):
        if len(best_fitness_values) > self.no_improvement_threshold:
            recent_best_fitness = best_fitness_values[-self.no_improvement_threshold:]
            improvement_check = all(recent_best_fitness[i] <= recent_best_fitness[i + 1] for i in range(self.no_improvement_threshold - 1))
            return improvement_check
        
        return False


    def extract_solution(self, population, fitness_values):
        best_index = np.argmin(fitness_values)
        best_individual = population[best_index]
        return best_individual


    def visualize_map(self):
        plt.imshow(self.data, cmap='YlGn', interpolation='nearest')
        cbar = plt.colorbar()
        plt.gca().invert_yaxis()
        cbar.set_label('Roughness Level')
        plt.title('Map Roughness')
        #     plt.show()
        return plt


    def visualize_chromosome(self, plt, chromosome):
        for i in range(len(chromosome) - 1):
            x1, y1 = chromosome[i]
            x2, y2 = chromosome[i + 1]
            if abs(x1 - x2) == 1 and abs(y1 - y2) <= 1:
                plt.plot([x1, x2], [y1, y2], linestyle='-', color='black')
            else:
                plt.plot([x1, x2], [y1, y2], linestyle='-', color='red')
        plt.title('chromosome')
        plt.show()


    def genetic_algorithm(self):
        self.initialize_population()
        average_fitness_values = []
        best_fitness_values = []

        last_generation = self.max_generations
        for generation in range(self.max_generations):
            fitness_values = [self.calculate_fitness(ind) for ind in self.population]
            best_fitness_values.append(min(fitness_values))
            average_fitness_values.append(np.mean(fitness_values))
            num_elite = int(len(self.population) * self.elite_percentage)
            offsprings = self.generate_offspring(len(self.population) - num_elite)
            self.population = self.elitism_survival_selection(fitness_values, offsprings, num_elite)
            if self.terminate(average_fitness_values):
  
                last_generation = generation
                break

        return last_generation, average_fitness_values, best_fitness_values


    def visulaize_solution(self):
        fitness_values = [self.calculate_fitness(ind) for ind in self.population]
        best_individual = self.extract_solution(self.population, fitness_values)
        print(self.calculate_fitness(best_individual))
        plt = self.visualize_map()
        self.visualize_chromosome(plt, best_individual)


    def plot_fitness_performance(self, last_generation, average_fitness_values, best_fitness_values):
        plt.figure()
        if last_generation != self.max_generations:
            plt.plot(range(1, last_generation + 2), average_fitness_values, label='Average Fitness', color='blue')
            plt.plot(range(1, last_generation + 2), best_fitness_values, label='best Fitness', color='green')
        else:
            plt.plot(range(1, last_generation + 1), average_fitness_values, label='Average Fitness', color='blue')
            plt.plot(range(1, last_generation + 1), best_fitness_values, label='best Fitness', color='green')
            
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Per Generation')
        plt.legend()
        plt.show()
        


    


