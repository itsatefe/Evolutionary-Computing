import json
from ObjectiveFunction import ObjectiveFunction
import numpy as np

class EvolutionaryStrategy:
    def __init__(self, config):
        
        es_config = config.get("evolutionary_strategy", {})
        obj_func_config = es_config.get("objective_function_config", {})
        self.objective_function = ObjectiveFunction(
            obj_func_config.get("objective_function", "schwefel_function"),
            obj_func_config.get("chromosome_length", 5),
            obj_func_config.get("range", [-500, 500]))
        self.survival_method = es_config.get("survival_method", "elitism")
        self.max_generations = es_config.get("max_generations", 200)
        self.population_size = es_config.get("population_size", 15)
        self.chromosome_length = obj_func_config.get("chromosome_length", 5)
        self.num_offspring = es_config.get("num_offspring", 7 * 15)
        self.initial_sigma = es_config.get("initial_sigma", 0.1)
        mutation_enable = es_config.get("mutation_enable", "True")
        self.mutation_enable = bool(mutation_enable.lower() == "true")


    def initialize_population(self):
        min_value, max_value = self.objective_function.range_x
        object_parameters = np.random.uniform(min_value, max_value, size=(self.population_size, self.chromosome_length))
        return [list(object_parameter) + self.chromosome_length * [self.initial_sigma] for object_parameter in object_parameters]

    def generate_offspring_hybrid(self, population, current_num_generation):
        offsprings = []
        while len(offsprings) < self.num_offspring:
            offspring = self.hybrid_recombination(population, current_num_generation)
            object_p, strategy_p = self.uncorrelated_mutation_n_sigma(offspring[:self.chromosome_length], offspring[self.chromosome_length:])
            offspring = list(object_p) + list(strategy_p)
            offsprings.append(offspring)
        return offsprings[:self.num_offspring]

    def generate_offspring(self, population, current_num_generation):
        offsprings = []
        while len(offsprings) < self.num_offspring:
            selected_parents_indices = np.random.choice(len(population), size=2, replace=True)
            selected_parents = [population[i] for i in selected_parents_indices]
            offspring = self.local_discrete_recombination(selected_parents)
            if self.mutation_enable:
                object_p, strategy_p = self.uncorrelated_mutation_n_sigma(offspring[:self.chromosome_length], offspring[self.chromosome_length:])
                offspring = list(object_p) + list(strategy_p)
            offsprings.append(offspring)
        return offsprings[:self.num_offspring]
    def global_discrete_recombination(self, population):
        offspring = np.zeros_like(population[0])
        num_genes = len(population[0])
        for gene in range(num_genes):
            parent_index = np.random.randint(0, len(population))
            offspring[gene] = population[parent_index][gene]
        return offspring

    def local_discrete_recombination(self, parents):
        offspring = np.zeros_like(parents[0])
        for gene in range(len(offspring)):
            selected_parent_indices = np.random.choice(len(parents), size=1)[0]
            offspring[gene] = parents[selected_parent_indices][gene]
        return offspring
    
    def hybrid_recombination(self, population, current_num_generation):
        offspring = []
        if current_num_generation < self.max_generations * 0.3:
            offspring = self.global_discrete_recombination(population)
        else:
            selected_parents_indices = np.random.choice(len(population), size=2, replace=True)
            selected_parents = [population[i] for i in selected_parents_indices]
            offspring = self.local_discrete_recombination(selected_parents)
        return offspring
            

    def uncorrelated_mutation_n_sigma(self, chromosome, n_sigma, epsilon=1e-10):
        random_values = np.random.randn(len(chromosome))
        tau = 1 / np.sqrt(2 * len(chromosome))
        tau_prime = 1 / np.sqrt(2 * np.sqrt(len(chromosome)))
        new_n_sigma = n_sigma * np.exp(tau_prime * np.random.randn() + tau * random_values)
        mask = new_n_sigma < epsilon
        new_n_sigma[mask] = n_sigma[mask]
        random_values = np.random.randn(len(chromosome))
        mutated_chromosome = chromosome + new_n_sigma * random_values
        min_value, max_value = self.objective_function.range_x
        
        for i in range(len(mutated_chromosome)):
            if mutated_chromosome[i] < min_value:
                mutated_chromosome[i] = min_value
            elif mutated_chromosome[i] > max_value:
                mutated_chromosome[i] = max_value
                
        return mutated_chromosome, new_n_sigma

    def elitism_survival_selection(self, parents, offsprings):
        combined_population = np.vstack((parents, offsprings))
        fitness_values = self.objective_function.evaluate_list(combined_population[:, :self.chromosome_length])
        new_generation_indices = np.argsort(fitness_values)[:len(parents)]
        new_generation = combined_population[new_generation_indices]
        return new_generation

    def generational_survival_selection(self, parents, offsprings):
        chromosomes = np.array(offsprings)[:, :self.chromosome_length]
        fitness_values = self.objective_function.evaluate_list(chromosomes)
        new_generation_indices = np.argsort(fitness_values)[:len(parents)]
        new_generation = np.array(offsprings)[new_generation_indices]
        return new_generation


    def check_solution(self, population):
        chromosomes = np.array(population)[:, range(0, self.chromosome_length)]
        fitness_values = self.objective_function.evaluate_list(chromosomes)
        min_index = np.argmin(fitness_values)
        min_value = fitness_values[min_index]
        min_chromosome = chromosomes[min_index]
        return min_value, list(min_chromosome)


    def run(self):
        population = self.initialize_population()
        for current_num_generation in range(self.max_generations):
            new_generation = []
            offsprings = self.generate_offspring(population, current_num_generation)
            if self.survival_method == "elitism":
                new_generation = self.elitism_survival_selection(population, offsprings)
            elif self.survival_method == "generational":
                new_generation = self.generational_survival_selection(population, offsprings)
            population = new_generation
        min_value, min_chromosome = self.check_solution(population)
        return min_value, min_chromosome
    
    def calculate_convergence_speed(objective_values):
        num_iterations = len(objective_values)
        improvement = abs(objective_values[-1] - objective_values[0])
        convergence_speed = improvement / num_iterations
        return convergence_speed




