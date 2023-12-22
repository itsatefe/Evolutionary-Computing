import json
from ObjectiveFunction import ObjectiveFunction
import numpy as np
import matplotlib.pyplot as plt

class EvolutionaryStrategy:
    def __init__(self, config):
        es_config = config.get("evolutionary_strategy", {})
        obj_func_config = es_config.get("objective_function_config", {})
        mutation_func_config = es_config.get("mutation_function_config", {})
        sigma_config = es_config.get("sigma_config", {})
        mutation_enable = mutation_func_config.get("mutation_enable", "True")
        self.mutation_enable = bool(mutation_enable.lower() == "true")
        self.mutation_probability = mutation_func_config.get("mutation_probability", 0.8)
        self.mean_log = mutation_func_config.get("mean_log", 0.6)
        self.sigma_log = mutation_func_config.get("sigma_log", 0.4)
        self.objective_function = ObjectiveFunction(
            obj_func_config.get("objective_function", "schwefel_function"),
            obj_func_config.get("chromosome_length", 5),
            obj_func_config.get("range", [-500, 500]))
        
        self.initialization_sigma_method = sigma_config.get("initialization_sigma_method","default")
        self.random_sigma_range = sigma_config.get("random_sigma_range",[0,10])
        self.initial_sigma = sigma_config.get("initial_sigma", 0.1)
        self.sigma_threshold = sigma_config.get("sigma_threshold", 1e-5)
        
        self.survival_method = es_config.get("survival_method", "elitism")
        self.max_generations = es_config.get("max_generations", 200)
        self.population_size = es_config.get("population_size", 15)
        self.chromosome_length = obj_func_config.get("chromosome_length", 5)
        self.num_offspring = es_config.get("num_offspring", 7 * 15)
        

        self.convergence_threshold = es_config.get("convergence_threshold", 1e-5)
        self.no_improvement_threshold = es_config.get("no_improvement_threshold", 1e-5)




    def initialize_population(self):
        min_value, max_value = self.objective_function.range_x
        object_parameters = np.random.uniform(min_value, max_value, size=(self.population_size, self.chromosome_length))
        strategy_parameters = []
        if self.initialization_sigma_method == 'default':
            strategy_parameters = self.chromosome_length * [self.initial_sigma]
        elif self.initialization_sigma_method == 'random':
            min_sigma, max_sigma = self.random_sigma_range
            strategy_parameters = list(np.random.uniform(min_sigma, max_sigma, self.chromosome_length))
        return [list(object_parameter) + strategy_parameters for object_parameter in object_parameters]

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
            if self.objective_function.name == "schwefel_function":
                offspring = self.local_discrete_recombination(selected_parents)
            elif self.objective_function.name == "ackley_function":
                offspring = self.global_discrete_recombination(population)
            if self.mutation_enable:
                object_p, strategy_p = self.uncorrelated_mutation_n_sigma(offspring[:self.chromosome_length], offspring[self.chromosome_length:])
            else:
                object_p = self.non_adaptive_mutation(offspring[:self.chromosome_length])
                #However, strategy parameters are not going to be used, since adaptive mutation is disabled.
                strategy_p = offspring[self.chromosome_length:]
                
            offspring = list(object_p) + list(strategy_p)
            offsprings.append(offspring)
        return offsprings[:self.num_offspring]
    
    # The larger the mean_log, the more the distribution is shifted towards larger changes.
    # A larger sigma_log will increase the variability, allowing for a wider range of changes.
    # creep_mutation
    def non_adaptive_mutation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_probability:
                mutation_value = np.random.lognormal(self.mean_log, self.sigma_log)
                mutated_chromosome[i] += mutation_value
        return mutated_chromosome
        
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
            

    def uncorrelated_mutation_n_sigma(self, chromosome, n_sigma):
        random_values = np.random.randn(len(chromosome))
        tau = 1 / np.sqrt(2 * len(chromosome))
        tau_prime = 1 / np.sqrt(2 * np.sqrt(len(chromosome)))
        new_n_sigma = n_sigma * np.exp(tau_prime * np.random.randn() + tau * random_values)
        mask = new_n_sigma < self.sigma_threshold
        new_n_sigma[mask] = n_sigma[mask]
        random_values = np.random.randn(len(chromosome))
        mutated_chromosome = chromosome + new_n_sigma * random_values
        mutated_chromosome = self._check_boundaries(mutated_chromosome)
        return mutated_chromosome, new_n_sigma
    
    def _check_boundaries(self, mutated_chromosome):
        min_value, max_value = self.objective_function.range_x
        for i in range(len(mutated_chromosome)):
            if mutated_chromosome[i] < min_value:
                mutated_chromosome[i] = min_value
            elif mutated_chromosome[i] > max_value:
                mutated_chromosome[i] = max_value
        return mutated_chromosome
    
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
        average_fitness_values = []
        best_fitness_values = []
        fitness_std = []
        last_generation = self.max_generations
        for current_num_generation in range(self.max_generations):
            chromosomes = np.array(population)[:, range(0, self.chromosome_length)]
            fitness_values = self.objective_function.evaluate_list(chromosomes)
            best_fitness_values.append(min(fitness_values))
            average_fitness_values.append(np.mean(fitness_values))
            fitness_std.append(np.std(fitness_values))
            new_generation = []
            offsprings = self.generate_offspring(population, current_num_generation)
            if self.survival_method == "elitism":
                new_generation = self.elitism_survival_selection(population, offsprings)
            elif self.survival_method == "generational":
                new_generation = self.generational_survival_selection(population, offsprings)
            population = new_generation
          

            if self.terminate(average_fitness_values):
                last_generation = current_num_generation
                break
                
        convergence_speed = self.calculate_convergence_speed(average_fitness_values)
        print(f'Convergence Speed at Generation based on avg-fitness-values {current_num_generation + 1}: {convergence_speed}')
        convergence_speed = self.calculate_convergence_speed(best_fitness_values)
        print(f'Convergence Speed at Generation based on best-fitness-values {current_num_generation + 1}: {convergence_speed}')
        min_value, min_chromosome = self.check_solution(population)
            
        return min_value, min_chromosome, last_generation, average_fitness_values, best_fitness_values, fitness_std
    
    def terminate(self, best_fitness_values):
        if self.has_converged(best_fitness_values):
            print("converged")
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

    
    def calculate_convergence_speed(objective_values):
        num_iterations = len(objective_values)
        improvement = abs(objective_values[-1] - objective_values[0])
        convergence_speed = improvement / num_iterations
        return convergence_speed
    
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
        
    def calculate_convergence_speed(self, objective_values):
        num_iterations = len(objective_values)
        improvement = abs(objective_values[-1] - objective_values[0])
        convergence_speed = improvement / num_iterations
        return convergence_speed
        
    def plot_diversity(self,fitness_stddev, last_generation):
        if last_generation != self.max_generations:
            plt.plot(range(1, last_generation + 2), fitness_stddev, linestyle='-')
        else:
            plt.plot(range(1, last_generation + 1), fitness_stddev, linestyle='-')

            
        plt.title('Diversity Per Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Standard Deviation')
        plt.show()




