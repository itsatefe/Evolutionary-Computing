import random
import numpy as np


class Recombination:
    def __init__(self, parent1, parent2):
        self.parent1 = parent1
        self.parent2 = parent2
        self.crossovers = [self.onepoint_crossover, self.twopoint_crossover, self.uniform_crossover, self.shuffle_crossover, self.reduced_surrogate_crossover]
        
    def onepoint_crossover(self):
        crossover_point = random.randint(1, len(self.parent1) - 1)
        offspring1 = self.parent1[:crossover_point] + self.parent2[crossover_point:]
        offspring2 = self.parent2[:crossover_point] + self.parent1[crossover_point:]
        return offspring1, offspring2
    
   
    def shuffle_crossover(self):
        indices = list(range(len(self.parent1)))
        random.shuffle(indices)
        shuffled_parent1 = [self.parent1[i] for i in indices]
        shuffled_parent2 = [self.parent2[i] for i in indices]
        crossover_point = random.randint(1, len(self.parent1) - 1)
        offspring1 = shuffled_parent1[:crossover_point] + shuffled_parent2[crossover_point:]
        offspring2 = shuffled_parent2[:crossover_point] + shuffled_parent1[crossover_point:]
        unshuffle = {v: k for k, v in enumerate(indices)}
        unshuffledOffspring1 = [offspring1[unshuffle[i]] for i in range(len(offspring1))]
        unshuffledOffspring2 = [offspring2[unshuffle[i]] for i in range(len(offspring2))]
        return unshuffledOffspring1, unshuffledOffspring2 
    
    def reduced_surrogate_crossover(self):
        diff_indices = [i for i in range(len(self.parent1)) if self.parent1[i] != self.parent2[i]]
        if not diff_indices:
            return self.parent1, self.parent2
        crossover_point = random.choice(diff_indices)
        offspring1 = self.parent1[:crossover_point] + self.parent2[crossover_point:]
        offspring2 = self.parent2[:crossover_point] + self.parent1[crossover_point:]
        return offspring1, offspring2
    
    
    def uniform_crossover(self):
        mask = [random.randint(0, 1) for _ in range(len(self.parent1))]
        offspring1 = [self.parent1[i] if mask[i] == 0 else self.parent2[i] for i in range(len(self.parent1))]
        offspring2 = [self.parent2[i] if mask[i] == 0 else self.parent1[i] for i in range(len(self.parent2))]
        return offspring1, offspring2

    def twopoint_crossover(self):
        point1 = random.randint(1, len(self.parent1) - 2)
        point2 = random.randint(point1, len(self.parent1) - 1)
        child1 = self.parent1[:point1] + self.parent2[point1:point2] + self.parent1[point2:]
        child2 = self.parent2[:point1] + self.parent1[point1:point2] + self.parent2[point2:]
        return child1, child2
    


    def roulette_wheel_selection(self, crossover_probability):
        population = self.crossovers
        total_probability = sum(crossover_probability.values())
        cumulative_probabilities = [sum(crossover_probability[pop.__name__] for pop in population[:i+1]) / total_probability for i in range(len(population))]
        random_value = random.uniform(0, 1)
        selected_index = next(i for i, prob in enumerate(cumulative_probabilities) if prob >= random_value)
        selected_individual = population[selected_index]
        return selected_individual

    
    def execute_crossover(self, crossover):
        return crossover()





