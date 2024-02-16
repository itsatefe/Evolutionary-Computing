import numpy as np
import random

class Recombination:
    def __init__(self, parents, parameters, crossover_probability=0.9):
        self.parents = parents
        self.crossover_probability = crossover_probability
        self.crossovers = {'sbx': self.sbx, 'pcx': self.pcx, 'spx': self.spx, 
                           'blx_alpha': self.blx_alpha, 'de_rand_1': self.de_rand_1}
        self.parameters = parameters

    def sbx(self, **kwargs):
        eta = kwargs.get('eta', 30)
        parent1, parent2 = self.parents[0], self.parents[1]
        u = np.random.rand(len(parent1))
        beta = np.empty_like(u)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (eta + 1))
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        offsprings = [offspring1, offspring2]
        return offsprings


    def pcx(self, **kwargs):
        sigma = kwargs.get('sigma', 0.1)
        eta = kwargs.get('eta', 0.1)
        parents = self.parents[:2]
        center_of_mass = np.mean(parents, axis=0)
        target_parent = parents[np.random.randint(len(parents))]
        direction_vector = target_parent - center_of_mass
        orthogonal_component = np.random.randn(*target_parent.shape) * sigma
        offspring = target_parent + eta * direction_vector + orthogonal_component
        return [offspring]


    def spx(self, **kwargs):
        epsilon = kwargs.get('epsilon', 1.0)
        parents = self.parents[:2]
        m, n = parents.shape
        center = np.mean(parents, axis=0)
        expanded_simplex = (1 + epsilon) * (parents - center)
        offsprings = np.empty_like(parents)
        for i in range(m):
            random_weights = np.random.dirichlet(np.ones(m), size=1)
            offsprings[i] = center + np.dot(random_weights, expanded_simplex)
        return offsprings


    def blx_alpha(self, **kwargs):
        alpha = kwargs.get('alpha', 0.5)
        parent1, parent2 = self.parents[0], self.parents[1]
        d = np.abs(parent1 - parent2)
        min_vals = np.minimum(parent1, parent2) - alpha * d
        max_vals = np.maximum(parent1, parent2) + alpha * d
        offspring = min_vals + np.random.rand(len(parent1)) * (max_vals - min_vals)
        return [offspring]


    def de_rand_1(self, **kwargs):
        if len(self.parents) < 4:
            return self.parents
        cr = kwargs.get('cr', 1)
        f = kwargs.get('f', 0.5)
        target, a, b, c = random.sample(list(self.parents), 4)
        size = len(target)
        jrand = np.random.randint(size)
        offspring = np.array(target)
        for j in range(size):
            if np.random.rand() < cr or j == jrand:
                offspring[j] = a[j] + f * (b[j] - c[j])
        return [offspring]
            
    
    def select_crossover_operator(self, operator_probabilities):
        operators, probabilities = zip(*operator_probabilities)
        selected_operator = random.choices(operators, weights=probabilities, k=1)[0]
        return selected_operator
    
    def execute_crossover(self, crossover_name):
        if np.random.rand() > self.crossover_probability:
            parent1, parent2 = self.parents[0], self.parents[1]
            return parent1, parent2
        crossover = self.crossovers[crossover_name]
        return crossover(**self.parameters[crossover_name])
