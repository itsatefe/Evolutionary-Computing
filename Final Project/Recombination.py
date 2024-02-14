
class Recombination:
    def __init__(self, parent1, parent2, crossover_probability=0.9):
        self.parent1 = parent1
        self.parent2 = parent2
        self.crossover_probability = crossover_probability
        self.crossovers = [self.sbx_crossover, self.pcx_crossover, self.spx_crossover, self.blx_alpha_crossover, self.de_rand_1_crossover]

    def sbx_crossover(self, eta=30):
        u = np.random.rand(len(self.parent1))
        beta = np.empty_like(u)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (eta + 1))
        offspring1 = 0.5 * ((1 + beta) * self.parent1 + (1 - beta) * self.parent2)
        offspring2 = 0.5 * ((1 - beta) * self.parent1 + (1 + beta) * self.parent2)
        return offspring1, offspring2


    def pcx_crossover(self, sigma=0.1, eta=0.1):
        center_of_mass = np.mean(parents, axis=0)
        parents = [self.parent1, self.parent2]
        target_parent = parents[np.random.randint(len(parents))]
        direction_vector = target_parent - center_of_mass
        orthogonal_component = np.random.randn(*target_parent.shape) * sigma
        offspring = target_parent + eta * direction_vector + orthogonal_component
        return offspring


    def spx_crossover(parents, epsilon=1.0):
        m, n = parents.shape
        center = np.mean(parents, axis=0)
        expanded_simplex = (1 + epsilon) * (parents - center)
        offspring = np.empty_like(parents)
        for i in range(m):
            random_weights = np.random.dirichlet(np.ones(m), size=1)
            offspring[i] = center + np.dot(random_weights, expanded_simplex)
        return offspring


    def blx_alpha_crossover(self, alpha=0.5):
        d = np.abs(self.parent1 - self.parent2)
        min_vals = np.minimum(self.parent1, self.parent2) - alpha * d
        max_vals = np.maximum(self.parent1, self.parent2) + alpha * d
        offspring = min_vals + np.random.rand(len(self.parent1)) * (max_vals - min_vals)
        return offspring


    def de_rand_1_crossover(target, a, b, c, cr=0.5):
        size = len(target)
        jrand = np.random.randint(size)
        offspring = np.array(target)
        for j in range(size):
            if np.random.rand() < cr or j == jrand:
                offspring[j] = a[j] + 0.5 * (b[j] - c[j])
        return offspring

    
    # implement crossver selection
    
    
    
    
    
    def execute_crossover(self, crossover):
        if np.random.rand() > self.crossover_probability:
            return self.parent1, self.parent2
        return crossover()
