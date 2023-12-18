import numpy as np

class ObjectiveFunction:
    def __init__(self, name, dimension, range_x):
        self.name = name
        self.dimension = dimension
        self.range_x = range_x

    def evaluate(self, x):
        if self.name == "schwefel_function":
            return self._evaluate_schwefel(x)
        elif self.name == "ackley_function":
            return self._evaluate_ackley(x)
        
    def evaluate_list(self, population):
        return [self.evaluate(np.array(chromosome)) for chromosome in population]

    def _evaluate_schwefel(self, x):
        a = 418.9829
        sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return a * self.dimension - sum_term

    def _evaluate_ackley(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt((1/self.dimension) * np.sum(x**2)))
        term2 = -np.exp((1/self.dimension) * np.sum(np.cos(c * x)))
        return term1 + term2 + 20 + np.exp(1)

