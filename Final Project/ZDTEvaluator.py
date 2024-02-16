import numpy as np
from pymoo.problems import get_problem

class ZDTEvaluator:
    def __init__(self, problem_name, n_var, M=2):  # ZDT problems are bi-objective
        self.n_var = n_var
        self.M = M  # Default is 2 because ZDT problems are typically bi-objective
        self.problem_name = problem_name
    
    def get_bounds(self):
        problem = get_problem(self.problem_name, n_var=self.n_var)
        lower_bounds = problem.xl
        upper_bounds = problem.xu
        return lower_bounds, upper_bounds

    
    def ideal_point(self):
        problem = get_problem(self.problem_name, n_var=self.n_var)
        return problem.ideal_point()
    
    def nadir_point(self):
        problem = get_problem(self.problem_name, n_var=self.n_var)
        return problem.nadir_point()

    def evaluate(self, x):
        if hasattr(self, self.problem_name):
            return getattr(self, self.problem_name)(x)
        else:
            raise ValueError(f"Problem {self.problem_name} is not defined.")

    def zdt1(self, x):
        problem = get_problem("zdt1", n_var=self.n_var)
        return problem.evaluate(x)

    def zdt2(self, x):
        problem = get_problem("zdt2", n_var=self.n_var)
        return problem.evaluate(x)

    def zdt3(self, x):
        problem = get_problem("zdt3", n_var=self.n_var)
        return problem.evaluate(x)

    def zdt4(self, x):
        problem = get_problem("zdt4", n_var=self.n_var)
        return problem.evaluate(x)

    def zdt6(self, x):
        problem = get_problem("zdt6", n_var=self.n_var)
        return problem.evaluate(x)