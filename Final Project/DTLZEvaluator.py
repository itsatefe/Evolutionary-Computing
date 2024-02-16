import numpy as np
from pymoo.problems import get_problem

class DTLZEvaluator:
    def __init__(self, problem_name, n_var, M):
        self.n_var = n_var
        self.M = M
        self.problem_name = problem_name
      
    def evaluate(self, x):
        if hasattr(self, self.problem_name):
            return getattr(self, self.problem_name)(x)
        else:
            raise ValueError(f"Problem {self.problem_name} is not defined.")
            
    def get_bounds(self):
        return [(0, 1) for _ in range(self.n_var)]
    def dtlz1(self, x):
        problem = get_problem("dtlz1", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz2(self, x):
        problem = get_problem("dtlz2", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz3(self, x):
        problem = get_problem("dtlz3", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz4(self, x):
        problem = get_problem("dtlz4", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz5(self, x):
        problem = get_problem("dtlz5", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz6(self, x):
        problem = get_problem("dtlz6", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def dtlz7(self, x):
        problem = get_problem("dtlz7", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)
