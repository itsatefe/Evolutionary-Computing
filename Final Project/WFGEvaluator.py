import numpy as np
from pymoo.problems import get_problem

class WFGEvaluator:
    def __init__(self, problem_name, n_var, M, k=None):
        self.n_var = n_var
        self.M = M
        self.k = k if k is not None else (self.n_var // 2)  # Set k if not provided, common default is n_var / 2
        self.problem_name = problem_name
  
    def evaluate(self, x):
        if hasattr(self, self.problem_name):
            return getattr(self, self.problem_name)(x)
        else:
            raise ValueError(f"Problem {self.problem_name} is not defined.")

    def get_bounds(self):
        # For WFG problems, the bounds for the first k variables (position parameters) are 0 <= x_i <= 2i
        # For the remaining variables (distance parameters), the bounds are 0 <= x_i <= 1
        problem = get_problem(self.problem_name, n_var=self.n_var, n_obj=self.M)
        lower_bounds = problem.xl
        upper_bounds = problem.xu
        
#         position_bounds = [(0, 2 * (i+1)) for i in range(self.k)]
#         distance_bounds = [(0, 1) for _ in range(self.n_var - self.k)]
#         return position_bounds + distance_bounds
        return lower_bounds, upper_bounds
    
    def ideal_point(self):
        problem = get_problem(self.problem_name, n_var=self.n_var, n_obj=self.M)
        return problem.ideal_point()
    
    def nadir_point(self):
        problem = get_problem(self.problem_name, n_var=self.n_var, n_obj=self.M)
        return problem.nadir_point()
    
    def wfg1(self, x):
        problem = get_problem("wfg1", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg2(self, x):
        problem = get_problem("wfg2", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg3(self, x):
        problem = get_problem("wfg3", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg4(self, x):
        problem = get_problem("wfg4", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg5(self, x):
        problem = get_problem("wfg5", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg6(self, x):
        problem = get_problem("wfg6", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg7(self, x):
        problem = get_problem("wfg7", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg8(self, x):
        problem = get_problem("wfg8", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)

    def wfg9(self, x):
        problem = get_problem("wfg9", n_var=self.n_var, n_obj=self.M)
        return problem.evaluate(x)


