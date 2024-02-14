#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pymoo.problems import get_problem

class WFGEvaluator:
    def __init__(self, n_var, M):
        self.n_var = n_var
        self.M = M

    def evaluate(self, x, problem_name):
        if hasattr(self, problem_name):
            return getattr(self, problem_name)(x)
        else:
            raise ValueError(f"Problem {problem_name} is not defined.")

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


