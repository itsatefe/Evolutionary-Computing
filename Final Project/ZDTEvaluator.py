#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pymoo.problems import get_problem

class ZDTEvaluator:
    def __init__(self, n_var, M=2):  # ZDT problems are bi-objective
        self.n_var = n_var
        self.M = M  # Default is 2 because ZDT problems are typically bi-objective

    def evaluate(self, x, problem_name):
        if hasattr(self, problem_name):
            return getattr(self, problem_name)(x)
        else:
            raise ValueError(f"Problem {problem_name} is not defined.")

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

n_var = 30  # Number of decision variables, commonly used setting for ZDT
evaluator = ZDTEvaluator(n_var)

