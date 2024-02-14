#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class Subspace:
    def __init__(self, coordinates, ideal_point, grid_intervals):
        self.coordinates = coordinates  # Coordinates of the subspace in the grid
        self.ideal_point = ideal_point  # Ideal point in the objective space
        self.grid_intervals = grid_intervals  # Grid intervals in each objective dimension
        self.solutions = []  # List to store solutions that belong to this subspace

    def calculate_scp(self):
        z_scp = self.ideal_point + self.grid_intervals * self.coordinates
        return z_scp

    def calculate_sdv(self, epsilon=1e-6):
        sdv = 1.0 / (self.coordinates + epsilon)
        return sdv

    def calculate_sws(self, x, epsilon=1e-6):
        sdv = self.calculate_sdv(epsilon)
        z_scp = self.calculate_scp()
        sws_value = np.sum(sdv * (x - z_scp))
        return sws_value

    def subspace_dominance(self, other):
        dominates = False
        for a, b in zip(self.coordinates, other.coordinates):
            if a > b:
                return False
            if a < b:
                dominates = True
        return dominates

    def strong_subspace_dominance(self, other):
        return all(a < b for a, b in zip(self.coordinates, other.coordinates))

    def weak_subspace_dominance(self, other):
        equal_in_at_least_one_dimension = False
        for a, b in zip(self.coordinates, other.coordinates):
            if a > b:
                return False
            if a == b:
                equal_in_at_least_one_dimension = True
        return equal_in_at_least_one_dimension
    
    def select_representative(self):
        if not self.solutions:
            return None
        solutions_sws = [self.calculate_sws(x) for x in self.solutions]
        min_index = np.argmin(solutions_sws)
        return self.solutions[min_index]

