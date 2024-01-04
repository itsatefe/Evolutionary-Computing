import numpy as np

class Chromosome:
    def __init__(self, values, objectives=None):
        self.values = values
        self.objectives = objectives or []
        self.domination_count = 0
        self.dominated_solutions = set()
        self.rank = None
        self.crowding_distance = 0

    def dominate(self, other):
        return all(x <= y for x, y in zip(self.objectives, other.objectives)) and any(x < y for x, y in zip(self.objectives, other.objectives))
