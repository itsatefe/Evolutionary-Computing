
import numpy as np
from collections import Counter

class Chromosome:
    def __init__(self, values, objectives=None):
        self.values = values
        self.objectives = objectives or []
        self.domination_count = 0
        self.dominated_solutions = set()
        self.rank = None
        self.crowding_distance = 0

def dominate(s1, s2):
    return all(x <= y for x, y in zip(s1.objectives, s2.objectives)) and any(x < y for x, y in zip(s1.objectives, s2.objectives))

def crowding_distance(front):
    if not front:
        return
    num_objectives = len(front[0].objectives)
    for solution in front:
        solution.crowding_distance = 0
    for i in range(num_objectives):
        front.sort(key=lambda solution: solution.objectives[i])
        front[0].crowding_distance = front[-1].crowding_distance = float('inf')
        for j in range(1, len(front) - 1):
            front[j].crowding_distance += (front[j + 1].objectives[i] - front[j - 1].objectives[i])


def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = set()

        for q in population:
            if dominate(p, q):
                p.dominated_solutions.add(q)
            elif dominate(q, p):
                p.domination_count += 1

        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)

        i += 1
        fronts.append(next_front)
    return fronts


