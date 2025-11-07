from __future__ import annotations
from typing import List
import random
from .diversity import bpd

class Population:
    def __init__(self, min_size=25, gen_size=40, elites=4, num_close=5):
        self.min_size = min_size
        self.gen_size = gen_size
        self.elites = elites
        self.num_close = num_close
        self.individuals = []  # list de Solution

    def add(self, sol):
        self.individuals.append(sol)

    def select_parents(self):
        # tournoi simple: 1 élite + 1 divers
        A = min(self.individuals, key=lambda s: s.cost)
        cand = random.sample(self.individuals, k=min(8, len(self.individuals)))
        cand.sort(key=lambda s: bpd(A, s), reverse=True)
        B = cand[0] if cand else A
        return A, B

    def survivor_selection(self):
        # garder élites + diversité basique
        self.individuals.sort(key=lambda s: s.cost)
        keep = self.individuals[:self.elites]
        rest = self.individuals[self.elites:]
        while len(keep) < self.min_size and rest:
            keep.append(rest.pop(0))
        self.individuals = keep
