import random
from strategies.base import SamplingStrategy

class PiecewiseEquispacedCDF(SamplingStrategy):
    tag = "piecewise_equispaced"

    def __init__(self, xs_equi):
        self.xs = xs_equi
        self.n = len(xs_equi) - 1

    def sample(self):
        i = random.randrange(self.n)
        u = random.random()
        return self.xs[i] + u * (self.xs[i+1] - self.xs[i])
