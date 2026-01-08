import random
from strategies.base import SamplingStrategy

class PiecewiseEquispacedCDF(SamplingStrategy):
    tag = "piecewise_equispaced"

    def __init__(self, xs_equi):
        self.xs = xs_equi
        self.n = len(xs_equi) - 1

    def sample(self):
        u = random.random()
        s = u * self.n
        i = int(s)
        if i == self.n:
            i = self.n - 1
            t = 1.0
        else:
            t = s - i

        return self.xs[i] + t * (self.xs[i+1] - self.xs[i])

