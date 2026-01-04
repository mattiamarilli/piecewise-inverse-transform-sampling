import random
from .base import SamplingStrategy

class PiecewiseLinearCDFLinearSearch(SamplingStrategy):
    def __init__(self, xs, cdf):
        self.xs = xs
        self.cdf = cdf
        self.tag = "PiecewiseLinearCDFLinearSearch"

    def sample(self):
        u = random.random()
        for i in range(len(self.cdf) - 1):
            if self.cdf[i] <= u <= self.cdf[i + 1]:
                t = (u - self.cdf[i]) / (self.cdf[i + 1] - self.cdf[i])
                return self.xs[i] + t * (self.xs[i + 1] - self.xs[i])
        return self.xs[-1]
