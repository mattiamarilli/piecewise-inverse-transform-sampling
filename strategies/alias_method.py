import random
from .base import SamplingStrategy
from utils.alias_table import AliasTable

class PiecewiseLinearCDFAlias(SamplingStrategy):
    tag = "PiecewiseLinearCDFAlias"

    def __init__(self, xs, cdf):
        self.xs = xs
        self.cdf = cdf

        segment_probs = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]
        self.alias = AliasTable(segment_probs)

    def sample(self):
        i = self.alias.sample()
        u = random.random()
        return self.xs[i] + u * (self.xs[i + 1] - self.xs[i])
