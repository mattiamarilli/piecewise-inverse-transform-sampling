import random
import numpy as np

class AliasTable:
    def __init__(self, probs):
        n = len(probs)

        scaled = [p * n for p in probs]

        self.prob = [0.0] * n
        self.alias = [0] * n

        small, large = [], []

        for i, p in enumerate(scaled):
            (small if p < 1 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()

            self.prob[s] = scaled[s]
            self.alias[s] = l

            scaled[l] = scaled[l] - (1 - scaled[s])

            (small if scaled[l] < 1 else large).append(l)

        for i in small + large:
            self.prob[i] = 1.0

    def sample(self):
        i = random.randrange(len(self.prob))
        return i if random.random() < self.prob[i] else self.alias[i]

class ITSAlias():
    tag = "ITSAlias"

    def __init__(self, xs, cdf):
        self.xs = xs
        self.cdf = cdf

        segment_probs = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]

        self.alias = AliasTable(segment_probs)

    def sample(self):
        i = self.alias.sample()
        u = random.random()

        return self.xs[i] + u * (self.xs[i + 1] - self.xs[i])
    
    def draw(self, N):
        samples = np.empty(N)
        for i in range(N):
            samples[i] = self.sample()
        return samples
