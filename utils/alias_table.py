import random

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
