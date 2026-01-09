import random


class AliasTable:
    """
    Implementation of the Alias Method for efficient discrete sampling.

    Given a discrete probability distribution with probabilities p_i,
    i = 0, ..., n−1, this method allows sampling in O(1) time after
    an O(n) preprocessing step.

    The idea is to represent each probability as a mixture of at most
    two outcomes, enabling constant-time sampling using a single
    uniform random variable.
    """

    def __init__(self, probs):
        """
        Build the alias table from a list of probabilities.

        Parameters:
            probs (list of float): Discrete probability distribution,
                                   with p_i ≥ 0 and ∑ p_i = 1.
        """
        n = len(probs)

        # Scale probabilities so that their average is 1
        scaled = [p * n for p in probs]

        # Probability and alias tables
        self.prob = [0.0] * n
        self.alias = [0] * n

        # Indices with scaled probability < 1 and ≥ 1
        small, large = [], []

        for i, p in enumerate(scaled):
            # Partition indices by whether they underfill or overfill a bin
            (small if p < 1 else large).append(i)

        # Construct the alias table
        while small and large:
            s = small.pop()
            l = large.pop()

            # s gets probability scaled[s] and aliases to l
            self.prob[s] = scaled[s]
            self.alias[s] = l

            # Transfer excess probability from l to s
            scaled[l] = scaled[l] - (1 - scaled[s])

            # Reassign l to the appropriate list
            (small if scaled[l] < 1 else large).append(l)

        # Remaining bins have probability 1
        for i in small + large:
            self.prob[i] = 1.0

    def sample(self):
        """
        Draw a sample from the discrete distribution.

        Algorithm:
        1. Choose an index i uniformly from {0, ..., n−1}
        2. Draw u ~ Uniform(0, 1)
        3. Return i if u < prob[i], otherwise return alias[i]

        Returns:
            int: Sampled index according to the original distribution.
        """
        # Uniformly choose a column
        i = random.randrange(len(self.prob))

        # Biased coin flip to select between i and its alias
        return i if random.random() < self.prob[i] else self.alias[i]
