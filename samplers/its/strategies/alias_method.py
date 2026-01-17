import random
from .base import SamplingStrategy
from utils.alias_table import AliasTable

class PiecewiseLinearCDFAlias(SamplingStrategy):
    """
    Sampling strategy based on a piecewise-linear CDF using the Alias Method.

    The CDF is discretized into intervals [x_i, x_{i+1}] with probability mass:

        p_i = F(x_{i+1}) − F(x_i)

    The Alias Method is used to sample an interval index i in O(1) time.
    A uniform sample is then drawn inside the selected interval, producing
    an approximate inverse CDF sample.

    This approach combines inverse transform sampling with efficient
    discrete sampling.
    """

    # Identifier for the strategy
    tag = "PiecewiseLinearCDFAlias"

    def __init__(self, xs, cdf):
        """
        Initialize the sampler with discretized CDF data.

        Parameters:
            xs (list of float): Monotonically increasing x-values.
            cdf (list of float): Corresponding CDF values, with
                                 0 ≤ cdf[i] ≤ 1 and non-decreasing.
        """
        self.xs = xs
        self.cdf = cdf

        # Probability mass of each interval:
        # p_i = F(x_{i+1}) − F(x_i)
        segment_probs = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]

        # Build the alias table for O(1) interval sampling
        self.alias = AliasTable(segment_probs)

    def sample(self):
        """
        Generate a single sample using the Alias Method and interpolation.

        Algorithm:
        1. Sample an interval index i according to p_i using the Alias Method
        2. Draw u ~ Uniform(0, 1)
        3. Sample uniformly in the interval [x_i, x_{i+1}]

        Returns:
            float: A sample approximately distributed according
                   to the target distribution.
        """
        # Sample interval index according to the discretized PDF
        i = self.alias.sample()

        # Uniform position inside the selected interval
        u = random.random()

        # Linear interpolation inside the interval
        return self.xs[i] + u * (self.xs[i + 1] - self.xs[i])
