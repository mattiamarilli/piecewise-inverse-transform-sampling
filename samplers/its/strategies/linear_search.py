import random
from .base import SamplingStrategy


class PiecewiseLinearCDFLinearSearch(SamplingStrategy):
    """
    Sampling strategy based on a piecewise-linear approximation
    of a generic cumulative distribution function (CDF).

    Given a discretized CDF defined by pairs (x_i, F(x_i)),
    this method draws a sample using inverse transform sampling
    by performing a linear search over the CDF values.

    Mathematically, for U ~ Uniform(0, 1), the algorithm finds
    the interval such that:

        F(x_i) ≤ U ≤ F(x_{i+1})

    and approximates the inverse CDF by linear interpolation.
    """

    # Identifier for the strategy
    tag = "PiecewiseLinearCDFLinearSearch"

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

    def sample(self):
        """
        Generate a single sample using linear search and interpolation.

        Algorithm:
        1. Draw u ~ Uniform(0, 1)
        2. Find i such that cdf[i] ≤ u ≤ cdf[i+1]
        3. Linearly interpolate between x_i and x_{i+1}

        This implementation uses a linear search, resulting in
        O(n) time complexity per sample.

        Returns:
            float: A sample approximately distributed according
                   to the target distribution.
        """
        # Uniform random variable in [0, 1)
        u = random.random()

        # Search for the CDF interval containing u
        for i in range(len(self.cdf) - 1):
            if self.cdf[i] <= u <= self.cdf[i + 1]:
                # Local coordinate in probability space
                t = (u - self.cdf[i]) / (self.cdf[i + 1] - self.cdf[i])

                # Linear interpolation in the x-domain
                return self.xs[i] + t * (self.xs[i + 1] - self.xs[i])

        # Fallback in case of numerical edge cases (u ≈ 1)
        return self.xs[-1]
