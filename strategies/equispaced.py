import random
from strategies.base import SamplingStrategy


class PiecewiseEquispacedCDF(SamplingStrategy):
    """
    Sampling strategy based on a piecewise-linear approximation
    of the inverse CDF.

    The input array `xs` represents points that are equispaced in
    probability space, i.e.:

        F(x_i) ≈ i / n ,   i = 0, ..., n

    Sampling is performed by drawing U ~ Uniform(0, 1) and
    linearly interpolating between the corresponding quantile
    points x_i and x_{i+1}.

    This method is an application of the inverse transform sampling
    technique using a discretized CDF.
    """

    # Identifier for the strategy (useful for configuration or logging)
    tag = "piecewise_equispaced"

    def __init__(self, xs_equi):
        """
        Initialize the sampler with equispaced CDF points.

        Parameters:
            xs_equi (list of float): Quantile points corresponding to
                                     uniformly spaced CDF values.
        """
        # Quantile values in the x-domain
        self.xs = xs_equi

        # Number of intervals in the piecewise approximation
        self.n = len(xs_equi) - 1

    def sample(self):
        """
        Generate a single sample using piecewise-linear interpolation.

        Algorithm:
        1. Draw u ~ Uniform(0, 1)
        2. Map u to an interval index i such that u ∈ [i/n, (i+1)/n)
        3. Linearly interpolate between x_i and x_{i+1}

        Returns:
            float: A sample approximately distributed according
                   to the target distribution.
        """
        # Uniform random variable in [0, 1)
        u = random.random()

        # Scale to the number of intervals
        s = u * self.n

        # Interval index
        i = int(s)

        # Handle the edge case u = 1 (numerical safety)
        if i == self.n:
            i = self.n - 1
            t = 1.0
        else:
            # Local coordinate inside the interval [0, 1)
            t = s - i

        # Linear interpolation:
        # x = (1 − t) x_i + t x_{i+1}
        return self.xs[i] + t * (self.xs[i + 1] - self.xs[i])
