from strategies.linear_search import PiecewiseLinearCDFLinearSearch
from strategies.alias_method import PiecewiseLinearCDFAlias
from strategies.equispaced import PiecewiseEquispacedCDF
from utils.cdf_utils import compute_equispaced_points

class SamplerContext:
    def __init__(self, xs, cdf, linear_threshold=100, use_equispaced=False, n_equi=100):
        self.xs = xs
        self.cdf = cdf
        self.linear_threshold = linear_threshold
        self.use_equispaced = use_equispaced
        self.n_equi = n_equi
        self.strategy = self._select_strategy()

    def _select_strategy(self):
        if self.use_equispaced:
            # Se CDF invertibile analiticamente (esponenziale)
            if hasattr(self, "xs_equi"):
                xs_equi = self.xs_equi
            else:
                xs_equi = compute_equispaced_points(self.xs, self.cdf, self.n_equi)
            return PiecewiseEquispacedCDF(xs_equi)

        n_segments = len(self.xs) - 1
        if n_segments <= self.linear_threshold:
            return PiecewiseLinearCDFLinearSearch(self.xs, self.cdf)
        return PiecewiseLinearCDFAlias(self.xs, self.cdf)

    def read_strategy(self):
        return self.strategy.tag

    def sample(self):
        return self.strategy.sample()
