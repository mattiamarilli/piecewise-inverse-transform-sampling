from strategies.linear_search import PiecewiseLinearCDFLinearSearch
from strategies.alias_method import PiecewiseLinearCDFAlias

class SamplerContext:
    """
    Context dello Strategy Pattern.
    Decide quale strategia usare e delega il sampling.
    """

    def __init__(self, xs, cdf, linear_threshold=100):
        self.xs = xs
        self.cdf = cdf
        self.linear_threshold = linear_threshold
        self.strategy = self._select_strategy()

    def _select_strategy(self):
        n_segments = len(self.xs) - 1
        if n_segments <= self.linear_threshold:
            print()
            return PiecewiseLinearCDFLinearSearch(self.xs, self.cdf)
        return PiecewiseLinearCDFAlias(self.xs, self.cdf)

    def read_strategy(self):
        return self.strategy.tag

    def sample(self):
        return self.strategy.sample()
