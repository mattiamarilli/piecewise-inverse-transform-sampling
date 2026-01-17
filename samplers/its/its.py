import numpy as np

class PiecewiseITS:
    """
    Sampler basato su inverse transform sampling con CDF piecewise.

    La logica di campionamento è completamente delegata a una
    SamplingStrategy (linear search, alias method, equispaced, ecc.).
    """

    def __init__(self, strategy, lb=None, ub=None):
        """
        Parameters
        ----------
        strategy : SamplingStrategy
            Oggetto che implementa il metodo sample().
        lb : float, optional
            Lower bound del supporto (solo informativo / diagnostico).
        ub : float, optional
            Upper bound del supporto.
        """
        self.strategy = strategy
        self.lb = lb
        self.ub = ub

    def sample(self):
        """
        Estrae un singolo campione delegando alla strategia.
        """
        return self.strategy.sample()

    def draw(self, N):
        """
        Estrae N campioni.

        Returns
        -------
        np.ndarray
        """
        samples = np.empty(N)
        for i in range(N):
            samples[i] = self.sample()
        return samples
