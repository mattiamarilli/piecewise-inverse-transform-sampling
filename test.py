import random
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ==========================
# INTERFACCIA STRATEGY
# ==========================
class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass

# ==========================
# PIECEWISE LINEAR CDF - RICERCA LINEARE (O(n))
# ==========================
class PiecewiseLinearCDFLinearSearch(SamplingStrategy):
    def __init__(self, xs, cdf):
        self.xs = xs
        self.cdf = cdf

    def sample(self):
        u = random.random()
        for i in range(len(self.cdf) - 1):
            if self.cdf[i] <= u <= self.cdf[i + 1]:
                t = (u - self.cdf[i]) / (self.cdf[i + 1] - self.cdf[i])
                return self.xs[i] + t * (self.xs[i + 1] - self.xs[i])
        return self.xs[-1]

# ==========================
# ALIAS TABLE
# ==========================
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

# ==========================
# PIECEWISE LINEAR CDF - ALIAS METHOD (O(1))
# ==========================
class PiecewiseLinearCDFAlias(SamplingStrategy):
    def __init__(self, xs, cdf):
        self.xs = xs
        self.cdf = cdf
        segment_probs = [cdf[i + 1] - cdf[i] for i in range(len(cdf) - 1)]
        self.alias = AliasTable(segment_probs)

    def sample(self):
        i = self.alias.sample()
        u = random.random()
        return self.xs[i] + u * (self.xs[i + 1] - self.xs[i])

# ==========================
# CREAZIONE CDF GAUSSIANA PIECEWISE
# ==========================
def gaussian_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def gaussian_cdf_piecewise(xs):
    cdf = [0.0]
    for i in range(1, len(xs)):
        # approccio semplice: integrazione approssimata con rettangoli
        dx = xs[i] - xs[i-1]
        pdf_val = (gaussian_pdf(xs[i]) + gaussian_pdf(xs[i-1])) / 2
        cdf.append(cdf[-1] + pdf_val * dx)
    # normalizzazione finale
    total = cdf[-1]
    cdf = [v / total for v in cdf]
    return cdf

# ==========================
# ESEMPIO DI UTILIZZO + PLOT
# ==========================
if __name__ == "__main__":
    # dominio della gaussiana
    xs = [i * 0.1 for i in range(-50, 51)]  # da -5 a 5
    cdf = gaussian_cdf_piecewise(xs)

    # inizializzazione sampler
    sampler_linear = PiecewiseLinearCDFLinearSearch(xs, cdf)
    sampler_alias = PiecewiseLinearCDFAlias(xs, cdf)

    n_samples = 10000
    samples_linear = [sampler_linear.sample() for _ in range(n_samples)]
    samples_alias = [sampler_alias.sample() for _ in range(n_samples)]

    # PDF gaussiana reale
    pdf_real = [gaussian_pdf(x) for x in xs]
    # normalizzazione per densità istogramma
    dx = xs[1] - xs[0]
    pdf_real_norm = [p for p in pdf_real]

    # PLOT
    plt.figure(figsize=(12, 6))
    plt.hist(samples_linear, bins=50, density=True, alpha=0.5, label="Linear Search")
    plt.plot(xs, pdf_real_norm, 'k--', label="Gaussian PDF")
    plt.title("Sampling strategies comparison for Gaussian")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # PLOT
    plt.figure(figsize=(12, 6))
    plt.hist(samples_alias, bins=50, density=True, alpha=0.5, label="Alias Method")
    plt.plot(xs, pdf_real_norm, 'k--', label="Gaussian PDF")
    plt.title("Sampling strategies comparison for Gaussian")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
