import time
import matplotlib.pyplot as plt

from strategies.linear_search import PiecewiseLinearCDFLinearSearch
from strategies.alias_method import PiecewiseLinearCDFAlias
from distributions.gaussian import gaussian_pdf, gaussian_cdf_piecewise

if __name__ == "__main__":
    # -----------------------------
    # Setup distribuzione
    # -----------------------------
    n_pieces = 500
    xs = [-5 + 10 * i / n_pieces for i in range(n_pieces + 1)]
    cdf = gaussian_cdf_piecewise(xs)

    n_samples = 1_000_000

    # -----------------------------
    # LINEAR SEARCH
    # -----------------------------
    linear_sampler = PiecewiseLinearCDFLinearSearch(xs, cdf)

    start = time.time()
    samples_linear = [linear_sampler.sample() for _ in range(n_samples)]
    end = time.time()

    linear_time = end - start
    print(f"Linear Search sampling time: {linear_time:.4f}s")

    # -----------------------------
    # ALIAS METHOD
    # -----------------------------
    alias_sampler = PiecewiseLinearCDFAlias(xs, cdf)

    start = time.time()
    samples_alias = [alias_sampler.sample() for _ in range(n_samples)]
    end = time.time()

    alias_time = end - start
    print(f"Alias Method sampling time: {alias_time:.4f}s")

    # -----------------------------
    # Speedup
    # -----------------------------
    print(f"Speedup (Linear / Alias): {linear_time / alias_time:.2f}x")

    # -----------------------------
    # Plot (opzionale, usa uno solo)
    # -----------------------------
    pdf_real = [gaussian_pdf(x) for x in xs]

    plt.figure(figsize=(12, 6))
    plt.hist(samples_alias, bins=500, density=True, alpha=0.5, label="Alias samples")
    plt.plot(xs, pdf_real, "k--", label="Gaussian PDF")
    plt.legend()
    plt.show()
