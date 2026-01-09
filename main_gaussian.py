import time
import matplotlib.pyplot as plt

from strategies.linear_search import PiecewiseLinearCDFLinearSearch
from strategies.alias_method import PiecewiseLinearCDFAlias
from distributions.gaussian import gaussian_pdf, gaussian_cdf_piecewise

if __name__ == "__main__":
    # -----------------------------
    # Setup Gaussian distribution
    # -----------------------------
    # Discretize the x-domain for CDF evaluation
    n_pieces = 500
    xs = [-5 + 10 * i / n_pieces for i in range(n_pieces + 1)]  # From -5 to 5

    # Compute a piecewise approximation of the CDF
    cdf = gaussian_cdf_piecewise(xs)

    # Number of samples to generate for each method
    n_samples = 1_000_000

    # -----------------------------
    # LINEAR SEARCH SAMPLING
    # -----------------------------
    # Initialize the sampler using linear search over the CDF
    linear_sampler = PiecewiseLinearCDFLinearSearch(xs, cdf)

    # Measure sampling time
    start = time.time()
    samples_linear = [linear_sampler.sample() for _ in range(n_samples)]
    end = time.time()

    linear_time = end - start
    print(f"Linear Search sampling time: {linear_time:.4f}s")

    # -----------------------------
    # ALIAS METHOD SAMPLING
    # -----------------------------
    # Initialize the sampler using the Alias Method (O(1) per sample)
    alias_sampler = PiecewiseLinearCDFAlias(xs, cdf)

    start = time.time()
    samples_alias = [alias_sampler.sample() for _ in range(n_samples)]
    end = time.time()

    alias_time = end - start
    print(f"Alias Method sampling time: {alias_time:.4f}s")

    # -----------------------------
    # Speed comparison
    # -----------------------------
    # Compute speedup relative to Linear Search
    print(f"Speedup (Linear / Alias): {linear_time / alias_time:.2f}x")

    # -----------------------------
    # Optional plot: compare samples with the true PDF
    # -----------------------------
    # Evaluate the theoretical Gaussian PDF
    pdf_real = [gaussian_pdf(x) for x in xs]

    # -----------------------------
    # Plot Linear Search samples
    # -----------------------------
    plt.figure(figsize=(12, 6))
    plt.hist(samples_linear, bins=500, density=True, alpha=0.7, color='blue', label="Linear Search samples")
    plt.plot(xs, pdf_real, "k--", label="Gaussian PDF")
    plt.title("Sampling from Gaussian Distribution (Linear Search)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # -----------------------------
    # Plot Alias Method samples
    # -----------------------------
    plt.figure(figsize=(12, 6))
    plt.hist(samples_alias, bins=500, density=True, alpha=0.7, color='green', label="Alias Method samples")
    plt.plot(xs, pdf_real, "k--", label="Gaussian PDF")
    plt.title("Sampling from Gaussian Distribution (Alias Method)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


