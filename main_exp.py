import time
import matplotlib.pyplot as plt

from distributions.exponential import (
    exponential_pdf,
    exponential_cdf_piecewise,
    compute_equispaced_points_exp
)

from strategies.linear_search import PiecewiseLinearCDFLinearSearch
from strategies.alias_method import PiecewiseLinearCDFAlias
from strategies.equispaced import PiecewiseEquispacedCDF

# -----------------------------
# Domain and parameters
# -----------------------------
# Number of points to discretize the CDF
n_pieces = 10_000
xs = [i * 0.2 for i in range(n_pieces + 1)]

# Compute the CDF values corresponding to xs
cdf = exponential_cdf_piecewise(xs, lambd=1.0)

# Number of samples to generate for each sampling strategy
n_samples = 1_000_000

# -----------------------------
# Linear Search Sampling
# -----------------------------
# Initialize the sampler using linear search
sampler_linear = PiecewiseLinearCDFLinearSearch(xs, cdf)

# Measure the sampling time
start = time.perf_counter()
samples_linear = [sampler_linear.sample() for _ in range(n_samples)]
end = time.perf_counter()
linear_time = end - start
print(f"Linear Search time: {linear_time:.4f} s")

# -----------------------------
# Alias Method Sampling
# -----------------------------
# Initialize the sampler using the Alias Method (O(1) per sample)
sampler_alias = PiecewiseLinearCDFAlias(xs, cdf)

start = time.perf_counter()
samples_alias = [sampler_alias.sample() for _ in range(n_samples)]
end = time.perf_counter()
alias_time = end - start
print(f"Alias Method time: {alias_time:.4f} s")

# -----------------------------
# Piecewise Equi-Spaced Sampling
# -----------------------------
# Generate points equispaced in probability space using the inverse CDF
xs_equi = compute_equispaced_points_exp(lambd=1.0, n_points=50)
sampler_equi = PiecewiseEquispacedCDF(xs_equi)

start = time.perf_counter()
samples_equi = [sampler_equi.sample() for _ in range(n_samples)]
end = time.perf_counter()
equi_time = end - start
print(f"Piecewise Equi-Spaced time: {equi_time:.4f} s")

# -----------------------------
# Speed comparison
# -----------------------------
# Print speedup relative to Linear Search
print(f"Speedup Linear/Alias: {linear_time / alias_time:.2f}x")
print(f"Speedup Linear/Equi:  {linear_time / equi_time:.2f}x")

# -----------------------------
# Plot distributions
# -----------------------------
plt.figure(figsize=(12, 6))

# Plot normalized histograms of samples for comparison
plt.hist(samples_linear, bins=500, density=True, alpha=0.5, label="Linear Search")
plt.hist(samples_alias, bins=500, density=True, alpha=0.5, label="Alias Method")
plt.hist(samples_equi, bins=500, density=True, alpha=0.5, label="Piecewise Equi-Spaced")

# Plot theoretical exponential PDF for reference
xs_plot = [i * 0.1 for i in range(101)]
plt.plot(xs_plot, [exponential_pdf(x) for x in xs_plot], "k--", label="Exponential PDF")

plt.legend()
plt.title("Sampling Comparison: Exponential Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
