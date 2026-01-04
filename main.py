import matplotlib.pyplot as plt
from distributions.exponential import exponential_pdf, exponential_cdf_piecewise, compute_equispaced_points_exp
from context.sampler_context import SamplerContext

# Dominio per Linear Search / Alias Method
n_pieces = 50
xs = [i*0.2 for i in range(n_pieces+1)]
cdf = exponential_cdf_piecewise(xs, lambd=1.0)
n_samples = 100_000

# Linear Search
sampler_linear = SamplerContext(xs, cdf, linear_threshold=100)
samples_linear = [sampler_linear.sample() for _ in range(n_samples)]

# Alias Method
sampler_alias = SamplerContext(xs, cdf, linear_threshold=0)
samples_alias = [sampler_alias.sample() for _ in range(n_samples)]

# Piecewise Equi-Spaced
xs_equi = compute_equispaced_points_exp(lambd=1.0, n_points=50)
sampler_equi = SamplerContext(xs, cdf, use_equispaced=True, n_equi=50)
sampler_equi.xs_equi = xs_equi  # passiamo i punti analitici
samples_equi = [sampler_equi.sample() for _ in range(n_samples)]

# Plot
plt.hist(samples_linear, bins=50, density=True, alpha=0.5, label="Linear Search")
plt.hist(samples_alias, bins=50, density=True, alpha=0.5, label="Alias Method")
plt.hist(samples_equi, bins=50, density=True, alpha=0.5, label="Piecewise Equi-Spaced")

xs_plot = [i*0.1 for i in range(101)]
plt.plot(xs_plot, [exponential_pdf(x) for x in xs_plot], "k--", label="Exponential PDF")
plt.legend()
plt.show()
