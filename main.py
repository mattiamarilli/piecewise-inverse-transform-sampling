import time
import matplotlib.pyplot as plt

from context.sampler_context import SamplerContext
from distributions.gaussian import gaussian_pdf, gaussian_cdf_piecewise

if __name__ == "__main__":
    n_pieces = 100
    xs = [-5 + 10 * i / n_pieces for i in range(n_pieces + 1)]
    cdf = gaussian_cdf_piecewise(xs)

    sampler = SamplerContext(xs, cdf)

    print("Sampling strategy:", sampler.read_strategy())

    n_samples = 1_000_000

    start = time.time()
    samples = [sampler.sample() for _ in range(n_samples)]
    end = time.time()

    print(f"Tempo sampling: {end - start:.4f}s")
    print(f"Strategia usata: {sampler.strategy.__class__.__name__}")

    pdf_real = [gaussian_pdf(x) for x in xs]

    plt.figure(figsize=(12, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples")
    plt.plot(xs, pdf_real, "k--", label="Gaussian PDF")
    plt.legend()
    plt.show()
