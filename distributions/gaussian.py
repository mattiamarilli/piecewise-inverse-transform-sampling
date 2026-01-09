import math


def gaussian_pdf(x, mu=0, sigma=1):
    """
    Probability Density Function (PDF) of the Gaussian (normal) distribution.

    The Gaussian distribution is defined by its mean μ and standard deviation σ:

        f(x) = (1 / (σ √(2π))) · exp(−(x − μ)² / (2σ²))

    It is symmetric around μ and integrates to 1 over ℝ.

    Parameters:
        x (float): Point at which the PDF is evaluated.
        mu (float, optional): Mean of the distribution. Default is 0.
        sigma (float, optional): Standard deviation (σ > 0). Default is 1.

    Returns:
        float: Value of the Gaussian PDF at x.
    """
    # Normalization constant ensures ∫ f(x) dx = 1
    normalization = 1 / (sigma * math.sqrt(2 * math.pi))

    # Exponential term measuring the squared distance from the mean
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)

    return normalization * math.exp(exponent)


def gaussian_cdf_piecewise(xs):
    """
    Approximate the Gaussian CDF using numerical integration.

    The CDF is defined as:

        F(x) = ∫_{−∞}^{x} f(t) dt

    Since the Gaussian CDF has no closed-form expression in elementary
    functions, this function approximates it by integrating the PDF
    over a discrete grid using the trapezoidal rule.

    Parameters:
        xs (list of float): Monotonically increasing grid points.

    Returns:
        list of float: Approximated CDF values, normalized to [0, 1].
    """
    # Initialize the CDF with zero probability at the first point
    cdf = [0.0]

    for i in range(1, len(xs)):
        # Step size between consecutive grid points
        dx = xs[i] - xs[i - 1]

        # Trapezoidal rule: average PDF value on the interval
        pdf_avg = (gaussian_pdf(xs[i]) + gaussian_pdf(xs[i - 1])) / 2

        # Incremental integral: ∫ f(x) dx ≈ f_avg · dx
        cdf.append(cdf[-1] + pdf_avg * dx)

    # Total integral over the grid (≈ 1 if the domain is wide enough)
    total = cdf[-1]

    # Normalize to ensure F(x_max) = 1
    return [v / total for v in cdf]
