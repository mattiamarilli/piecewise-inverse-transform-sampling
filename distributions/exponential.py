import math


def exponential_pdf(x, lambd=1.0):
    """
    Probability Density Function (PDF) of the exponential distribution.

    The exponential distribution models the waiting time between events
    in a Poisson process with rate λ.

        f(x) = λ e^(−λx), x ≥ 0
        f(x) = 0, x < 0

    Parameters:
        x (float): Point at which the PDF is evaluated.
        lambd (float, optional): Rate parameter (λ > 0). Default is 1.0.

    Returns:
        float: Value of the exponential PDF at x.
    """
    # The support of the exponential distribution is x ≥ 0
    if x < 0:
        return 0.0

    # Analytical expression of the PDF
    return lambd * math.exp(-lambd * x)


def exponential_cdf(x, lambd=1.0):
    """
    Cumulative Distribution Function (CDF) of the exponential distribution.

    The CDF is obtained by integrating the PDF:

        F(x) = ∫₀ˣ λ e^(−λt) dt = 1 − e^(−λx),   x ≥ 0

    Parameters:
        x (float): Point at which the CDF is evaluated.
        lambd (float, optional): Rate parameter (λ > 0). Default is 1.0.

    Returns:
        float: Value of the exponential CDF at x.
    """
    # Probability mass below zero is zero
    if x < 0:
        return 0.0

    return 1 - math.exp(-lambd * x)


def exponential_cdf_piecewise(xs, lambd=1.0):
    """
    Evaluate the exponential CDF on a discrete set of points.

    This function applies F(x) independently to each point in xs,
    producing a discretized representation of the CDF.

    Parameters:
        xs (list of float): Points at which the CDF is evaluated.
        lambd (float, optional): Rate parameter (λ > 0). Default is 1.0.

    Returns:
        list of float: CDF values corresponding to xs.
    """
    return [exponential_cdf(x, lambd) for x in xs]


def compute_equispaced_points_exp(lambd=1.0, n_points=100, x_max=10.0):
    """
    Compute points that are equispaced in probability space for
    the exponential distribution.

    Using the inverse CDF (quantile function):

        F⁻¹(u) = −ln(1 − u) / λ ,   u ∈ [0, 1)

    these points correspond to uniform subdivisions of the CDF,
    resulting in non-uniform spacing in the x-domain.

    Parameters:
        lambd (float, optional): Rate parameter (λ > 0). Default is 1.0.
        n_points (int, optional): Number of probability subdivisions.
        x_max (float, optional): Maximum x-value used to close the
                                 final interval.

    Returns:
        list of float: Quantile points of the exponential distribution.
    """
    xs_equi = []

    for k in range(n_points):
        # Uniform probability level u ∈ [0, 1), last point excluded
        u = k / n_points

        # Inverse CDF (quantile function)
        x = -math.log(1 - u) / lambd
        xs_equi.append(x)

    # Append a cutoff to bound the last interval
    xs_equi.append(x_max)

    return xs_equi
