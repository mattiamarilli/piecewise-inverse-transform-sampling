def compute_equispaced_points(xs, cdf, n_points):
    """
    Compute points that are equispaced in probability space for a
    generic distribution.

    Given a discretized CDF defined by pairs (x_i, F(x_i)), this function
    computes the quantile points corresponding to uniformly spaced
    probability levels:

        u_k = k / n_points,   k = 0, ..., n_points

    The inverse CDF is approximated by linear interpolation between
    consecutive CDF samples.

    Parameters:
        xs (list of float): Monotonically increasing x-values.
        cdf (list of float): Corresponding CDF values, with
                             0 ≤ cdf[i] ≤ 1 and non-decreasing.
        n_points (int): Number of probability subdivisions.

    Returns:
        list of float: Quantile points equispaced in CDF space.
    """
    xs_equi = []

    # Uniform probability levels in [0, 1]
    u_values = [i / n_points for i in range(n_points + 1)]

    for u in u_values:
        # Find the interval such that F(x_j) ≤ u ≤ F(x_{j+1})
        for j in range(len(cdf) - 1):
            if cdf[j] <= u <= cdf[j + 1]:
                # Local coordinate in probability space
                t = (u - cdf[j]) / (cdf[j + 1] - cdf[j])

                # Linear interpolation of the inverse CDF
                x_val = xs[j] + t * (xs[j + 1] - xs[j])
                xs_equi.append(x_val)
                break

    return xs_equi
