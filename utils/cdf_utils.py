def compute_equispaced_points(xs, cdf, n_points):
    xs_equi = []
    u_values = [i / n_points for i in range(n_points+1)]

    for u in u_values:
        for j in range(len(cdf)-1):
            if cdf[j] <= u <= cdf[j+1]:
                t = (u - cdf[j]) / (cdf[j+1] - cdf[j])
                x_val = xs[j] + t * (xs[j+1] - xs[j])
                xs_equi.append(x_val)
                break
    return xs_equi
