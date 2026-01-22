# Utils
import numpy as np
from samplers.ars import ARS
from utils.distributions_utils import (
    beta_cdf_piecewise,
    beta_log_pdf,
    beta_log_pdf_prime,
    gamma_cdf_piecewise,
    gamma_log_pdf,
    gamma_log_pdf_prime,
    gaussian_cdf_piecewise,
    gaussian_log_pdf,
    gaussian_log_pdf_prime,
    kumaraswamy_cdf,
    kumaraswamy_log_pdf,
    kumaraswamy_log_pdf_prime
)
from scipy.stats import gamma


def compute_equispaced_points(xs, cdf, n_points):
    xs_equi = []
    u_values = [i / n_points for i in range(n_points + 1)]

    for u in u_values:
        for j in range(len(cdf) - 1):
            if cdf[j] <= u <= cdf[j + 1]:
                t = (u - cdf[j]) / (cdf[j + 1] - cdf[j])

                x_val = xs[j] + t * (xs[j + 1] - xs[j])
                xs_equi.append(x_val)
                break

    return xs_equi


def build_cdf_grid(dist_name, n_pieces, params):
    if dist_name == "gaussian":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        # support approssimato [-5σ, 5σ]
        lb, ub = mu - 5 * sigma, mu + 5 * sigma
        xs = np.linspace(lb, ub, n_pieces)
        cdf = gaussian_cdf_piecewise(xs, mu=mu, sigma=sigma)
        return xs, cdf

    elif dist_name == "gamma":
        shape = params["shape"]
        scale = params.get("scale", 1.0)
        upper = gamma.ppf(0.999, a=shape, scale=scale)
        xs = np.linspace(0.0, upper, n_pieces)
        cdf = gamma_cdf_piecewise(xs, shape=shape, scale=scale)
        return xs, cdf

    elif dist_name == "beta":
        a = params["a"]
        b = params["b"]
        xs = np.linspace(0.0, 1.0, n_pieces)
        cdf = beta_cdf_piecewise(xs, a=a, b=b)
        return xs, cdf

    elif dist_name == "kumaraswamy":
        a = params["a"]
        b = params["b"]
        xs = np.linspace(0.0, 1.0, n_pieces)
        cdf = kumaraswamy_cdf(xs, a, b)
        return xs, cdf

    else:
        raise ValueError(f"Distribuzione non supportata: {dist_name}")
    

def build_ars_sampler(dist_name, ns, params):
    if dist_name == "gaussian":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        # xi intorno alla moda (mu)
        xi = [mu - sigma, mu + sigma]
        lb = -np.inf
        ub = np.inf
        return ARS(
            f=gaussian_log_pdf,
            fprima=gaussian_log_pdf_prime,
            xi=xi,
            lb=lb,
            ub=ub,
            use_lower=False,
            ns=ns,
            mu=mu,
            sigma=sigma,
        )

    elif dist_name == "gamma":
        shape = params["shape"]
        scale = params.get("scale", 1.0)
        if shape > 1:
            mode = (shape - 1) * scale
            xi = [mode * 0.5, mode * 1.5]
        else:
            xi = [scale * 1e-4, scale * 0.5, scale * 2.0]
        lb = 0.0
        ub = np.inf
        return ARS(
            f=gamma_log_pdf,
            fprima=gamma_log_pdf_prime,
            xi=xi,
            lb=lb,
            ub=ub,
            use_lower=False,
            ns=ns,
            shape=shape,
            scale=scale,
        )

    elif dist_name == "beta":
        a = params["a"]
        b = params["b"]
        lb = 0.0
        ub = 1.0
        if a > 1 and b > 1:
            eps = 1e-3
            mode = (a - 1) / (a + b - 2)
            xi = [
                max(lb + eps, mode - 3*eps),
                mode - eps,
                mode + eps,
                min(ub - eps, mode + 3*eps)
            ]
        else:
           xi = [0.2, 0.5, 0.8]
        return ARS(
            f=beta_log_pdf,
            fprima=beta_log_pdf_prime,
            xi=xi,
            lb=lb,
            ub=ub,
            use_lower=False,
            ns=ns,
            a=a,
            b=b,
        )

    elif dist_name == "kumaraswamy":
        a = params["a"]
        b = params["b"]
        xi = [0.2, 0.5, 0.8]
        lb = 0.0
        ub = 1.0
        return ARS(
            f=kumaraswamy_log_pdf,
            fprima=kumaraswamy_log_pdf_prime,
            xi=xi,
            lb=lb,
            ub=ub,
            use_lower=False,
            ns=ns,
            a=a,
            b=b,
        )

    else:
        raise ValueError(f"Distribuzione non supportata per ARS: {dist_name}")