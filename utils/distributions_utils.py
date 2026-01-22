import numpy as np
from scipy.stats import beta, expon, gamma, norm

#Beta
def beta_pdf(x, a, b):
    return beta.pdf(x, a, b)

def beta_log_pdf(x, a, b):
    return (a-1)*np.log(x)+(b-1)*np.log(1-x)

def beta_log_pdf_prime(x, a, b):
    return (a-1)/x-(b-1)/(1-x)

def beta_cdf_piecewise(xs, a, b):
    xs = np.asarray(xs)
    return beta.cdf(xs, a, b)

def exponential_pdf(x, lambd=1.0):
    return lambd * np.exp(-lambd * x) if x >= 0 else 0.0

#Exp
def exponential_cdf_piecewise(xs, lambd=1.0):
    xs = np.asarray(xs)
    return expon.cdf(xs, scale=1 / lambd)

def compute_equispaced_points_exp(lambd=1.0, n_points=100, x_max=10.0):
    u = np.linspace(0, 1, n_points, endpoint=False)
    xs = expon.ppf(u, scale=1 / lambd)
    xs = np.append(xs, x_max)
    return xs

#Gamma
def gamma_pdf(x, shape, scale=1.0):
    return gamma.pdf(x, a=shape, scale=scale)


def gamma_log_pdf(x, shape, scale=1.0):
    return (shape-1)*np.log(x)-x/scale

def gamma_log_pdf_prime(x, shape, scale=1.0):
    return (shape-1)/x-1/scale

def gamma_cdf_piecewise(xs, shape, scale=1.0):
    xs = np.asarray(xs)
    return gamma.cdf(xs, a=shape, scale=scale)

#Gaussian
def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return norm.pdf(x, loc=mu, scale=sigma)

def gaussian_log_pdf(x, mu=0.0, sigma=1.0):
    return -1/(2*sigma**2)*(x-mu)**2

def gaussian_log_pdf_prime(x, mu=0.0, sigma=1.0):
    return -1/sigma**2*(x-mu)

def gaussian_cdf_piecewise(xs, mu=0.0, sigma=1.0):
    xs = np.asarray(xs)
    return norm.cdf(xs, loc=mu, scale=sigma)

# Kumaraswamy
def kumaraswamy_pdf(x, a, b):
    if isinstance(x, (float, int)):
        if 0 < x < 1:
            return a * b * (x ** (a - 1)) * ((1 - x ** a) ** (b - 1))
        else:
            return 0.0

    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)
    mask = (x > 0) & (x < 1)
    pdf[mask] = a * b * (x[mask] ** (a - 1)) * ((1 - x[mask] ** a) ** (b - 1))
    return pdf

def kumaraswamy_cdf(x, a, b):
    if isinstance(x, (float, int)):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return 1.0 - (1 - x**a) ** b

    x = np.asarray(x)
    cdf = np.zeros_like(x, dtype=float)
    cdf[x >= 1] = 1.0
    mask = (x > 0) & (x < 1)
    cdf[mask] = 1.0 - (1 - x[mask] ** a) ** b
    return cdf

def kumaraswamy_inv_cdf(u, a, b):
    u = np.asarray(u)
    return (1 - (1 - u) ** (1.0 / b)) ** (1.0 / a)

def kumaraswamy_equiprobable_points(a, b, n_points):
    u_values = np.linspace(0.0, 1.0, n_points + 1)
    xs = kumaraswamy_inv_cdf(u_values, a, b)
    return xs

def kumaraswamy_log_pdf(x, a=3.0, b=3.0):
    x = np.asarray(x)
    eps = 1e-15
    x_clipped = np.clip(x, eps, 1.0 - eps)
    return (
        np.log(a) + np.log(b)
        + (a - 1) * np.log(x_clipped)
        + (b - 1) * np.log(1 - x_clipped**a)
    )

def kumaraswamy_log_pdf_prime(x, a=3.0, b=3.0):
    x = np.asarray(x)
    eps = 1e-15
    x_clipped = np.clip(x, eps, 1.0 - eps)
    num = (a - 1) / x_clipped - (b - 1) * (a * x_clipped**(a - 1)) / (1 - x_clipped**a)
    return num

