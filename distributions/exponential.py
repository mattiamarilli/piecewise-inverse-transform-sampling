import math

def exponential_pdf(x, lambd=1.0):
    if x < 0:
        return 0.0
    return lambd * math.exp(-lambd * x)

def exponential_cdf(x, lambd=1.0):
    if x < 0:
        return 0.0
    return 1 - math.exp(-lambd * x)

def exponential_cdf_piecewise(xs, lambd=1.0):
    return [exponential_cdf(x, lambd) for x in xs]

def compute_equispaced_points_exp(lambd=1.0, n_points=100, x_max=10.0):
    """
    Genera punti x_i equi-probabilità per Piecewise Equally-Spaced su esponenziale.
    Nota: u=1 corrisponde a infinito, quindi si aggiunge x_max per chiudere l'intervallo finale.
    """
    xs_equi = []
    for k in range(n_points):
        u = k / n_points  # u < 1, ultimo punto escluso
        x = -math.log(1-u)/lambd
        xs_equi.append(x)
    xs_equi.append(x_max)  # chiude l'ultimo intervallo
    return xs_equi


