import math

def gaussian_pdf(x, mu=0, sigma=1):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -(x - mu) ** 2 / (2 * sigma ** 2)
    )

def gaussian_cdf_piecewise(xs):
    cdf = [0.0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        pdf_avg = (gaussian_pdf(xs[i]) + gaussian_pdf(xs[i - 1])) / 2
        cdf.append(cdf[-1] + pdf_avg * dx)

    total = cdf[-1]
    return [v / total for v in cdf]
