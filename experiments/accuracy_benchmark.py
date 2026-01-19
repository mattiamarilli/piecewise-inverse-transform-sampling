import os
import sys
import numpy as np
from scipy.stats import entropy
import time
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# ROOT DIR & import
# -------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Sampler
from samplers.ars.ars import ARS
from samplers.its.its import PiecewiseITS
from samplers.its.strategies.linear_search import PiecewiseLinearCDFLinearSearch
from samplers.its.strategies.alias_method import PiecewiseLinearCDFAlias
from samplers.its.strategies.equiprob import PiecewiseEquiprobCDF

# Distribuzioni / utils
from utils.distributions_utils import (
    gaussian_log_pdf,
    gamma_log_pdf,
    beta_log_pdf,
    kumaraswamy_log_pdf,
    gaussian_cdf_piecewise,
    gamma_cdf_piecewise,
    beta_cdf_piecewise,
    kumaraswamy_cdf,
    kumaraswamy_equiprobable_points
)

# Import per ARS builder
from performance_benchmark import build_ars_sampler

# -------------------------------------------------------------------------
# PDF / CDF helper
# -------------------------------------------------------------------------

def true_log_pdf(dist, x, params):
    if dist == "gaussian":
        return gaussian_log_pdf(x, **params)
    if dist == "gamma":
        return gamma_log_pdf(x, **params)
    if dist == "beta":
        return beta_log_pdf(x, **params)
    if dist == "kumaraswamy":
        return kumaraswamy_log_pdf(x, **params)
    raise ValueError(f"Distribuzione non supportata: {dist}")

def true_pdf(dist, xs, params):
    return np.exp(true_log_pdf(dist, xs, params))

def true_cdf(dist, xs, params):
    if dist == "gaussian":
        return gaussian_cdf_piecewise(xs, **params)
    if dist == "gamma":
        return gamma_cdf_piecewise(xs, **params)
    if dist == "beta":
        return beta_cdf_piecewise(xs, **params)
    if dist == "kumaraswamy":
        return kumaraswamy_cdf(xs, **params)
    raise ValueError(f"Distribuzione non supportata: {dist}")

# -------------------------------------------------------------------------
# Metriche PDF-based (Jensen-Shannon)
# -------------------------------------------------------------------------

def pdf_metrics(samples, dist, params, n_bins=100):
    samples = np.asarray(samples)
    ll = np.mean(true_log_pdf(dist, samples, params))

    x_min, x_max = samples.min(), samples.max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    hist_empirical, _ = np.histogram(samples, bins=bins, density=True)
    xs_center = 0.5 * (bins[:-1] + bins[1:])
    pdf_true = true_pdf(dist, xs_center, params)

    P = pdf_true / pdf_true.sum()
    Q = hist_empirical / hist_empirical.sum()
    M = 0.5 * (P + Q)
    js_div = 0.5 * (entropy(P, M) + entropy(Q, M))

    return {"loglik_mean": ll, "js_divergence": js_div}

# -------------------------------------------------------------------------
# Metriche CDF-based
# -------------------------------------------------------------------------

def cdf_metrics(samples, dist, params, alpha=0.05, n_grid=200):
    samples = np.sort(samples)
    N = len(samples)
    xs = np.linspace(samples.min(), samples.max(), n_grid)
    ecdf = np.searchsorted(samples, xs, side="right") / N
    cdf_true = true_cdf(dist, xs, params)
    sup_err = np.max(np.abs(ecdf - cdf_true))
    eps = np.sqrt(np.log(2 / alpha) / (2 * N))
    return {
        "sup_cdf_error": sup_err,
        "dkw_epsilon": eps,
        "dkw_covered": sup_err <= eps,
        # dati per grafico
        "xs": xs,
        "ecdf": ecdf,
        "cdf_true": cdf_true,
        "dkw_upper": np.minimum(cdf_true + eps, 1.0),
        "dkw_lower": np.maximum(cdf_true - eps, 0.0)
    }


# -------------------------------------------------------------------------
# Benchmark ITS
# -------------------------------------------------------------------------

def benchmark_accuracy_its(dist_name, strategy_tag, N_batch_list, n_pieces_list, params):
    results = []

    print(f"\n=== Benchmark ITS {strategy_tag} per distribuzione {dist_name} ===")

    for n_pieces in n_pieces_list:
        print(f"Costruzione strategia con n_pieces={n_pieces}")
        if strategy_tag in ("linear", "alias"):
            xs = np.linspace(0, 1, n_pieces)
            cdf = true_cdf(dist_name, xs, params)
            if strategy_tag == "linear":
                strategy = PiecewiseLinearCDFLinearSearch(xs, cdf)
            else:
                strategy = PiecewiseLinearCDFAlias(xs, cdf)
        elif strategy_tag == "equiprob":
            xs = kumaraswamy_equiprobable_points(params["a"], params["b"], n_points=n_pieces)
            strategy = PiecewiseEquiprobCDF(xs)
        else:
            raise ValueError(f"Strategy non riconosciuta: {strategy_tag}")

        sampler = PiecewiseITS(strategy)

        for N in N_batch_list:
            print(f"Generazione batch N={N} ... ", end="")
            t0 = time.perf_counter()
            samples = sampler.draw(N)
            t1 = time.perf_counter()
            print(f"completato in {t1-t0:.4f}s")

            pdf_m = pdf_metrics(samples, dist_name, params)
            cdf_m = cdf_metrics(samples, dist_name, params)
            results.append({
                "method": f"ITS-{strategy_tag}",
                "dist": dist_name,
                "N": N,
                "n_pieces": n_pieces,
                **pdf_m,
                **cdf_m
            })
    return results

# -------------------------------------------------------------------------
# Benchmark ARS
# -------------------------------------------------------------------------

def benchmark_accuracy_ars(dist_name, N_batch_list, ns_list, params):
    results = []

    print(f"\n=== Benchmark ARS per distribuzione {dist_name} ===")

    for ns in ns_list:
        print(f"Creazione ARS con ns={ns}")
        ars = build_ars_sampler(dist_name, ns, params)

        for N in N_batch_list:
            print(f"Generazione batch N={N} ... ", end="")
            t0 = time.perf_counter()
            samples = ars.draw(N)
            t1 = time.perf_counter()
            print(f"completato in {t1-t0:.4f}s")

            pdf_m = pdf_metrics(samples, dist_name, params)
            cdf_m = cdf_metrics(samples, dist_name, params)
            results.append({
                "method": "ARS",
                "dist": dist_name,
                "N": N,
                "ns": ns,
                **pdf_m,
                **cdf_m
            })
    return results



def plot_dkw_bands(cdf_data, title="Bande DKW", save_path=None):
    xs = cdf_data["xs"]
    ecdf = cdf_data["ecdf"]
    cdf_true = cdf_data["cdf_true"]
    upper = cdf_data["dkw_upper"]
    lower = cdf_data["dkw_lower"]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, cdf_true, label="CDF vera", color="blue", linewidth=2)
    plt.step(xs, ecdf, where="post", label="ECDF campione", color="red")
    plt.fill_between(xs, lower, upper, color="gray", alpha=0.3, label=f"Bande DKW ±{cdf_data['dkw_epsilon']:.3f}")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main():
    N_batch_list = [10**3, 10**4, 10**5]
    n_pieces_list = [100, 500, 1000]
    ns_list = [20, 50, 100]

    dists = {
        "gaussian": {"mu": 0.0, "sigma": 1.0},
        "gamma": {"shape": 2.0, "scale": 1.0},
        "beta": {"a": 2.0, "b": 5.0},
        "kumaraswamy": {"a": 3.0, "b": 3.0},
    }

    all_results = []

    # ITS linear / alias
    for dist, params in dists.items():
        for strat in ["linear", "alias"]:
            all_results.extend(
                benchmark_accuracy_its(dist, strat, N_batch_list, n_pieces_list, params)
            )

    # ITS equiprobabile solo kumaraswamy
    all_results.extend(
        benchmark_accuracy_its("kumaraswamy", "equiprob", N_batch_list, n_pieces_list, dists["kumaraswamy"])
    )

    # ARS
    for dist, params in dists.items():
        all_results.extend(benchmark_accuracy_ars(dist, N_batch_list, ns_list, params))

    # Stampa CSV
    print("\n=== RISULTATI CSV ===")
    print("method,dist,N,n_pieces,ns,loglik,js_div,sup_cdf_err,dkw_eps,dkw_ok")
    for r in all_results:
        print(
            f"{r['method']},{r['dist']},{r['N']},"
            f"{r.get('n_pieces','')},{r.get('ns','')},"
            f"{r['loglik_mean']:.6f},{r['js_divergence']:.6f},"
            f"{r['sup_cdf_error']:.6f},{r['dkw_epsilon']:.6f},{r['dkw_covered']}"
        )

    # esempio ITS lineare per Gaussian
    xs = np.linspace(-4, 4, 1000)  # 4 sigma a destra e sinistra della media
    cdf = true_cdf("gaussian", xs, {"mu":0,"sigma":1})
    strategy = PiecewiseLinearCDFLinearSearch(xs, cdf)
    sampler = PiecewiseITS(strategy)

    samples = sampler.draw(1000)  # campioni
    cdf_data = cdf_metrics(samples, "gaussian", {"mu":0,"sigma":1})
    plot_dkw_bands(cdf_data, title="ITS Linear - Gaussian")



if __name__ == "__main__":
    main()
