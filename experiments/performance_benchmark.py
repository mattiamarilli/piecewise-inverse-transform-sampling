import os
import sys
import time
import numpy as np
from scipy.stats import gamma


# Aggiunge la root del progetto a sys.path per import relativi
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import sampler ARS
from samplers.ars.ars import ARS

# Import ITS + strategie
from samplers.its.its import PiecewiseITS
from samplers.its.strategies.linear_search import PiecewiseLinearCDFLinearSearch
from samplers.its.strategies.alias_method import PiecewiseLinearCDFAlias
from samplers.its.strategies.equiprob import PiecewiseEquiprobCDF

# Utils
from utils.cdf_utils import compute_equispaced_points
from utils.distributions_utils import (
    beta_log_pdf,
    beta_log_pdf_prime,
    beta_cdf_piecewise,
    gamma_log_pdf,
    gamma_log_pdf_prime,
    gamma_cdf_piecewise,
    gaussian_log_pdf,
    gaussian_log_pdf_prime,
    gaussian_cdf_piecewise,
    kumaraswamy_cdf,
    kumaraswamy_equiprobable_points,
    kumaraswamy_log_pdf,
    kumaraswamy_log_pdf_prime
)

# -------------------------------------------------------------------------
# Helper per costruire CDF piecewise su un grid
# -------------------------------------------------------------------------

def build_cdf_grid(dist_name, n_pieces, params):
    """
    Costruisce una griglia xs e la corrispondente CDF piecewise per la
    distribuzione specificata.

    Parameters
    ----------
    dist_name : str
        'gaussian', 'gamma', 'beta', 'kumaraswamy'
    n_pieces : int
        Numero di punti di discretizzazione (pieces)
    params : dict
        Parametri della distribuzione.

    Returns
    -------
    xs : np.ndarray
    cdf : np.ndarray
    """
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
        # support [0, upper] con upper = quantile 0.999
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


# -------------------------------------------------------------------------
# Costruzione ARS per le varie distribuzioni
# -------------------------------------------------------------------------

def build_ars_sampler(dist_name, ns, params):
    """
    Costruisce un oggetto ARS per la distribuzione indicata.

    Parameters
    ----------
    dist_name : str
        'gaussian', 'gamma', 'beta', 'kumaraswamy'
    ns : int
        Numero massimo di punti di supporto memorizzati (ns in ARS).
    params : dict
        Parametri della distribuzione.

    Returns
    -------
    ars : ARS
    """

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
        # moda ~ (shape-1)*scale, scegliamo xi intorno ad essa se shape>1
        if shape > 1:
            mode = (shape - 1) * scale
            xi = [mode * 0.5, mode * 1.5]
        else:
            # se shape <= 1, moda in 0 (comunque ok ma più tricky)
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
        # moda ~ (a-1)/(a+b-2), scegliamo xi attorno
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
        # moda interna per a>1,b>1, facciamo xi in (0,1)
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


# -------------------------------------------------------------------------
# Benchmark ITS (piecewise CDF) + equiprobabile
# -------------------------------------------------------------------------

def benchmark_its(
    dist_name,
    strategy_tag,
    N_batch_list,
    n_pieces_list,
    params,
    equiprob_only=False,
):
    """
    Esegue benchmark ITS per la distribuzione specificata.

    strategy_tag:
      - "linear"
      - "alias"
      - "equiprob" (usa PiecewiseEquiprobCDF)
    equiprob_only=True:
      - usa solo "equiprob" (per kumaraswamy ed esponenziale).
    """
    results = []

    for n_pieces in n_pieces_list:
        # -- Costruzione strategia in base a strategy_tag
        if strategy_tag in ("linear", "alias"):
            xs, cdf = build_cdf_grid(dist_name, n_pieces, params)

            if strategy_tag == "linear":
                strategy = PiecewiseLinearCDFLinearSearch(xs, cdf)
            elif strategy_tag == "alias":
                strategy = PiecewiseLinearCDFAlias(xs, cdf)
            else:
                raise ValueError

            sampler = PiecewiseITS(strategy=strategy)

        elif strategy_tag == "equiprob":
            # equiprobabile: solo per kumaraswamy
            if dist_name == "kumaraswamy":
                a = params["a"]
                b = params["b"]
                xs_equi = kumaraswamy_equiprobable_points(a, b, n_points=n_pieces)
            else:
                # non dovrebbe essere chiamato su altre distribuzioni
                continue

            strategy = PiecewiseEquiprobCDF(xs_equi=xs_equi)
            sampler = PiecewiseITS(strategy=strategy)
        else:
            raise ValueError(f"Strategy non riconosciuta: {strategy_tag}")

        for N in N_batch_list:
            t0 = time.perf_counter()
            _ = sampler.draw(N)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            results.append(
                {
                    "method": f"ITS-{strategy_tag}",
                    "dist": dist_name,
                    "N": N,
                    "n_pieces": n_pieces,
                    "ns": None,
                    "time_sec": elapsed,
                }
            )

    return results


# -------------------------------------------------------------------------
# Benchmark ARS
# -------------------------------------------------------------------------

def benchmark_ars(dist_name, N_batch_list, ns_list, params):
    results = []
    for ns in ns_list:
        ars = build_ars_sampler(dist_name, ns=ns, params=params)

        for N in N_batch_list:
            t0 = time.perf_counter()
            _ = ars.draw(N)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            results.append(
                {
                    "method": "ARS",
                    "dist": dist_name,
                    "N": N,
                    "n_pieces": None,
                    "ns": ns,
                    "time_sec": elapsed,
                }
            )

    return results


# -------------------------------------------------------------------------
# Main: configurazione esperimenti
# -------------------------------------------------------------------------

def main():
    # Dimensioni batch
    N_batch_list = [10**3, 10**4, 10**5]

    # Numero di pieces per ITS
    n_pieces_list = [100, 500, 1000]

    # Numero massimo di punti ARS
    ns_list = [20, 50, 100]

    # Parametri distribuzioni
    dists = {
        "gaussian": {"mu": 0.0, "sigma": 1.0},
        "gamma": {"shape": 2.0, "scale": 1.0},
        "beta": {"a": 2.0, "b": 5.0},
        # Kumaraswamy log-concava con parametri >1 e densità unimodale
        "kumaraswamy": {"a": 3.0, "b": 3.0},
    }

    all_results = []

    # ITS - linear & alias per tutte
    for dist_name, params in dists.items():
        print(f"=== ITS linear/alias per {dist_name} ===")

        # Linear search
        all_results.extend(
            benchmark_its(
                dist_name=dist_name,
                strategy_tag="linear",
                N_batch_list=N_batch_list,
                n_pieces_list=n_pieces_list,
                params=params,
            )
        )

        # Alias
        all_results.extend(
            benchmark_its(
                dist_name=dist_name,
                strategy_tag="alias",
                N_batch_list=N_batch_list,
                n_pieces_list=n_pieces_list,
                params=params,
            )
        )

    # ITS - equiprobabile solo per kumaraswamy
    for dist_name in ["kumaraswamy"]:
        params = dists[dist_name]
        print(f"=== ITS equiprob per {dist_name} ===")

        all_results.extend(
            benchmark_its(
                dist_name=dist_name,
                strategy_tag="equiprob",
                N_batch_list=N_batch_list,
                n_pieces_list=n_pieces_list,
                params=params,
                equiprob_only=True,
            )
        )

    # ARS per tutte le distribuzioni
    for dist_name, params in dists.items():
        print(f"=== ARS per {dist_name} ===")
        all_results.extend(
            benchmark_ars(
                dist_name=dist_name,
                N_batch_list=N_batch_list,
                ns_list=ns_list,
                params=params,
            )
        )

    # Stampa risultati in formato tabellare semplice
    print("\n==== RISULTATI BENCHMARK ====")
    print("method,dist,N,n_pieces,ns,time_sec")
    for r in all_results:
        print(
            f"{r['method']},{r['dist']},{r['N']},"
            f"{r['n_pieces'] if r['n_pieces'] is not None else ''},"
            f"{r['ns'] if r['ns'] is not None else ''},"
            f"{r['time_sec']:.6f}"
        )


if __name__ == "__main__":
    main()