import os
import sys
import time
import numpy as np
from scipy.stats import expon, gamma


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
    exponential_cdf_piecewise,
    compute_equispaced_points_exp,
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
        'gaussian', 'gamma', 'beta', 'kumaraswamy', 'exponential'
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

    elif dist_name == "exponential":
        lambd = params["lambd"]
        # support [0, upper] con upper = quantile 0.999
        upper = expon.ppf(0.999, scale=1.0 / lambd)
        xs = np.linspace(0.0, upper, n_pieces)
        cdf = exponential_cdf_piecewise(xs, lambd=lambd)
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
        'gaussian', 'gamma', 'beta', 'kumaraswamy', 'exponential'
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
        xi = [-4,1,40]
        return ARS(
            f=gaussian_log_pdf,
            fprima=gaussian_log_pdf_prime,
            xi=xi,
            mu=mu,
            sigma=sigma,
        )

    elif dist_name == "gamma":
        shape = params["shape"]
        scale = params.get("scale", 1.0)
        xi = [0.1,1,40]
        return ARS(
            f=gamma_log_pdf,
            fprima=gamma_log_pdf_prime,
            xi=xi,
            lb=0,
            shape=shape,
            scale=scale,
        )

    elif dist_name == "beta":
        a = params["a"]
        b = params["b"]
        xi = [0.1, 0.5, 0.9]
        return ARS(
            f=beta_log_pdf,
            fprima=beta_log_pdf_prime,
            xi=xi,
            lb=0,
            ub=1,
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

    elif dist_name == "exponential":
        lambd = params["lambd"]
        # moda in 0, scegliamo alcuni punti positivi
        xi = [1.0 / (3 * lambd), 1.0 / lambd, 3.0 / lambd]
        lb = 0.0
        ub = np.inf

        def exp_log_pdf(x, lambd=1.0):
            x = np.asarray(x)
            # log(lambd) - lambd * x (ignoro normale perché ARS lavora con log-densità a costante additiva)
            return np.log(lambd) - lambd * x

        def exp_log_pdf_prime(x, lambd=1.0):
            x = np.asarray(x)
            return -lambd * np.ones_like(x)

        return ARS(
            f=exp_log_pdf,
            fprima=exp_log_pdf_prime,
            xi=xi,
            lb=lb,
            ub=ub,
            use_lower=False,
            ns=ns,
            lambd=lambd,
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
            # equiprobabile: solo per kumaraswamy ed esponenziale come da richiesta
            if dist_name == "exponential":
                lambd = params["lambd"]
                xs_equi = compute_equispaced_points_exp(
                    lambd=lambd,
                    n_points=n_pieces,
                    x_max=10.0,  # o altro upper bound
                )
            elif dist_name == "kumaraswamy":
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
        "gaussian": {"mu": 2.0, "sigma": 3.0},
        "gamma": {"shape": 2.0, "scale": 0.5},
        "beta": {"a": 1.3, "b": 2.7},
        #Kumaraswamy log-concava con parametri >1 e densità unimodale
        "kumaraswamy": {"a": 3.0, "b": 3.0},
        "exponential": {"lambd": 1.0},
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

    # ITS - equiprobabile solo per kumaraswamy ed esponenziale
    for dist_name in ["kumaraswamy", "exponential"]:
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
        if dist_name in ["kumaraswamy", "exponential"]:
             # ARS non implementato per queste due distribuzioni
             continue
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