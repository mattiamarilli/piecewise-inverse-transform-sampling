import time

# Import samplers ITS
from samplers.its_linear import ITSLinear
from samplers.its_alias import ITSAlias
from samplers.its_equiprob import ITSEquiprob

# Utils
from utils.benchmark_utils import build_ars_sampler, build_cdf_grid
from utils.distributions_utils import kumaraswamy_equiprobable_points

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
    results = []

    for n_pieces in n_pieces_list:
        # -- Costruzione strategia in base a strategy_tag
        if strategy_tag in ("linear", "alias"):
            xs, cdf = build_cdf_grid(dist_name, n_pieces, params)

            if strategy_tag == "linear":
                sampler = ITSLinear(xs, cdf)
            elif strategy_tag == "alias":
                sampler = ITSAlias(xs, cdf)
            else:
                raise ValueError

        elif strategy_tag == "equiprob":
            # equiprobabile: solo per kumaraswamy
            if dist_name == "kumaraswamy":
                a = params["a"]
                b = params["b"]
                xs_equi = kumaraswamy_equiprobable_points(a, b, n_points=n_pieces)
            else:
                # non dovrebbe essere chiamato su altre distribuzioni
                continue

            sampler = ITSEquiprob(xs_equi=xs_equi)
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
    N_batch_list = [100, 1000, 10000, 100000, 1000000]
    n_pieces_list = [10, 100, 500, 1000]
    ns_list = [20, 50, 100]

    # Parametri distribuzioni
    dists = {
        "gaussian": {"mu": 0.0, "sigma": 1.0},
        "gamma": {"shape": 2.0, "scale": 1.0},
        "beta": {"a": 2.0, "b": 5.0},
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