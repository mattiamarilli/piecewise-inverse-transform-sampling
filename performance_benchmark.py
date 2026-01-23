import time
import csv

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
        print(f"[ITS] {dist_name}, strategia: {strategy_tag}, n_pieces: {n_pieces}")

        if strategy_tag in ("linear", "alias"):
            xs, cdf = build_cdf_grid(dist_name, n_pieces, params)
            sampler = ITSLinear(xs, cdf) if strategy_tag == "linear" else ITSAlias(xs, cdf)

        elif strategy_tag == "equiprob":
            if dist_name != "kumaraswamy":
                continue
            a, b = params["a"], params["b"]
            xs_equi = kumaraswamy_equiprobable_points(a, b, n_points=n_pieces)
            sampler = ITSEquiprob(xs_equi=xs_equi)
        else:
            raise ValueError(f"Strategy non riconosciuta: {strategy_tag}")

        for N in N_batch_list:
            print(f"  Disegno batch N={N}...")
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
        print(f"[ARS] {dist_name}, ns={ns}")
        ars = build_ars_sampler(dist_name, ns=ns, params=params)

        for N in N_batch_list:
            print(f"  Disegno batch N={N}...")
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
    N_batch_list = [100, 1000, 10000, 100000]
    n_pieces_list = [10, 100, 500]
    ns_list = [3,10,20,50]

    dists = {
        "gaussian": {"mu": 0.0, "sigma": 1.0},
        "beta": {"a": 2.0, "b": 100.0},
        "kumaraswamy": {"a": 3.0, "b": 3.0},
        #"gamma": {"shape": 2.0, "scale": 1.0},
    }

    all_results = []

    for dist_name, params in dists.items():
        print(f"\n=== ITS linear/alias per {dist_name} ===")
        all_results.extend(
            benchmark_its(dist_name, "linear", N_batch_list, n_pieces_list, params)
        )
        all_results.extend(
            benchmark_its(dist_name, "alias", N_batch_list, n_pieces_list, params)
        )

    for dist_name in ["kumaraswamy"]:
        params = dists[dist_name]
        print(f"\n=== ITS equiprob per {dist_name} ===")
        all_results.extend(
            benchmark_its(dist_name, "equiprob", N_batch_list, n_pieces_list, params, equiprob_only=True)
        )

    for dist_name, params in dists.items():
        print(f"\n=== ARS per {dist_name} ===")
        all_results.extend(
            benchmark_ars(dist_name, N_batch_list, ns_list, params)
        )

    output_file = "benchmark_results.csv"
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # intestazioni
        writer.writerow(["method", "dist", "N", "n_pieces", "ns", "time_sec"])
        # dati
        for r in all_results:
            writer.writerow([
                r["method"],
                r["dist"],
                r["N"],
                r["n_pieces"] if r["n_pieces"] is not None else "",
                r["ns"] if r["ns"] is not None else "",
                f"{r['time_sec']:.6f}"
            ])

    print(f"\nRisultati salvati in '{output_file}'")


if __name__ == "__main__":
    main()
