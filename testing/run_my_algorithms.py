from testing.comparing_cpp_algorithms import *
from metrics.benchMark import measure
from utils.create_table import create_results_df
from utils.read_files import read_fvecs, read_ivecs
from pathlib import Path


def make_sliced_search_fn(full_results):
    def sliced_search_fn(Xq, k):
        return full_results[:, :k]
    return sliced_search_fn


def run_benchMark(
    Xb, Xq, I_gt,
    ks=(10,),
    Ms=(16,),
    efCs=(200,),
    efSs=(20, 50, 100),
    Rts=(0.95,),
    ipis=(200,),
    mpis=(20,),
    offline_target_recall=(0.85, 0.90, 0.95, 0.99),
    pip_gammas=(95.0,),
    pip_deltas=(20,),
):
    rows = []
    k_max = max(ks)

    # -------------------------
    # HNSW
    # -------------------------
    for M in Ms:
        for efC in efCs:
            idx = build_hnsw_cpp(Xb, M=M, efC=efC)   # build once
            for efS in efSs:
                search_fn = hnsw_cpp_search_fn(idx, efS=efS)

                full_results = search_fn(Xq, k_max)
                sliced_search_fn = make_sliced_search_fn(full_results)

                for k in ks:
                    out = measure(sliced_search_fn, Xq, k, I_true=I_gt)
                    rows.append({
                        "Method": "hnsw_cpp",
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": efS,
                        "k": k,
                        **out
                    })

    # -------------------------
    # DARTH
    # -------------------------
    for M in Ms:
        for efC in efCs:
            idx = build_hnsw_darth_cpp(Xb, M=M, efC=efC)   # build once
            for efS in efSs:
                for Rt in Rts:
                    for ipi in ipis:
                        for mpi in mpis:
                            search_fn = hnsw_darth_cpp_search_fn(
                                idx, efS=efS, Rt=Rt, ipi=ipi, mpi=mpi, predictor=None
                            )

                            full_results = search_fn(Xq, k_max)
                            sliced_search_fn = make_sliced_search_fn(full_results)

                            for k in ks:
                                out = measure(sliced_search_fn, Xq, k, I_true=I_gt)
                                rows.append({
                                    "Method": "hnswDarth_cpp",
                                    "M": M,
                                    "efConstruction": efC,
                                    "efSearch": efS,
                                    "k": k,
                                    "Rt": Rt,
                                    "ipi": ipi,
                                    "mpi": mpi,
                                    **out
                                })

    # -------------------------
    # Ada-ef
    # -------------------------
    for M in Ms:
        for efC in efCs:
            idx = build_hnsw_adaef_cpp(Xb, M=M, efC=efC)   # already good
            for target_recall in offline_target_recall:
                search_fn = hnsw_adaef_cpp_search_fn(idx, target_recall=target_recall)

                full_results = search_fn(Xq, k_max)
                sliced_search_fn = make_sliced_search_fn(full_results)

                for k in ks:
                    out = measure(sliced_search_fn, Xq, k, I_true=I_gt)
                    rows.append({
                        "Method": "hnsw_adaef_cpp",
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": int(target_recall * 500),
                        "k": k,
                        "target_recall": target_recall,
                        **out
                    })

    # -------------------------
    # PiP
    # -------------------------
    for M in Ms:
        for efC in efCs:
            for pip_gamma in pip_gammas:
                for pip_delta in pip_deltas:
                    idx = build_hnsw_pip_cpp(
                        Xb, M=M, efC=efC,
                        pip_gamma=pip_gamma, pip_delta=pip_delta
                    )   # build once
                    for efS in efSs:
                        search_fn = hnsw_pip_cpp_search_fn(idx, efS=efS)

                        full_results = search_fn(Xq, k_max)
                        sliced_search_fn = make_sliced_search_fn(full_results)

                        for k in ks:
                            out = measure(sliced_search_fn, Xq, k, I_true=I_gt)
                            rows.append({
                                "Method": "hnsw_pip_cpp",
                                "M": M,
                                "efConstruction": efC,
                                "efSearch": efS,
                                "k": k,
                                "pip_gamma": pip_gamma,
                                "pip_delta": pip_delta,
                                **out
                            })

    return create_results_df(rows)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    DATASET_PATH = BASE_DIR / "Datasets"
    RESULTS_PATH = BASE_DIR / "results_csv"

    XB_PATH = f"{DATASET_PATH}/sift/sift_base.fvecs"
    XQ_PATH = f"{DATASET_PATH}/sift/sift_query.fvecs"
    GT_PATH = f"{DATASET_PATH}/sift/sift_groundtruth.ivecs"

    print("Loading datasets...")
    xb = read_fvecs(XB_PATH)
    xq = read_fvecs(XQ_PATH)
    I_gt = read_ivecs(GT_PATH)

    print(f"Base vectors: {xb.shape}")
    print(f"Query vectors: {xq.shape}")
    print(f"GT shape: {I_gt.shape}")

    ks = (10, 20)
    Ms = (8, 16)
    efCs = (64, 128)
    efSs = (20, 50, 100)
    Rts = (0.90, 0.95)
    ipis = (200,)
    mpis = (20,)
    offline_target_recall = (0.85, 0.90, 0.95, 0.99)
    pip_gammas = (95.0,)
    pip_deltas = (20,)

    print("Running Benchmark...")
    df = run_benchMark(
        xb, xq, I_gt=I_gt,
        ks=ks, Ms=Ms, efCs=efCs, efSs=efSs,
        Rts=Rts, ipis=ipis, mpis=mpis,
        offline_target_recall=offline_target_recall,
        pip_gammas=pip_gammas, pip_deltas=pip_deltas
    )

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_PATH / "my_algos_cpp_benchmark_results.csv"
    df.to_csv(out_path, index=True)
    print("Benchmark Finished")
    print(f"Results saved to: {out_path}")