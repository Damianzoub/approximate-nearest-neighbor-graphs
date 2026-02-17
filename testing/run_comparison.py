from testing.comparing_algorithm import *
from metrics.benchMark import measure
from utils.create_table import create_results_df
from utils.read_files import read_fvecs , read_ivecs
from pathlib import Path
def run_benchMark(Xb,Xq,I_gt,ks=(10,),Ms=(16,),efCs=(32,64,128,200,),efSs=(20,50,100)):
    rows = []

    #hnswlib
    for M in Ms:
        for efC in efCs:
            p = build_hnswlib(Xb,M=M,efC=efC)
            for efS in efSs:
                search_fn = hnswlib_search_fn(p,efS=efS)
                for k in ks:
                    out = measure(search_fn,Xq,k,I_true=I_gt,warmup=1)
                    rows.append({
                        "Method":"hnswlib",
                        "M":M,
                        "efConstruction":efC,
                        "efSearch":efS,
                        "k":k,
                        **out
                    })
    
    #Faiss Hnsw
    for M in Ms:
        for efC in efCs:
            index = build_faiss_hnsw(Xb,M,efC)
            for efS in efSs:
                search_fn = faiss_search_fn(index,efS=efS)
                for k in ks:
                    out = measure(search_fn,Xq,k,I_true=I_gt,warmup=1)
                    rows.append({
                        "Method": "faiss-hnsw",
                        "M": M,
                        "efConstruction": efC,
                        "efSearch": efS,
                        "k": k,
                        **out
                    })

    #HNSW NEW 
    for M in Ms:
        for efC in efCs:
            idx = build_hnsw_New(Xb,M=M,efC=efC)
            for efS in efSs:
                search_fn = hnsw_New_search_fn(idx,efS=efS)
                for k in ks:
                    out = measure(search_fn,Xq,k,I_true = I_gt)
                    rows.append({
                        "Method": "HNSW NEW",
                        "M": M,
                        "efConstruction": efC,
                        "efSearch":efS,
                        "k":k,
                        **out
                    })
    
    df = create_results_df(rows)
    return df


if __name__ =="__main__":
    BASE_DIR = Path(__file__).parent.parent
    DATASET_PATH = BASE_DIR/"Datasets/siftsmall"
    XB_PATH = f"{DATASET_PATH}/siftsmall_base.fvecs"
    XQ_PATH = f"{DATASET_PATH}/siftsmall_query.fvecs"
    GT_PATH = f"{DATASET_PATH}/siftsmall_groundtruth.ivecs"

    print("Loading datasets...")
    xb = read_fvecs(XB_PATH)
    xq = read_fvecs(XQ_PATH)
    I_gt = read_ivecs(GT_PATH)

    print(f"Base vectors: {xb.shape}")
    print(f"Query vectors: {xq.shape}")
    print(f"GT shape: {I_gt.shape}")

    ks = (10,20)
    Ms = (8,16)
    efCs = (64,128)
    efSs = (20,50,100)

    #Run benchmark
    print("Running Benchmark...")
    df = run_benchMark(xb,xq,I_gt=I_gt,ks=ks,Ms=Ms,efCs=efCs,efSs=efSs)
    out_path = "benchmark_results.csv"
    df.to_csv(out_path,index=True)
    print(f"Benchmark Finished")
    print(f"Results saved to: {out_path}")

