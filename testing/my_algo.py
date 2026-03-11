from testing.comparing_algorithm import *
from utils.read_files import *
if __name__ == "__main__":
    from pathlib import Path

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    BASE_DIR = Path(__file__).parent.parent
    DATASET = BASE_DIR / "Datasets" / "siftsmall"

    XB_PATH = DATASET / "siftsmall_base.fvecs"
    XQ_PATH = DATASET / "siftsmall_query.fvecs"
    GT_PATH = DATASET / "siftsmall_groundtruth.ivecs"

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    print("Loading sift...")
    Xb = read_fvecs(str(XB_PATH))
    Xq = read_fvecs(str(XQ_PATH))
    I_gt = read_ivecs(str(GT_PATH))

    print("Shapes:")
    print("Xb:", Xb.shape)
    print("Xq:", Xq.shape)
    print("GT:", I_gt.shape)

    # -------------------------------------------------
    # Build HNSW_NEW
    # -------------------------------------------------
    M = 16
    efC = 100   
    efS = 40
    k = 10

    print("\nBuilding HNSW_NEW...")
    #idx_pip = build_hnsw_pip(Xb, M=16, efC=200, metric='l2', pip_gamma=95.0, pip_delta=20)
    #search_pip = hnsw_pip_search_fn(idx_pip, efS=100)
    #D, I = search_pip(Xq, k=10)

    idx_adaef = build_hnsw_adaef(
        Xb,
        M=16,
        efC=200,
        metric='cosine',
        offline_k=10,
        offline_target_recall=0.95,
        offline_ef_values=[50, 75, 100, 150, 200, 300, 400]
    )

    search_adaef = hnsw_adaef_search_fn(idx_adaef, target_recall=0.95)
   

    # -------------------------------------------------
    # Run search
    # -------------------------------------------------
    print("Running search...")
    D, I = search_adaef(Xq, k)

    print("Output shapes:")
    print("D:", D.shape, D.dtype)
    print("I:", I.shape, I.dtype)

    # -------------------------------------------------
    # Recall@k
    # -------------------------------------------------
    recall = np.mean([
        len(set(I[i]) & set(I_gt[i, :k])) / k
        for i in range(len(Xq))
    ])

    print(f"\nHNSW_NEW recall@{k}: {recall:.4f}")
    print("✅ HNSW_NEW test completed")
