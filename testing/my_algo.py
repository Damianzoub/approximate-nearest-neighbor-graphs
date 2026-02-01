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
    print("Loading siftsmall...")
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
    efS = 50
    k = 10

    print("\nBuilding HNSW_NEW...")
    idx = build_hnsw_New(Xb, M=M, efC=efC)

    search_fn = hnsw_New_search_fn(idx, efS)

    # -------------------------------------------------
    # Run search
    # -------------------------------------------------
    print("Running search...")
    D, I = search_fn(Xq, k)

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
