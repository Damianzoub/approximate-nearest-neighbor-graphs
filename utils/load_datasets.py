from utils.read_files import read_fvecs, read_ivecs

def load_sift1M(base_dir):
    xb = read_fvecs(f"{base_dir}/sift_base.fvecs")
    xq = read_fvecs(f"{base_dir}/sift_query.fvecs")
    xt = read_ivecs(f"{base_dir}/sift_groundtruth.ivecs")
    return xb,xq,xt