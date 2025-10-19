from utils.read_files import read_fvecs, read_ivecs

def load_siftsmall(base_dir):
    xb = read_fvecs(f"{base_dir}/siftsmall_base.fvecs")
    xq = read_fvecs(f"{base_dir}/siftsmall_query.fvecs")
    xt = read_ivecs(f"{base_dir}/siftsmall_groundtruth.ivecs")
    return xb,xq,xt
