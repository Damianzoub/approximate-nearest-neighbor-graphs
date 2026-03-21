import numpy as np
import sys
import importlib.util
from pathlib import Path

hnsw_cpp_path = str(Path(__file__).parent.parent / 'hnsw_cpp' / 'src' / 'build')
sys.path.insert(0, hnsw_cpp_path)
import hnsw_cpp

hnswDarth_cpp_path = str(Path(__file__).parent.parent / 'hnswDarth_cpp' / 'src' / 'build')
sys.path.insert(0, hnswDarth_cpp_path)
import hnswDarth_cpp

hnsw_pip_so_path = Path(__file__).parent.parent / 'hnsw_pip_cpp' / 'src' / 'build' / 'hnsw_pip.cpython-311-darwin.so'
spec = importlib.util.spec_from_file_location('hnswPip_cpp', hnsw_pip_so_path)
hnsw_pip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hnsw_pip)

hnsw_adaef_cpp_path = Path(__file__).parent.parent / 'hnsw_adaef_cpp' / 'src' / 'build'
hnsw_adaef_so = list(hnsw_adaef_cpp_path.glob('*.so'))
if hnsw_adaef_so:
    sys.path.insert(0, str(hnsw_adaef_cpp_path))
    import hnswAdaEF_cpp
    _HAS_ADAEF_CPP = True
else:
    hnsw_adaef_cpp = None
    _HAS_ADAEF_CPP = False


def build_hnsw_cpp(Xb, M=16, efC=200, metric='l2'):
    Xb = np.asarray(Xb, dtype=np.float32, order='C')
    idx = hnsw_cpp.HNSWIndex(dim=Xb.shape[1], M=int(M), efConstruction=int(efC), metric=str(metric))
    idx.add(Xb)
    return idx


def hnsw_cpp_search_fn(idx, efS):
    efS = int(efS)

    def _search(Xq, k):
        Xq = np.asarray(Xq, dtype=np.float32, order='C')
        return idx.search(Xq, k=int(k), efSearch=efS)

    return _search


def build_hnsw_darth_cpp(Xb, M=16, efC=200, metric='l2'):
    Xb = np.asarray(Xb, dtype=np.float32, order='C')
    idx = hnswDarth_cpp.HNSWDarthIndex(dim=Xb.shape[1], M=int(M), efConstruction=int(efC), metric=str(metric))
    idx.add(Xb)
    return idx


def hnsw_darth_cpp_search_fn(idx, efS, Rt=0.95, ipi=200, mpi=20, predictor=None):
    efS = int(efS)
    Rt = float(Rt)
    ipi = int(ipi)
    mpi = int(mpi)

    def _search(Xq, k):
        Xq = np.asarray(Xq, dtype=np.float32, order='C')
        return idx.search_darth(Xq, k=int(k), efSearch=efS, Rt=Rt, predictor=predictor, ipi=ipi, mpi=mpi)

    return _search


def build_hnsw_pip_cpp(Xb, M=16, efC=200, metric='l2', pip_gamma=95.0, pip_delta=20):
    Xb = np.asarray(Xb, dtype=np.float32, order='C')
    idx = hnsw_pip.HNSWPiPIndex(
        dim=Xb.shape[1],
        M=int(M),
        efConstruction=int(efC),
        metric=str(metric),
        pip_gamma=float(pip_gamma),
        pip_delta=int(pip_delta)
    )
    idx.add(Xb)
    return idx


def hnsw_pip_cpp_search_fn(idx, efS):
    efS = int(efS)

    def _search(Xq, k):
        Xq = np.asarray(Xq, dtype=np.float32, order='C')
        return idx.search(Xq, k=int(k), efSearch=efS)

    return _search


def build_hnsw_adaef_cpp(
    Xb,
    M=16,
    efC=200,
    metric='cosine',
    adaef_bins=5,
    adaef_delta=0.001,
    adaef_sample_size=200,
    offline_k=10,
    offline_target_recall=0.95,
    offline_ef_values=None,
):
    if not _HAS_ADAEF_CPP:
        raise ImportError("hnsw_adaef_cpp module not built. Run: cd hnsw_adaef_cpp/src/build && cmake .. && make")
    Xb = np.asarray(Xb, dtype=np.float32, order='C')
    idx = hnswAdaEF_cpp.HNSWAdaEFIndex(
        dim=Xb.shape[1],
        M=int(M),
        efConstruction=int(efC),
        metric=str(metric),
        adaef_bins=int(adaef_bins),
        adaef_delta=float(adaef_delta),
        adaef_sample_size=int(adaef_sample_size)
    )
    idx.add(Xb)

    if offline_ef_values is None:
        offline_ef_values = [50, 75, 100, 150, 200, 300, 400, 500]

    idx.build_adaef_offline(
        k=int(offline_k),
        target_recall=float(offline_target_recall),
        ef_values=offline_ef_values
    )

    return idx


def hnsw_adaef_cpp_search_fn(idx, target_recall=0.95):
    target_recall = float(target_recall)

    def _search(Xq, k):
        Xq = np.asarray(Xq, dtype=np.float32, order='C')
        return idx.search(Xq, k=int(k), target_recall=target_recall)

    return _search
