import hnswlib
import faiss 
import numpy as np 
import sys
from pathlib import Path 

from hnsw_construction import HNSW_NEW
from hsnw_constructionDARTH import HNSW_DARTH, DummyPredictor
CPP = Path(__file__).parent.parent / 'hnsw_cpp' / 'src' / 'build'
sys.path.insert(0, str(CPP))
import hnsw_cpp

DarthCPP = Path(__file__).parent.parent/ "hnswDarth_cpp" / "src" / "build"
sys.path.insert(0, str(DarthCPP))
import hnswDarth_cpp
from hnsw_construction import HNSW_NEW
faiss.omp_set_num_threads(1)

def build_hnsw_darth(Xb, M=16, efC=200, metric='l2'):
    Xb = np.asarray(Xb, dtype=np.float32, order='C')
    d = Xb.shape[1]

    index = HNSW_DARTH(dim=d, M=M, efConstruction=efC, metric=metric)

    for i, vec in enumerate(Xb):
        index._insert_(vec, node_id=i)

    return index

def hnsw_darth_search_fn(index, efS, Rt=0.95, ipi=200, mpi=20, predictor=None):
    if predictor is None:
            predictor = DummyPredictor()
    def _search(Xq, k):
        Xq = np.asarray(Xq, dtype=np.float32, order='C')
        D, I = index.search_darth(
            Xq,
            k=k,
            efSearch=efS,
            Rt=Rt,
            ipi=ipi,
            mpi=mpi,
            predictor=predictor
        )

        return D.astype(np.float32), I.astype(np.int32)

    return _search

def build_hnsw_DARTH_cpp(Xb,M=16,efC=200,metric='l2'):
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
        return idx.search_darth(Xq, k=int(k), efSearch=efS, Rt=Rt, ipi=ipi, mpi=mpi, predictor=predictor)
    return _search


def build_hnsw_New(Xb,M=16,efC=200,metric='l2'):
    Xb  = np.asarray(Xb,dtype=np.float32,order='C')
    idx = hnsw_cpp.HNSWIndex(dim=Xb.shape[1],M=int(M),efConstruction=int(efC),metric=str(metric))
    idx.add(Xb)
    return idx

def hnsw_New_search_fn(idx,efS):
    efS = int(efS)
    def _search(Xq,k):
        Xq = np.asarray(Xq,dtype=np.float32,order='C')
        return idx.search(Xq,k=int(k),efSearch=efS)
    return _search 

def build_hnswlib(Xb,M=16,efC=200):
    d = Xb.shape[1]
    p = hnswlib.Index(space='l2',dim=d)
    p.init_index(max_elements=len(Xb),ef_construction=efC,M=M)
    p.add_items(Xb,np.arange(len(Xb)))
    return p

def hnswlib_search_fn(p,efS):
    p.set_ef(efS)
    def _search(Xq,k):
        I,D = p.knn_query(Xq,k=k)
        return D.astype(np.float32), I.astype(np.int32)
    return _search


def build_faiss_hnsw(Xb,M=16,efC=200):
    d = Xb.shape[1]
    index = faiss.IndexHNSWFlat(d,M)
    index.hnsw.efConstruction = efC 
    index.add(Xb)
    return index 

def faiss_search_fn(index,efS):
    index.hnsw.efSearch = efS 
    def _search(Xq,k):
        D,I = index.search(Xq,k)
        return D.astype(np.float32),I.astype(np.int32)
    return _search


