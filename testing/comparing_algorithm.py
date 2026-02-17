import hnswlib
import faiss 
import numpy as np 
from hnsw_construction import HNSW_NEW
from pathlib import Path 
import sys
CPP = Path(__file__).parent.parent / 'hnsw_cpp' / 'src' / 'build' 
sys.path.append(str(CPP))
import hnsw_cpp

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


