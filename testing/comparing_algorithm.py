import hnswlib
import faiss 
import numpy as np 
from hsnw_constructionDARTH import HNSW_NEW
from pathlib import Path 

def build_hnsw_New(Xb,M=16,efC=200):
    d = Xb.shape[1]
    idx = HNSW_NEW(dim=d,M=M,efConstruction=efC,metric="l2",seed=42)
    idx.use_heuristic=True 

    for i,v in enumerate(Xb):
        idx._insert_(v,node_id=i)
    return idx 

def hnsw_New_search_fn(idx,efS):
    def _search(Xq,k):
        return idx.search(Xq,k,efSearch=efS)
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


