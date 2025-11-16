import numpy as np 
from heapq import heappush,heappop

class HNSW:

    def __init__(self,dim,M,efConstruction,metric='l2',seed=42):
        self.dim = dim 
        self.M = int(M)
        self.M0 = 2* int(M)
        self.efConstruction = int(efConstruction)
        self.layers=[]
        self.vectors = {}
        self.metric = metric
        self.maxlevel = -1
        self.rng = np.random.default_rng(seed) #more faster
        self.entry_id = None 


    #return last level
    def max_level(self) -> int: 
        return self.maxlevel

    #list of levels
    def levels(self) -> list:
        return list(range(self.maxlevel+1))
    
    #add nodes here 
    def add(self):
        pass

    #search
    def search(self,k):
        pass 

    #probability of levels
    # l = 1/ln(M)
    def probab_levels(self,l): 
        L=0 
        while self.rng.random() < np.exp(-1.0/l):
            L+=1
        return L
    
    #calculate dist 
    def dist(self,a:np.ndarray,b:np.ndarray)->float:
        if self.metric =='l2':
            return float(np.linalg.norm(a-b))
        elif self.metric =='cosine':
            return 1.0- float( np.dot(a,b)) 
        else:
            raise ValueError("Unknown metric")

    #select neighbors
    def select_neighbors(self):
        pass

    #returns the entry point ? maybe an id ?
    def entry_point(self):
        return self.entry_id

    #connect from node level and below
    def _connect(self,u,v,level):
        pass

    def _prune_node(self,u,level):
        pass
    #beam search implementation
    def beam_search(self):
        pass



#INFO: Assigning levels is purely probabilistic and independent of data distribution. and levels are assigned when nodes are added.