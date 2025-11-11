import numpy as np 


class HNSW:

    def __init__(self,dim,M,efConstruction,metric='l2'):
        self.dim = dim 
        self.M = M
        self.M0 = 2* int(M)
        self.efConstruction = int(efConstruction)
        self.layers=[]
        self.vectors = {}
        self.metric = metric
        self.maxlevel = -1



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
    def probab_levels(self,l=0.01):
        return np.exp(-self.level/l)
    
    #calculate dist 
    def dist(self,a:np.ndarray,b:np.ndarray)->float:
        if self.metric =='l2':
            return float(np.linalg.norm(a-b))
        elif self.metric =='cosine':
            return 1.0- float( np.dot(a,b)) 
        else:
            raise ValueError("Unknown metric")
        pass

    #select neighbors
    def select_neighbors(self):
        pass

    #returns the entry point ? maybe an id ?
    def entry_point(self):
        pass