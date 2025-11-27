import numpy as np 
from heapq import heappush,heappop
import math
class HNSW_NEW:

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
        self.mL = 1.0/math.log(self.M)
    """
    From paper  algorithm 1: full insertion
                algorithm 2: searching layer to get candidate list
                algorithm 3/4 : selecting M neighbors for connection
    """
    #return last level
    def max_level(self) -> int: 
        return self.maxlevel

    #list of levels
    def levels(self) -> list:
        return list(range(self.maxlevel+1))
    
    #add nodes here 
    def _insert_(self,vector,node_id=None):
        vec = np.asarray(vector,dtype=float)
        if vec.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vec.shape}")

        if node_id is None:
            node_id = len(self.vectors)
        if node_id in self.vectors:
            raise ValueError(f"Node id {node_id} already exists")
        
        self.vectors[node_id] = vec
        l = self.probab_levels(self.mL)

        #for first node
        if self.entry_id is None:
            for _ in range(l+1):
                self.layers.append({})
            self.entry_id = node_id 
            self.maxlevel = l
            return node_id 
        
        #from here: we have at least one node already
        old_top = self.maxlevel
        if l > self.maxlevel:
            for _ in range(self.maxlevel +1,l+1):
                self.layers.append({})
            self.maxlevel = l 
        
        current_entryPoint = self.entry_id 
        L = old_top 

        #TODO: IMPLEMENT SEARCH_LAYER
        for lc in range(L,l,-1):
            ep = self._search_layer_greedy(vec,current_entryPoint,lc) #search if they are existing upper layers thorught this node
        
        #phase 2: for layer min(L,l) down to 0 use efconstruction,selet neighbors ,connect ,prune
        for lc in range(min(L,l), -1 ,-1):
            if node_id not in self.layers[lc]:
                self.layers[lc][node_id] = set()
            
            W = self._search_layer(vec,current_entryPoint,lc,self.efConstruction)
            neighbors = self._select_neighbors(vec,W,lc,self.M if lc > 0 else self.M0)

            for nb in neighbors:
                if nb not in self.layers[lc]:
                    self.layers[lc][nb] = set()
                self.layers[lc][node_id].add(nb)
                self.layers[lc][nb].add(node_id)

                Mmax = self.M0 if lc == 0 else self.M 
                if len(self.layers[lc][nb]) > Mmax:
                    pruned = self._prune_connections(nb,lc,Mmax)
                    self.layers[lc][nb]=pruned 
            if len(W)>0:
                ep = W[0]
        
        if l > old_top:
            self.entry_id = node_id 
            return node_id
    
    def _search_layer_greedy(self):
        pass 
        
    def _search_layer(self):
        pass

    def _prune_connections(self):
        pass 
    
    def _select_neighbors(self):
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
