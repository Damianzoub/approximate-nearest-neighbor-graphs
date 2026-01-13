import numpy as np 
import heapq
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
        
        ep = self.entry_id
        L = old_top 

        #TODO: IMPLEMENT SEARCH_LAYER
        for lc in range(L,l,-1):
            ep = self._search_layer_greedy(vec,ep,lc,ef=1) #search if they are existing upper layers thorught this node
        
        #phase 2: for layer min(L,l) down to 0 use efconstruction,selet neighbors ,connect ,prune
        for lc in range(min(L,l), -1 ,-1):
            if node_id not in self.layers[lc]:
                self.layers[lc][node_id] = set()
            #θα το σκεφτω μετα το current_entryPoint αν ειναι σωστο
            W = self._search_layer(vec,ep,lc,self.efConstruction)
            neighbors = self.select_neighbors_heuristic(vec,W,lc,self.M if lc > 0 else self.M0)

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
    #searchs from entry point all the way to the layer that they new node exists
    def _search_layer_greedy(self,vec,curr_entryPointID:int,lc:int,ef:int=1)->int:

        best = curr_entryPointID
        best_dist = self.dist(vec,self.vectors[best])
        improved = True
        while improved:
            improved = False 
            neighbors = self.layers[lc].get(best,set())

            for nb in neighbors:
                d = self.dist(vec,self.vectors[nb])
                if d < best_dist:
                    best_dist = d
                    best = nb 
                    improved = True 
                    break
        return best        
    #beam_search   
    def _search_layer(self,vec,ep_id:int,layer:int,ef:int):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer])==0:
            return []
        visited =set() #for visited nodes
        W = [] #w: beam of the best nodes
        C = [] #min-heap

        dist_ep = self.dist(vec,self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C,(dist_ep,ep_id))
        W.append((dist_ep,ep_id))
        W.sort(key=lambda x: x[0]) #sort ascending to keep the furthest as the last one

        while C:
            dist_c,c_id = heapq.heappop(C)
            dist_f , f_id = W[-1]

            #if worst than the last one then break
            if dist_c > dist_f:
                break
            neighbors = self.layers[layer].get(c_id,set())
            for nb in neighbors:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self.dist(vec,self.vectors[nb])

                if len(W) < ef or d < W[-1][0]: #if we have space or d better than the last element in the W
                    heapq.heappush(C,(d,nb))
                    W.append((d,nb))
                    W.sort(key=lambda x: x[0])

                    if len(W) > ef:
                        W.pop()
            W.sort(key=lambda x:x[0])
        return [node_id for (dist,node_id) in W]
    
    #here we check about how many nodes are going to become neighbors from the select_layers candidates 
    def _select_neighbors_simple(self,vec,candidates,layer:int,Mmax:int):
        unique_candidates = list(dict.fromkeys(candidates)) #remove duplicates

        if len(unique_candidates) <= Mmax:
            return unique_candidates
        dist_list = []
        for nb in unique_candidates:
            d = self.dist(vec,self.vectors[nb])
            dist_list.append((d,nb))
        dist_list.sort(key=lambda x: x[0])
        selected = [nb for (d,nb) in dist_list[:Mmax]]
        return selected
    
    def _select_neighbors_heuristic_paper(self,vec,candidates,layer:int,M:int,extend_candidates:bool=True,keep_pruned_connections:bool=False):
        R = []
        R_ids = set()
        W_set = set(candidates)
        W = []
        for nb in W_set:
            d = self.dist(vec,self.vectors[nb])
            heapq.heappush(W,(d,nb))
        if extend_candidates:
            original_C = list(W_set)
            for e in original_C:
                neighs = self.layers[layer].get(e,set())
                for eadj in neighs:
                    if eadj not in W_set:
                        W_set.add(eadj)
                        d = self.dist(vec,self.vectors[eadj])
                        heapq.heappush(W,(d,eadj))
        
        Wd =[]
        while W and len(R) < M:
            d_e , e = heapq.heappop(W)

            if not R:
                R.append((d_e,e))
                R_ids.add(e)
            else:
                min_d_in_R = min(d_r for (d_r,r) in R)
                if d_e < min_d_in_R:
                    R.append((d_e,e))
                    R_ids.add(e)
                else:
                    Wd.append((d_e,e))
        
        if keep_pruned_connections and len(R) < M and Wd:
            Wd.sort(key=lambda x: x[0])
            for d_e, e in Wd:
                if len(R) >= M:
                    break
                if e not in R_ids:
                    R.append((d_e,e))
                    R_ids.add(e)
        return [e for (d_e,e) in R]
         

    def select_neighbors_heuristic(self,vec,candidates,layer:int,Mmax:int): #a little bit different from mine
        unique_candidates = list(dict.fromkeys(candidates))
        cand_with_dist = []
        for nb in unique_candidates:
            d = self.dist(vec,self.vectors[nb])
            cand_with_dist.append((d,nb))
        
        cand_with_dist.sort(key=lambda x: x[0])
        R =[]
        for d_e,e in cand_with_dist:
            if len(R)>=Mmax:
                break
            good = True 
            for d_r,r in R:
                d_er = self.dist(self.vectors[e],self.vectors[r])
                if d_er < d_e:
                    good =False 
                    break 
                    
            if good:
                R.append((d_e,e))
        return [e for (d_e,e) in R]
    #search
    def _query(self,q_vec,K,numSearch): #algorithms 5 k-nn search
        if self.entry_id is None:
            return []
        ep = self.entry_id
        L = self.maxlevel
        #greedy in the above layers
        for lc in range(L,0,-1):
            ep = self._search_layer_greedy(q_vec,ep,lc,ef=1)
        
        W = self._search_layer(q_vec,ep,0,numSearch)
        return W[:K]

    #probability of levels
    # mL=l = 1/ln(M)
    def probab_levels(self,l): 
        U = max(self.rng.random(),1e-12)
        return int(-math.log(U)*l)
        
    
    #calculate dist 
    def dist(self,a:np.ndarray,b:np.ndarray)->float:
        if self.metric =='l2':
            return float(np.linalg.norm(a-b))
        elif self.metric =='cosine':
            return 1.0- float( np.dot(a,b)) 
        else:
            raise ValueError("Unknown metric")


    #returns the entry point ? maybe an id ?
    def entry_point(self):
        return self.entry_id

    def _prune_connections(self,node_id:int,layer:int,Mmax:int):
        neigh_set = self.layers[layer].get(node_id,set())
        if len(neigh_set) <= Mmax:
            return neigh_set
        
        neighbors = list(neigh_set)
        q_vec = self.vectors[node_id]

        if getattr(self,"use_heuristic",False):
            new_neigh_list = self._select_neighbors_heuristic_paper(
                q_vec,neighbors,layer=layer,M=Mmax,extend_candidates=True,keep_pruned_connections=False
            )
        else:
            new_neigh_list = self._select_neighbors_simple(
                q_vec,neighbors,layer,Mmax
            )
        new_neigh_set = set(new_neigh_list)
        removed = neigh_set-new_neigh_set
        #maintain bidirectionality
        for nb in removed:
            if nb in self.layers[layer]:
                self.layers[layer][nb].discard(node_id)
        self.layers[layer][node_id] = new_neigh_set
        return new_neigh_set
        
    



#INFO: Assigning levels is purely probabilistic and independent of data distribution. and levels are assigned when nodes are added.
