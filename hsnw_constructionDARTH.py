import numpy as np 
import heapq
import math

class DummyPredictor:
    def predict(self, feats):
        # Dummy predictor that returns random scores
        return 0.0

def darth_extract_features(W_heap, ndis, nstep, firstNN, ninserts):
    """
    Table-1 feature set (11 features):
    ndis, inserts, nstep,
    firstNN, closestNN, furthestNN,
    avg, var, med, perc25, perc75
    """
    dists = np.array([-neg_d for (neg_d, _) in W_heap], dtype=np.float32)

    if dists.size == 0:
        return {
            "ndis": int(ndis),
            "inserts": int(ninserts),
            "nstep": int(nstep),
            "firstNN": float(firstNN),
            "closestNN": float("inf"),
            "furthestNN": float("inf"),
            "avg": float("inf"),
            "var": 0.0,
            "med": float("inf"),
            "perc25": float("inf"),
            "perc75": float("inf"),
        }

    return {
        "ndis": int(ndis),
        "inserts": int(ninserts),
        "nstep": int(nstep),
        "firstNN": float(firstNN),
        "closestNN": float(np.min(dists)),
        "furthestNN": float(np.max(dists)),
        "avg": float(np.mean(dists)),
        "var": float(np.var(dists)),
        "med": float(np.median(dists)),
        "perc25": float(np.percentile(dists, 25)),
        "perc75": float(np.percentile(dists, 75)),
    }

class HNSW_DARTH:

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
        self.use_heuristic = True 
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
            for lc in range(l+1):
                self.layers[lc][node_id] = set()
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
            neighbors = self._select_neighbors_heuristic_paper(
                        vec, W, layer=lc,
                        M=(self.M if lc > 0 else self.M0),
                        extend_candidates=True,
                        keep_pruned_connections=False
                         )
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
        
        while True:
            improved = False 
            neighbors = self.layers[lc].get(best,set())

            for nb in neighbors:
                d = self.dist(vec,self.vectors[nb])
                if d < best_dist:
                    best_dist = d
                    best = nb 
                    improved = True 
                    
            if not improved:
                break
        return best   

    #DARTH search_layer on layer 0 
    def _search_layer_darth(
        self,
        q_vec,
        ep_id: int,
        layer: int,
        efSearch: int,
        k: int,
        Rt: float,
        predictor,
        ipi: int = 200,
        mpi: int = 0,
    ):
    # If predictor missing, disable early stop (keeps baseline recall)
        if predictor is None:
            class _NoStop:
                def predict(self, feats): return 0.0
            predictor = _NoStop()

        NBL = ep_id
        firstNN = self.dist(q_vec, self.vectors[NBL])

        # resultSet: max-heap of size <= k  -> store (-dist, id)
        resultSet = [(-firstNN, NBL)]
        heapq.heapify(resultSet)

        # candidateQueue: min-heap -> store (dist, id)
        candidateQueue = [(firstNN, NBL)]
        heapq.heapify(candidateQueue)

        visited = {NBL}

        # counters (consistent with initialization above)
        ndis = 1        # computed firstNN
        nstep = 0
        inserts = 1     # inserted NBL into resultSet
        idis = 0

        pi = float(ipi)

        def get_maxdist_result():
            if len(resultSet) < k:
                return float("inf")
            return -resultSet[0][0]   # worst among top-k

        def try_add_result(node_id, d):
            nonlocal inserts
            if len(resultSet) < k:
                heapq.heappush(resultSet, (-d, node_id))
                inserts += 1
                return True
            if d < -resultSet[0][0]:
                heapq.heapreplace(resultSet, (-d, node_id))
                inserts += 1
                return True
            return False

        while candidateQueue:
            # line 11
            dc, c_id = heapq.heappop(candidateQueue)

            # common HNSW termination (keeps search sane)
            if dc > get_maxdist_result():
                break

            # line 12-13 (paper increments + "compute" cDis; dc already is cDis)
            ndis += 1
            idis += 1

            # line 14-17
            try_add_result(c_id, dc)

            # line 18-23
            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue
                visited.add(nb)

                nd = self.dist(q_vec, self.vectors[nb])
                ndis += 1

                if nd < get_maxdist_result() or len(candidateQueue) < efSearch:
                    heapq.heappush(candidateQueue, (nd, nb))

            # line 24-32 (paper-style)
            pi_int = max(1, int(round(pi)))
            if (idis % pi_int) == 0:
                feats = darth_extract_features(resultSet, ndis, nstep, firstNN, inserts)
                Rp = float(predictor.predict(feats))

                if Rp >= Rt:
                    break

                # adjust prediction interval (line 30)
                pi = mpi + (ipi - mpi) * (Rt - Rp)

                # clamp to [mpi, ipi] to avoid nonsense values
                pi = max(float(mpi), min(float(ipi), float(pi)))

                # reset interval counter (line 31)
                idis = 0

            # line 33
            nstep += 1

        # return resultSet (top-k)
        out = [(-neg_d, node_id) for (neg_d, node_id) in resultSet]
        out.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in out[:k]]


    #beam_search   
    def _search_layer(self, vec, ep_id: int, layer: int, ef: int):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if ef <= 0:
            return []

        visited = set()

        # C: min-heap (distance, id)
        C = []
        # W: max-heap via (-distance, id)
        W = []

        dist_ep = self.dist(vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        heapq.heappush(W, (-dist_ep, ep_id))

        while C:
            dist_c, c_id = heapq.heappop(C)

            worst_dist = -W[0][0]  # farthest among W (because max-heap with -dist)
            if dist_c > worst_dist:
                break

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                d = self.dist(vec, self.vectors[nb])

                if len(W) < ef:
                    # accept
                    visited.add(nb)
                    heapq.heappush(C, (d, nb))
                    heapq.heappush(W, (-d, nb))
                else:
                    worst_dist = -W[0][0]
                    if d < worst_dist:
                        # accept
                        visited.add(nb)
                        heapq.heappush(C, (d, nb))
                        # replace worst in W
                        heapq.heapreplace(W, (-d, nb))

        # return W sorted ascending by distance
        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in result]
    
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
    
    def _select_neighbors_heuristic_paper(self,q_vec,candidates,layer: int,M: int,extend_candidates: bool = True,keep_pruned_connections: bool = False):
        if M <= 0:
            return []

        # unique candidates
        W_set = set(candidates)

        # extend candidates (paper option)
        if extend_candidates:
            base = list(W_set)
            for e in base:
                if e not in self.layers[layer]:
                    continue
                for adj in self.layers[layer].get(e, set()):
                    W_set.add(adj)

        # sort candidates by distance to q_vec
        cand = [(self.dist(q_vec, self.vectors[e]), e) for e in W_set]
        cand.sort(key=lambda x: x[0])

        R = []          # selected ids
        discarded = []  # (d(q,e), e)

        for d_qe, e in cand:
            good = True
            for r in R:
                # diversification rule
                if self.dist(self.vectors[e], self.vectors[r]) < d_qe:
                    good = False
                    break

            if good:
                R.append(e)
                if len(R) == M:
                    break
            else:
                discarded.append((d_qe, e))

        if keep_pruned_connections and len(R) < M:
            discarded.sort(key=lambda x: x[0])
            for _, e in discarded:
                if e not in R:
                    R.append(e)
                    if len(R) == M:
                        break

        return R
         
    #search
    def query_darth(self, q_vec, k: int, efSearch: int, Rt: float,
                    predictor: DummyPredictor, ipi: int = 200, mpi: int = 0, use_mpi: bool = True):
        if self.entry_id is None:
            return []
        ep = self.entry_id
        L = self.maxlevel
        #greedy in the above layers
        for lc in range(L,0,-1):
            ep = self._search_layer_greedy(q_vec,ep,lc,ef=1)
        
        return self._search_layer_darth(
            q_vec, ep_id=ep, layer=0, efSearch=efSearch, k=k,
            Rt=Rt, predictor=predictor, ipi=ipi, mpi=mpi
        )

    #probability of levels
    # mL=l = 1/ln(M)
    def probab_levels(self,l): 
        U = max(self.rng.random(),1e-12)
        return int(-math.log(U)*l)
    
    def search_darth(self, Xq: np.ndarray, k: int, efSearch: int,
                     Rt: float, predictor, ipi: int = 200, mpi: int = 0, use_mpi: bool = True):
        Xq = np.asarray(Xq, dtype=np.float32)
        I = np.full((Xq.shape[0], k), -1, dtype=np.int32)
        D = np.full((Xq.shape[0], k), np.inf, dtype=np.float32)

        for i, q in enumerate(Xq):
            ids = self.query_darth(q, k=k, efSearch=efSearch, Rt=Rt,
                                  predictor=predictor, ipi=ipi, mpi=mpi, use_mpi=use_mpi)
            m = min(len(ids), k)
            if m > 0:
                I[i, :m] = ids[:m]
                for j in range(m):
                    D[i, j] = self.dist(q, self.vectors[int(I[i, j])])
        return D, I

    #so i can compare the results with faiss and hnswlib
    def search(self,Xq:np.ndarray,k:int , efSearch:int):
        Xq = np.asarray(Xq,dtype=np.float32)
        I = np.empty((Xq.shape[0],k),dtype=np.int32)
        D = np.empty((Xq.shape[0],k),dtype=np.float32)

        for i , q in enumerate(Xq):
            ids = self._query(q,K=k,numSearch=efSearch)
            #fill output
            I[i,:len(ids)] = ids
            if len(ids) < k:
                I[i,len(ids):] =-1
            
            #compute distances for returned ids
            for j in range(k):
                idx = I[i,j]
                if idx ==-1:
                    D[i,j] = np.inf
                else:
                    D[i,j] = self.dist(q,self.vectors[int(idx)])
        return D,I


    #calculate dist 
    def dist(self,a:np.ndarray,b:np.ndarray)->float:
        if self.metric =='l2':
            diff = a-b
            return float(np.dot(diff,diff))
        elif self.metric =='cosine':
            denom = (np.linalg.norm(a)*np.linalg.norm(b)) 
            if denom ==0:
                return 1.0
            return 1.0 - float(np.dot(a,b)/denom)
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
        neighbors = [x for x in neighbors if x != node_id]
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
        
    