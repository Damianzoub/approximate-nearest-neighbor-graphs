import numpy as np 
import heapq
import math

class DummyPredictor:
    def predict(self, feats):
        # Dummy predictor that returns random scores
        return 0.0
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
    

    def _search_base_layer_darth(self, q, ep_id, layer, efSearch, k, Rt, predictor, ipi, mpi):
        """
        DARTH Algorithm 1 (paper-faithful control flow)

        resultSet: size k (max-heap via (-dist, id))
        candidateQueue: min-heap (dist, id), admission rule uses efSearch
        """
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if efSearch <= 0 or k <= 0:
            return []

        # ---------- helpers ----------
        def rs_maxdist(rs_heap):
            # GetMaxDistance(resultSet)
            if len(rs_heap) < k:
                return float("inf")
            return -rs_heap[0][0]

        def rs_try_insert(rs_heap, d, idx):
            """
            Insert into resultSet if it improves.
            returns True iff resultSet changed (counts as an 'insert' update).
            """
            if len(rs_heap) < k:
                heapq.heappush(rs_heap, (-d, idx))
                return True
            if d < -rs_heap[0][0]:
                heapq.heapreplace(rs_heap, (-d, idx))
                return True
            return False

        # ---------- init ----------
        visited = set([ep_id])

        # candidateQueue (min-heap)
        C = []

        # resultSet (max-heap via -dist), size k
        R = []

        # counters (paper semantics)
        ndis = 0       # total distance computations
        idis = 0       # distance computations since last prediction
        nstep = 0      # number of while-loop iterations
        inserts = 0    # number of updates to resultSet

        # line: compute firstNN = Distance(q, ep)  (+distance counters)
        firstNN = self.dist(q, self.vectors[ep_id]); ndis += 1; idis += 1

        # resultSet <- {ep}, candidateQueue <- {ep}
        heapq.heappush(R, (-firstNN, ep_id)); inserts += 1
        heapq.heappush(C, (firstNN, ep_id))

        # prediction interval
        mpi = max(1, int(mpi))
        ipi = max(mpi, int(ipi))
        pi = ipi  # start at ipi (paper uses ipi initially)

        # ---------- main loop ----------
        while C:
            nstep += 1

            # extract closest candidate
            _, c = heapq.heappop(C)

            # paper explicitly computes cDis = Distance(q,c)
            cDis = self.dist(q, self.vectors[c]); ndis += 1; idis += 1

            # if cDis < GetMaxDistance(resultSet): insert c into resultSet
            if cDis < rs_maxdist(R):
                if rs_try_insert(R, cDis, c):
                    inserts += 1

            # explore neighbors of c
            for nb in self.layers[layer].get(c, set()):
                if nb in visited:
                    continue
                visited.add(nb)

                # nDis = Distance(q, nb)
                nDis = self.dist(q, self.vectors[nb]); ndis += 1; idis += 1

                # if nDis < GetMaxDistance(resultSet): insert nb into resultSet
                if nDis < rs_maxdist(R):
                    if rs_try_insert(R, nDis, nb):
                        inserts += 1

                # if nDis < maxDist(resultSet) OR |candidateQueue| < efSearch: push
                if (nDis < rs_maxdist(R)) or (len(C) < efSearch):
                    heapq.heappush(C, (nDis, nb))

                # ---------- DARTH predictor trigger ----------
                # Paper condition is "every pi distance computations"
                if idis >= pi:
                    feats = darth_extract_features(
                        W_heap=R,        # IMPORTANT: features from resultSet (size k)
                        ndis=ndis,
                        nstep=nstep,
                        firstNN=firstNN,
                        ninserts=inserts
                    )
                    Rp = float(predictor.predict(feats))

                    # early terminate immediately
                    if Rp >= Rt:
                        res = [(-neg_d, idx) for (neg_d, idx) in R]
                        res.sort(key=lambda x: x[0])
                        return [idx for (_, idx) in res[:k]]

                    # adaptive interval update
                    pi = int(mpi + (ipi - mpi) * (Rt - Rp))
                    pi = max(mpi, min(ipi, pi))
                    idis = 0

            # (optional) standard “break” condition is implicit in paper via maxDist checks
            # We keep loop going as long as candidateQueue not empty.

        # natural termination
        res = [(-neg_d, idx) for (neg_d, idx) in R]
        res.sort(key=lambda x: x[0])
        return [idx for (_, idx) in res[:k]]

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
        cand = [(self.dist(q_vec, self.vectors[e]), e) for e in W_set if e in self.vectors]
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
    def search(self, Xq, k, efSearch, Rt=0.95, ipi=200, mpi=20, predictor=None):
        Xq = np.asarray(Xq, dtype=np.float32)
        I = np.empty((Xq.shape[0], k), dtype=np.int32)
        D = np.empty((Xq.shape[0], k), dtype=np.float32)

        if self.entry_id is None:
            I.fill(-1)
            D.fill(np.inf)
            return D, I

        if predictor is None:
            predictor = DummyPredictor()

        for i, q in enumerate(Xq):
            ids = self._query(q, k=k, efSearch=efSearch, Rt=Rt, ipi=ipi, mpi=mpi, predictor=predictor)

            I[i, :len(ids)] = ids
            if len(ids) < k:
                I[i, len(ids):] = -1

            for j in range(k):
                idx = I[i, j]
                D[i, j] = np.inf if idx == -1 else self.dist(q, self.vectors[int(idx)])

        return D, I

    def _query(self, q, k, efSearch, Rt=0.95, ipi=200, mpi=20, predictor=None):
        if self.entry_id is None:
            return []

        if predictor is None:
            predictor = DummyPredictor()

        ep = self.entry_id
        for lc in range(self.maxlevel, 0, -1):
            ep = self._search_layer_greedy(q, ep, lc, ef=1)

        return self._search_base_layer_darth(
            q=q, ep_id=ep, layer=0,
            efSearch=int(efSearch),
            k=int(k),
            Rt=float(Rt),
            predictor=predictor,
            ipi=int(ipi),
            mpi=int(mpi),
        )

    #probability of levels
    # mL=l = 1/ln(M)
    def probab_levels(self,l): 
        U = max(self.rng.random(),1e-12)
        return int(-math.log(U)*l)
    
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