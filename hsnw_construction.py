import numpy as np
import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ----------------------------
# DARTH minimal definitions
# ----------------------------

@dataclass
class DarthParams:
    k: int
    Rt: float
    efSearch: int
    ipi: int
    mpi: int


class RecallPredictor:
    """Interface-like base class."""
    def predict_one(self, features11: np.ndarray) -> float:
        raise NotImplementedError


class DummyPredictor(RecallPredictor):
    """
    Simple dummy predictor so you can test DARTH integration.
    """
    def predict_one(self, features11: np.ndarray) -> float:
        ndis = float(features11[1])
        rp = 1.0 - np.exp(-ndis / 200.0)
        return float(np.clip(rp, 0.0, 1.0))


class HNSW_NEW:
    def __init__(self, dim, M, efConstruction, metric="l2", seed=42):
        self.dim = int(dim)
        self.M = int(M)
        self.M0 = 2 * int(M)
        self.efConstruction = int(efConstruction)

        self.layers = []          # list of dict: node_id -> set(neighbors)
        self.vectors = {}         # node_id -> np.ndarray

        self.metric = metric
        self.maxlevel = -1
        self.entry_id = None

        self.rng = np.random.default_rng(seed)
        self.mL = 1.0 / math.log(self.M)

        self.use_heuristic = True


    def max_level(self) -> int:
        return self.maxlevel

    def levels(self) -> list:
        return list(range(self.maxlevel + 1))

    def probab_levels(self, l: float) -> int:
        U = max(self.rng.random(), 1e-12)
        return int(-math.log(U) * l)

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "l2":
            diff = a - b
            return float(np.dot(diff, diff))  # squared L2
        elif self.metric == "cosine":
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                return 1.0
            return 1.0 - float(np.dot(a, b) / denom)
        else:
            raise ValueError("Unknown metric")


    def _insert_(self, vector, node_id=None):
        vec = np.asarray(vector, dtype=float)
        if vec.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vec.shape}")

        if node_id is None:
            node_id = len(self.vectors)
        if node_id in self.vectors:
            raise ValueError(f"Node id {node_id} already exists")

        self.vectors[node_id] = vec
        l = self.probab_levels(self.mL)

        # first node
        if self.entry_id is None:
            for _ in range(l + 1):
                self.layers.append({})
            for lc in range(l + 1):
                self.layers[lc][node_id] = set()
            self.entry_id = node_id
            self.maxlevel = l
            return node_id

        old_top = self.maxlevel

        # grow layers if needed
        if l > self.maxlevel:
            for _ in range(self.maxlevel + 1, l + 1):
                self.layers.append({})
            self.maxlevel = l

        ep = self.entry_id
        L = old_top

        # phase 1: greedy descent from top down to l+1
        for lc in range(L, l, -1):
            ep = self._search_layer_greedy(vec, ep, lc, ef=1)

        # phase 2: connect from min(L,l) down to 0
        for lc in range(min(L, l), -1, -1):
            if node_id not in self.layers[lc]:
                self.layers[lc][node_id] = set()

            W = self._search_layer(vec, ep, lc, self.efConstruction)

            Mmax = self.M if lc > 0 else self.M0
            neighbors = self.select_neighbors_heuristic(vec, W, lc, Mmax)

            for nb in neighbors:
                if nb not in self.layers[lc]:
                    self.layers[lc][nb] = set()

                self.layers[lc][node_id].add(nb)
                self.layers[lc][nb].add(node_id)

                if len(self.layers[lc][nb]) > Mmax:
                    self.layers[lc][nb] = self._prune_connections(nb, lc, Mmax)

            if len(W) > 0:
                ep = W[0]

        if l > old_top:
            self.entry_id = node_id

        return node_id
    
    def _search_layer_greedy(self, vec, curr_entryPointID: int, lc: int, ef: int = 1) -> int:
        best = curr_entryPointID
        best_dist = self.dist(vec, self.vectors[best])

        while True:
            improved = False
            for nb in self.layers[lc].get(best, set()):
                d = self.dist(vec, self.vectors[nb])
                if d < best_dist:
                    best_dist = d
                    best = nb
                    improved = True
            if not improved:
                break
        return best

    def _search_layer(self, vec, ep_id: int, layer: int, ef: int):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if ef <= 0:
            return []

        visited = set()
        C = []  # min-heap (dist, id)
        W = []  # max-heap by (-dist, id)

        dist_ep = self.dist(vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        heapq.heappush(W, (-dist_ep, ep_id))

        while C:
            dist_c, c_id = heapq.heappop(C)
            worst_dist = -W[0][0]
            if dist_c > worst_dist:
                break

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                d = self.dist(vec, self.vectors[nb])

                if len(W) < ef:
                    visited.add(nb)
                    heapq.heappush(C, (d, nb))
                    heapq.heappush(W, (-d, nb))
                else:
                    worst_dist = -W[0][0]
                    if d < worst_dist:
                        visited.add(nb)
                        heapq.heappush(C, (d, nb))
                        heapq.heapreplace(W, (-d, nb))

        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (dist, node_id) in result]

    # ----------------------------
    # Neighbor selection
    # ----------------------------

    def _select_neighbors_simple(self, vec, candidates, layer: int, Mmax: int):
        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) <= Mmax:
            return unique_candidates

        dist_list = [(self.dist(vec, self.vectors[nb]), nb) for nb in unique_candidates]
        dist_list.sort(key=lambda x: x[0])
        return [nb for (d, nb) in dist_list[:Mmax]]

    def select_neighbors_heuristic(self, vec, candidates, layer: int, Mmax: int):
        # your “light” heuristic
        unique_candidates = list(dict.fromkeys(candidates))
        cand_with_dist = [(self.dist(vec, self.vectors[nb]), nb) for nb in unique_candidates]
        cand_with_dist.sort(key=lambda x: x[0])

        R = []  # list of (d_e, e)
        for d_e, e in cand_with_dist:
            if len(R) >= Mmax:
                break
            good = True
            for d_r, r in R:
                if self.dist(self.vectors[e], self.vectors[r]) < d_e:
                    good = False
                    break
            if good:
                R.append((d_e, e))
        return [e for (d_e, e) in R]

    def _select_neighbors_heuristic_paper(self, vec, candidates, layer: int, M: int,
                                         extend_candidates: bool = True,
                                         keep_pruned_connections: bool = False):
        W_set = set(candidates)

        if extend_candidates:
            base = list(W_set)
            for e in base:
                for eadj in self.layers[layer].get(e, set()):
                    W_set.add(eadj)

        cand = [(self.dist(vec, self.vectors[e]), e) for e in W_set]
        cand.sort(key=lambda x: x[0])

        R = []
        discarded = []
        for d_e, e in cand:
            good = True
            for r in R:
                if self.dist(self.vectors[e], self.vectors[r]) < d_e:
                    good = False
                    break
            if good:
                R.append(e)
                if len(R) == M:
                    break
            else:
                discarded.append((d_e, e))

        if keep_pruned_connections and len(R) < M:
            discarded.sort(key=lambda x: x[0])
            for _, e in discarded:
                if e not in R:
                    R.append(e)
                    if len(R) == M:
                        break
        return R

    def _prune_connections(self, node_id: int, layer: int, Mmax: int):
        neigh_set = self.layers[layer].get(node_id, set())
        if len(neigh_set) <= Mmax:
            return neigh_set

        neighbors = list(neigh_set)
        q_vec = self.vectors[node_id]

        if self.use_heuristic:
            new_neigh_list = self._select_neighbors_heuristic_paper(
                q_vec, neighbors, layer=layer, M=Mmax,
                extend_candidates=True, keep_pruned_connections=False
            )
        else:
            new_neigh_list = self._select_neighbors_simple(q_vec, neighbors, layer, Mmax)

        new_neigh_set = set(new_neigh_list)
        removed = neigh_set - new_neigh_set

        for nb in removed:
            if nb in self.layers[layer]:
                self.layers[layer][nb].discard(node_id)

        self.layers[layer][node_id] = new_neigh_set
        return new_neigh_set

    # ----------------------------
    # Normal HNSW query (baseline)
    # ----------------------------

    def _query(self, q_vec, K: int, numSearch: int):
        if self.entry_id is None:
            return []
        ep = self.entry_id
        L = self.maxlevel

        for lc in range(L, 0, -1):
            ep = self._search_layer_greedy(q_vec, ep, lc, ef=1)

        W = self._search_layer(q_vec, ep, 0, numSearch)
        return W[:K]

    def search(self, Xq: np.ndarray, k: int, efSearch: int):
        Xq = np.asarray(Xq, dtype=np.float32)
        I = np.empty((Xq.shape[0], k), dtype=np.int32)
        D = np.empty((Xq.shape[0], k), dtype=np.float32)

        for i, q in enumerate(Xq):
            ids = self._query(q, K=k, numSearch=efSearch)

            I[i, :len(ids)] = ids
            if len(ids) < k:
                I[i, len(ids):] = -1

            for j in range(k):
                idx = I[i, j]
                if idx == -1:
                    D[i, j] = np.inf
                else:
                    D[i, j] = self.dist(q, self.vectors[int(idx)])
        return D, I

    # ----------------------------
    # DARTH helpers (staticmethod)
    # ----------------------------

    @staticmethod
    def _compute_nn_stats(dists):
        if not dists:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(dists, dtype=np.float32)
        return (
            float(arr.mean()),
            float(arr.var()),
            float(np.median(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        )

    @staticmethod
    def _darth_update_pi(ipi: int, mpi: int, Rt: float, Rp: float) -> int:
        val = mpi + (ipi - mpi) * (Rt - Rp)
        val = max(mpi, min(ipi, val))
        return int(max(1, round(val)))

    @staticmethod
    def _heap_to_sorted_ids(result_heap):
        tmp = [(-nd, nid) for (nd, nid) in result_heap]
        tmp.sort(key=lambda x: x[0])
        return [nid for (_, nid) in tmp]

    @staticmethod
    def _heap_to_sorted_dists(result_heap):
        tmp = [(-nd, nid) for (nd, nid) in result_heap]
        tmp.sort(key=lambda x: x[0])
        return [d for (d, _) in tmp]

    @staticmethod
    def _resultset_worst_dist(result_heap, k: int) -> float:
        if len(result_heap) < k:
            return float("inf")
        return -result_heap[0][0]

    @staticmethod
    def _try_insert_result(result_heap, d: float, nid: int, k: int) -> bool:
        if len(result_heap) < k:
            heapq.heappush(result_heap, (-d, nid))
            return True
        worst = -result_heap[0][0]
        if d < worst:
            heapq.heapreplace(result_heap, (-d, nid))
            return True
        return False

    # ----------------------------
    # DARTH base layer search
    # ----------------------------

    def _search_layer_darth_base(self, q_vec, ep_id: int, params: DarthParams, model: RecallPredictor):
        layer = 0
        if ep_id not in self.vectors:
            return []
        if layer >= len(self.layers):
            return []

        visited = set()
        candidateQ = []   # min-heap (d, id)
        resultSet = []    # max-heap (-d, id), size k

        ndis = 0
        nstep = 0
        ninserts = 0
        idis = 0

        d0 = self.dist(q_vec, self.vectors[ep_id]); ndis += 1; idis += 1
        firstNN = d0

        visited.add(ep_id)
        heapq.heappush(candidateQ, (d0, ep_id))
        if self._try_insert_result(resultSet, d0, ep_id, params.k):
            ninserts += 1

        pi = max(1, int(params.ipi))

        while candidateQ:
            d_c, c_id = heapq.heappop(candidateQ)
            nstep += 1

            worst_rs = self._resultset_worst_dist(resultSet, params.k)
            if d_c > worst_rs:
                break

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                d = self.dist(q_vec, self.vectors[nb]); ndis += 1; idis += 1

                pushed = False
                worst_rs = self._resultset_worst_dist(resultSet, params.k)

                # soft cap using efSearch, but still allow improvements
                if len(resultSet) < params.k or d < worst_rs or len(candidateQ) < params.efSearch:
                    heapq.heappush(candidateQ, (d, nb))
                    pushed = True

                inserted = self._try_insert_result(resultSet, d, nb, params.k)
                if inserted:
                    ninserts += 1

                if pushed or inserted:
                    visited.add(nb)

                # predictor call every pi distance calcs
                if idis >= pi:
                    idis = 0

                    dists_sorted = self._heap_to_sorted_dists(resultSet)
                    closestNN = dists_sorted[0] if dists_sorted else float("inf")
                    furthestNN = dists_sorted[-1] if dists_sorted else float("inf")
                    avg, var, med, p25, p75 = self._compute_nn_stats(dists_sorted)

                    x = np.array([
                        float(nstep),
                        float(ndis),
                        float(ninserts),
                        float(firstNN),
                        float(closestNN),
                        float(furthestNN),
                        float(avg),
                        float(var),
                        float(med),
                        float(p25),
                        float(p75),
                    ], dtype=np.float32)

                    Rp = float(model.predict_one(x))

                    if Rp >= params.Rt:
                        return self._heap_to_sorted_ids(resultSet)[:params.k]

                    pi = self._darth_update_pi(params.ipi, params.mpi, params.Rt, Rp)

        return self._heap_to_sorted_ids(resultSet)[:params.k]

    # ----------------------------
    # Public DARTH query + batch search
    # ----------------------------

    def query_darth(self, q_vec, k: int, efSearch: int, Rt: float, model: RecallPredictor, ipi: int = 32, mpi: int = 4):
        if self.entry_id is None:
            return []

        ep = self.entry_id
        L = self.maxlevel

        for lc in range(L, 0, -1):
            ep = self._search_layer_greedy(q_vec, ep, lc, ef=1)

        params = DarthParams(k=k, Rt=Rt, efSearch=efSearch, ipi=ipi, mpi=mpi)
        return self._search_layer_darth_base(q_vec, ep, params, model)

    def search_darth(self, Xq: np.ndarray, k: int, efSearch: int,
                     Rt: float, model: RecallPredictor, ipi: int = 32, mpi: int = 4):
        Xq = np.asarray(Xq, dtype=np.float32)
        I = np.empty((Xq.shape[0], k), dtype=np.int32)
        D = np.empty((Xq.shape[0], k), dtype=np.float32)

        for i, q in enumerate(Xq):
            ids = self.query_darth(q, k=k, efSearch=efSearch, Rt=Rt, model=model, ipi=ipi, mpi=mpi)

            I[i, :len(ids)] = ids
            if len(ids) < k:
                I[i, len(ids):] = -1

            for j in range(k):
                idx = I[i, j]
                if idx == -1:
                    D[i, j] = np.inf
                else:
                    D[i, j] = self.dist(q, self.vectors[int(idx)])

        return D, I

