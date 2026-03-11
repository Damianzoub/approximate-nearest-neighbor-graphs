import numpy as np
import heapq
import math


class HNSW_PiP:
    def __init__(self, dim, M, efConstruction, metric='l2', seed=42,
                 pip_gamma=95.0, pip_delta=20):
        self.dim = dim
        self.M = int(M)
        self.M0 = 2 * int(M)
        self.efConstruction = int(efConstruction)
        self.layers = []
        self.vectors = {}
        self.metric = metric
        self.maxlevel = -1
        self.rng = np.random.default_rng(seed)
        self.entry_id = None
        self.mL = 1.0 / math.log(self.M)
        self.use_heuristic = True

        # PiP paper parameters
        self.pip_gamma = float(pip_gamma)   # γ
        self.pip_delta = int(pip_delta)     # Δ

    """
    From paper algorithm 1: full insertion
               algorithm 2: searching layer to get candidate list
               algorithm 3/4: selecting M neighbors for connection

    PiP modification:
    - construction remains standard HNSW
    - query-time search at layer 0 uses patience-based early termination
    """

    def max_level(self) -> int:
        return self.maxlevel

    def levels(self) -> list:
        return list(range(self.maxlevel + 1))

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
        if l > self.maxlevel:
            for _ in range(self.maxlevel + 1, l + 1):
                self.layers.append({})
            self.maxlevel = l

        ep = self.entry_id
        L = old_top

        # greedy descent on upper layers
        for lc in range(L, l, -1):
            ep = self._search_layer_greedy(vec, ep, lc, ef=1)

        # standard HNSW construction on layers min(L,l) ... 0
        for lc in range(min(L, l), -1, -1):
            if node_id not in self.layers[lc]:
                self.layers[lc][node_id] = set()

            W = self._search_layer_standard(vec, ep, lc, self.efConstruction)

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
                    pruned = self._prune_connections(nb, lc, Mmax)
                    self.layers[lc][nb] = pruned

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
            neighbors = self.layers[lc].get(best, set())

            for nb in neighbors:
                d = self.dist(vec, self.vectors[nb])
                if d < best_dist:
                    best_dist = d
                    best = nb
                    improved = True

            if not improved:
                break

        return best

    def _topk_ids_from_heap(self, heap_obj, k: int):
        if k <= 0 or len(heap_obj) == 0:
            return []
        items = [(-neg_d, node_id) for (neg_d, node_id) in heap_obj]
        items.sort(key=lambda x: x[0])  # ascending distance
        return [node_id for (_, node_id) in items[:k]]

    # Standard HNSW / HNSW* layer search
    def _search_layer_standard(self, vec, ep_id: int, layer: int, ef: int, k_thresh: int = None):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if ef <= 0:
            return []

        visited = set()
        C = []  # min-heap (distance, id)
        W = []  # max-heap via (-distance, id), size <= ef

        use_k_thresh = (k_thresh is not None and k_thresh > 0)
        Kheap = []  # max-heap via (-distance, id), size <= k_thresh

        def threshold():
            if not use_k_thresh:
                return -W[0][0]
            if len(Kheap) < k_thresh:
                return float("inf")
            return -Kheap[0][0]

        def try_add_topk(d, nb):
            if not use_k_thresh:
                return
            if len(Kheap) < k_thresh:
                heapq.heappush(Kheap, (-d, nb))
            else:
                if d < -Kheap[0][0]:
                    heapq.heapreplace(Kheap, (-d, nb))

        dist_ep = self.dist(vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        heapq.heappush(W, (-dist_ep, ep_id))
        try_add_topk(dist_ep, ep_id)

        while C:
            dist_c, c_id = heapq.heappop(C)

            if dist_c > threshold():
                break

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                visited.add(nb)
                d = self.dist(vec, self.vectors[nb])

                if len(W) < ef or d < threshold():
                    heapq.heappush(C, (d, nb))

                    if len(W) < ef:
                        heapq.heappush(W, (-d, nb))
                    else:
                        if d < -W[0][0]:
                            heapq.heapreplace(W, (-d, nb))

                    try_add_topk(d, nb)

        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in result]

    # PiP layer search: same HNSW traversal, but with patience-based early stop
    def _search_layer_pip(self, vec, ep_id: int, layer: int, ef: int, k: int):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if ef <= 0:
            return []
        if k <= 0:
            return []

        visited = set()
        C = []   # min-heap of candidates to expand
        W = []   # max-heap via (-distance, id), size <= ef
        Kheap = []  # current top-k result set for PiP saturation tracking

        def threshold():
            return -W[0][0]

        def try_add_topk(d, nb):
            if len(Kheap) < k:
                heapq.heappush(Kheap, (-d, nb))
            else:
                if d < -Kheap[0][0]:
                    heapq.heapreplace(Kheap, (-d, nb))

        dist_ep = self.dist(vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        heapq.heappush(W, (-dist_ep, ep_id))
        try_add_topk(dist_ep, ep_id)

        prev_topk = None
        saturation_counter = 0

        while C:
            dist_c, c_id = heapq.heappop(C)

            if dist_c > threshold():
                break

            # expand one candidate
            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                visited.add(nb)
                d = self.dist(vec, self.vectors[nb])

                if len(W) < ef or d < threshold():
                    heapq.heappush(C, (d, nb))

                    if len(W) < ef:
                        heapq.heappush(W, (-d, nb))
                    else:
                        if d < -W[0][0]:
                            heapq.heapreplace(W, (-d, nb))

                    try_add_topk(d, nb)

            # PiP paper criterion:
            # phi_h,l(q) = 100 * |N_{h-1,l}(q) ∩ N_{h,l}(q)| / k
            # stop if phi_h,l(q) >= gamma for delta consecutive iterations
            curr_topk = self._topk_ids_from_heap(Kheap, k)

            if len(curr_topk) == k:
                curr_topk_set = set(curr_topk)

                if prev_topk is not None:
                    phi_h = 100.0 * len(curr_topk_set.intersection(prev_topk)) / k

                    if phi_h >= self.pip_gamma:
                        saturation_counter += 1
                    else:
                        saturation_counter = 0

                    if saturation_counter >= self.pip_delta:
                        break

                prev_topk = curr_topk_set

        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in result]

    def _select_neighbors_simple(self, vec, candidates, layer: int, Mmax: int):
        unique_candidates = list(dict.fromkeys(candidates))

        if len(unique_candidates) <= Mmax:
            return unique_candidates

        dist_list = []
        for nb in unique_candidates:
            d = self.dist(vec, self.vectors[nb])
            dist_list.append((d, nb))

        dist_list.sort(key=lambda x: x[0])
        selected = [nb for (d, nb) in dist_list[:Mmax]]
        return selected

    def _select_neighbors_heuristic_paper(
        self,
        q_vec,
        candidates,
        layer: int,
        M: int,
        extend_candidates: bool = True,
        keep_pruned_connections: bool = False
    ):
        if M <= 0:
            return []

        W_set = set(candidates)

        if extend_candidates:
            base = list(W_set)
            for e in base:
                if e not in self.layers[layer]:
                    continue
                for adj in self.layers[layer].get(e, set()):
                    W_set.add(adj)

        cand = [(self.dist(q_vec, self.vectors[e]), e) for e in W_set]
        cand.sort(key=lambda x: x[0])

        R = []
        discarded = []

        for d_qe, e in cand:
            good = True
            for r in R:
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

    def _query(self, q_vec, K, numSearch, use_hnsw_star: bool = False):
        if self.entry_id is None:
            return []

        ep = self.entry_id
        L = self.maxlevel

        # upper layers: standard greedy descent
        for lc in range(L, 0, -1):
            ep = self._search_layer_greedy(q_vec, ep, lc, ef=1)

        # bottom layer: PiP search
        W = self._search_layer_pip(q_vec, ep, 0, numSearch, k=K)

        return W[:K]

    def probab_levels(self, l):
        U = max(self.rng.random(), 1e-12)
        return int(-math.log(U) * l)

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

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == 'l2':
            diff = a - b
            return float(np.dot(diff, diff))
        elif self.metric == 'cosine':
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 1.0
            return 1.0 - float(np.dot(a, b) / denom)
        else:
            raise ValueError("Unknown metric")

    def entry_point(self):
        return self.entry_id

    def _prune_connections(self, node_id: int, layer: int, Mmax: int):
        neigh_set = self.layers[layer].get(node_id, set())
        if len(neigh_set) <= Mmax:
            return neigh_set

        neighbors = list(neigh_set)
        neighbors = [x for x in neighbors if x != node_id]
        q_vec = self.vectors[node_id]

        if getattr(self, "use_heuristic", False):
            new_neigh_list = self._select_neighbors_heuristic_paper(
                q_vec,
                neighbors,
                layer=layer,
                M=Mmax,
                extend_candidates=True,
                keep_pruned_connections=False
            )
        else:
            new_neigh_list = self._select_neighbors_simple(
                q_vec, neighbors, layer, Mmax
            )

        new_neigh_set = set(new_neigh_list)
        removed = neigh_set - new_neigh_set

        for nb in removed:
            if nb in self.layers[layer]:
                self.layers[layer][nb].discard(node_id)

        self.layers[layer][node_id] = new_neigh_set
        return new_neigh_set