import numpy as np
import heapq
import math
from statistics import NormalDist

class HNSW_AdaEF:
    """
    HNSW + Ada-ef

    Design:
    - Standard HNSW construction
    - Standard HNSW search helper: _search_layer(...)
    - Ada-ef only changes query-time layer-0 search:
        1) collect an initial distance list D
        2) estimate query difficulty
        3) estimate ef for a target recall
        4) continue normal HNSW search with that ef
    """

    def __init__(self, dim, M, efConstruction, metric="cosine", seed=42):
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

        # Ada-ef offline state
        self.adaef_bins = 5
        self.adaef_delta = 0.001
        self.adaef_sample_size = 200

        self.dataset_mean = None
        self.dataset_cov = None
        self.ef_estimation_table = {}
        self.wae_by_target = {}
        self.offline_ready = False

    # ---------------------------------------------------------
    # Basic index info
    # ---------------------------------------------------------
    def max_level(self) -> int:
        return self.maxlevel

    def levels(self) -> list:
        return list(range(self.maxlevel + 1))

    def entry_point(self):
        return self.entry_id

    # mL = 1 / ln(M)
    def probab_levels(self, l):
        U = max(self.rng.random(), 1e-12)
        return int(-math.log(U) * l)

    # ---------------------------------------------------------
    # Distance
    # ---------------------------------------------------------
    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 1.0
            return 1.0 - float(np.dot(a, b) / denom)
        else:
            raise ValueError("HNSW_AdaEF is intended for cosine distance.")

    # ---------------------------------------------------------
    # HNSW construction
    # ---------------------------------------------------------
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

        # standard HNSW construction search
        for lc in range(min(L, l), -1, -1):
            if node_id not in self.layers[lc]:
                self.layers[lc][node_id] = set()

            W = self._search_layer(vec, ep, lc, self.efConstruction)

            neighbors = self._select_neighbors_heuristic_paper(
                vec,
                W,
                layer=lc,
                M=(self.M if lc > 0 else self.M0),
                extend_candidates=True,
                keep_pruned_connections=False,
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

    # Standard HNSW beam search
    def _search_layer(self, vec, ep_id: int, layer: int, ef: int):
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []
        if ef <= 0:
            return []

        visited = set()
        C = []  # min-heap (distance, id)
        W = []  # max-heap via (-distance, id)

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

                visited.add(nb)
                d = self.dist(vec, self.vectors[nb])

                if len(W) < ef:
                    heapq.heappush(C, (d, nb))
                    heapq.heappush(W, (-d, nb))
                else:
                    worst_dist = -W[0][0]
                    if d < worst_dist:
                        heapq.heappush(C, (d, nb))
                        heapq.heapreplace(W, (-d, nb))

        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in result]

    # ---------------------------------------------------------
    # Neighbor selection / pruning
    # ---------------------------------------------------------
    def _select_neighbors_simple(self, vec, candidates, layer: int, Mmax: int):
        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) <= Mmax:
            return unique_candidates

        dist_list = []
        for nb in unique_candidates:
            d = self.dist(vec, self.vectors[nb])
            dist_list.append((d, nb))
        dist_list.sort(key=lambda x: x[0])

        return [nb for (d, nb) in dist_list[:Mmax]]

    def _select_neighbors_heuristic_paper(
        self,
        q_vec,
        candidates,
        layer: int,
        M: int,
        extend_candidates: bool = True,
        keep_pruned_connections: bool = False,
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

    def _prune_connections(self, node_id: int, layer: int, Mmax: int):
        neigh_set = self.layers[layer].get(node_id, set())
        if len(neigh_set) <= Mmax:
            return neigh_set

        neighbors = [x for x in neigh_set if x != node_id]
        q_vec = self.vectors[node_id]

        if getattr(self, "use_heuristic", False):
            new_neigh_list = self._select_neighbors_heuristic_paper(
                q_vec,
                neighbors,
                layer=layer,
                M=Mmax,
                extend_candidates=True,
                keep_pruned_connections=False,
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

    # ---------------------------------------------------------
    # Ada-ef offline helpers
    # ---------------------------------------------------------
    def _compute_dataset_statistics(self):
        X = np.vstack([self.vectors[i] for i in sorted(self.vectors.keys())]).astype(np.float64)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        Xn = X / norms

        self.dataset_mean = Xn.mean(axis=0)
        self.dataset_cov = np.cov(Xn, rowvar=False)

    def _estimate_fdl_params(self, q_vec):
        q = np.asarray(q_vec, dtype=np.float64)
        qn = q / max(np.linalg.norm(q), 1e-12)

        mu_cs = float(qn @ self.dataset_mean)
        var_cs = float(qn @ self.dataset_cov @ qn.T)
        var_cs = max(var_cs, 1e-12)

        # cosine distance = 1 - cosine similarity
        mu_cd = 1.0 - mu_cs
        sigma_cd = math.sqrt(var_cs)
        return mu_cd, sigma_cd

    def _compute_bins(self, mu, sigma):
        nd = NormalDist()
        thresholds = []

        for i in range(1, self.adaef_bins + 1):
            p = self.adaef_delta * i
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            theta_i = mu + sigma * nd.inv_cdf(p)
            thresholds.append(theta_i)

        return thresholds

    def _compute_query_score(self, q_vec, D):
        if len(D) == 0:
            return 0.0

        mu, sigma = self._estimate_fdl_params(q_vec)
        thresholds = self._compute_bins(mu, sigma)

        counts = [0] * self.adaef_bins
        for d in D:
            for i, th in enumerate(thresholds):
                if d <= th:
                    counts[i] += 1
                    break

        score = 0.0
        Dlen = float(len(D))
        for i, c_i in enumerate(counts, start=1):
            w_i = 100.0 * math.exp(-(i - 1))
            score += w_i * (c_i / Dlen)

        return score

    def _exact_knn_excluding_self(self, q_vec, qid, k):
        arr = []
        for nid, v in self.vectors.items():
            if nid == qid:
                continue
            arr.append((self.dist(q_vec, v), nid))
        arr.sort(key=lambda x: x[0])
        return [nid for (_, nid) in arr[:k]]

    def _recall_at_k(self, gt_ids, pred_ids, k):
        gt = set(gt_ids[:k])
        pr = set(pred_ids[:k])
        return len(gt.intersection(pr)) / max(k, 1)

    def _entry_after_upper_layers(self, q_vec):
        ep = self.entry_id
        L = self.maxlevel
        for lc in range(L, 0, -1):
            ep = self._search_layer_greedy(q_vec, ep, lc, ef=1)
        return ep

    def _two_hop_size(self, ep_id, layer=0):
        if ep_id not in self.layers[layer]:
            return 1

        reach = {ep_id}
        one_hop = set(self.layers[layer].get(ep_id, set()))
        reach.update(one_hop)

        for nb in one_hop:
            reach.update(self.layers[layer].get(nb, set()))

        return max(1, len(reach))

    def _collect_distance_list(self, q_vec, ep_id, layer=0):
        l = self._two_hop_size(ep_id, layer=layer)

        visited = set()
        C = []
        D = []

        dist_ep = self.dist(q_vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        D.append(dist_ep)

        while C and len(D) < l:
            dist_c, c_id = heapq.heappop(C)

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                visited.add(nb)
                d = self.dist(q_vec, self.vectors[nb])
                D.append(d)
                heapq.heappush(C, (d, nb))

                if len(D) >= l:
                    break

        return D

    def build_adaef_offline(self, k, target_recall=0.95, ef_values=None, sample_ids=None):
        """
        Offline stage:
        - compute dataset statistics
        - sample proxy queries
        - group queries by integer score
        - for each score group, evaluate recall across ef_values
        - store score -> [(ef, recall), ...]
        - compute WAE for the target recall
        """
        if len(self.vectors) == 0:
            raise ValueError("Index is empty")

        if ef_values is None:
            ef_values = [50, 75, 100, 150, 200, 300, 400, 500]

        self._compute_dataset_statistics()

        all_ids = sorted(self.vectors.keys())
        if sample_ids is None:
            sample_size = min(self.adaef_sample_size, len(all_ids))
            sample_ids = list(self.rng.choice(all_ids, size=sample_size, replace=False))

        score_groups = {}
        gt_cache = {}

        for qid in sample_ids:
            q_vec = self.vectors[qid]
            gt_ids = self._exact_knn_excluding_self(q_vec, qid, k)
            gt_cache[qid] = gt_ids

            ep = self._entry_after_upper_layers(q_vec)
            D = self._collect_distance_list(q_vec, ep, layer=0)

            score = self._compute_query_score(q_vec, D)
            score_int = int(score)
            score_groups.setdefault(score_int, []).append(qid)

        ef_table = {}
        total_queries = 0
        weighted_sum = 0.0

        for score_int, qids in score_groups.items():
            pairs = []

            for ef in ef_values:
                recalls = []
                for qid in qids:
                    q_vec = self.vectors[qid]
                    found = self._query_standard(q_vec, K=k, numSearch=ef, exclude_id=qid)
                    recall = self._recall_at_k(gt_cache[qid], found, k)
                    recalls.append(recall)

                avg_recall = float(np.mean(recalls)) if recalls else 0.0
                pairs.append((ef, avg_recall))

            ef_table[score_int] = pairs

            chosen_ef = pairs[-1][0]
            for ef, rec in pairs:
                if rec >= target_recall:
                    chosen_ef = ef
                    break

            total_queries += len(qids)
            weighted_sum += len(qids) * chosen_ef

        self.ef_estimation_table = ef_table
        self.wae_by_target[target_recall] = weighted_sum / max(total_queries, 1)
        self.offline_ready = True

    def estimate_ef(self, q_vec, D, target_recall):
        if not self.offline_ready:
            raise RuntimeError("Call build_adaef_offline(...) first")

        score = self._compute_query_score(q_vec, D)
        score_int = int(score)

        if score_int in self.ef_estimation_table:
            row = self.ef_estimation_table[score_int]
        else:
            available = sorted(self.ef_estimation_table.keys())
            nearest = min(available, key=lambda x: abs(x - score_int))
            row = self.ef_estimation_table[nearest]

        ef = row[-1][0]
        for EF, Recall in row:
            if Recall >= target_recall:
                ef = EF
                break

        wae = self.wae_by_target.get(target_recall, ef)
        return max(int(ef), int(math.ceil(wae)))

    # ---------------------------------------------------------
    # Query methods
    # ---------------------------------------------------------
    def _query_standard(self, q_vec, K, numSearch, exclude_id=None):
        if self.entry_id is None:
            return []

        ep = self._entry_after_upper_layers(q_vec)
        W = self._search_layer(q_vec, ep, 0, numSearch)

        if exclude_id is not None:
            W = [x for x in W if x != exclude_id]

        return W[:K]

    def _search_layer_adaef(self, q_vec, ep_id: int, layer: int, target_recall: float):
        """
        Ada-ef online stage:
        - start with ef = inf
        - collect D
        - estimate ef
        - continue standard HNSW exploration with bounded ef
        """
        if layer < 0 or layer >= len(self.layers) or len(self.layers[layer]) == 0:
            return []
        if ep_id not in self.vectors:
            return []

        visited = set()
        C = []
        W = []

        dist_ep = self.dist(q_vec, self.vectors[ep_id])
        visited.add(ep_id)
        heapq.heappush(C, (dist_ep, ep_id))
        heapq.heappush(W, (-dist_ep, ep_id))

        D = [dist_ep]
        l = self._two_hop_size(ep_id, layer=layer)
        ef = math.inf
        ef_estimated = False

        def threshold():
            if ef == math.inf:
                return float("inf")
            return -W[0][0]

        while C:
            dist_c, c_id = heapq.heappop(C)

            if ef != math.inf and dist_c > threshold():
                break

            for nb in self.layers[layer].get(c_id, set()):
                if nb in visited:
                    continue

                visited.add(nb)
                d = self.dist(q_vec, self.vectors[nb])

                # Phase 1: collect D with ef = infinity
                if ef == math.inf:
                    D.append(d)
                    heapq.heappush(C, (d, nb))
                    heapq.heappush(W, (-d, nb))

                    if len(D) >= l and not ef_estimated:
                        ef = self.estimate_ef(q_vec, D, target_recall)
                        ef_estimated = True

                        if len(W) > ef:
                            best = [(-neg_d, nid) for (neg_d, nid) in W]
                            best.sort(key=lambda x: x[0])
                            best = best[:ef]
                            W = [(-d2, nid) for (d2, nid) in best]
                            heapq.heapify(W)

                # Phase 2: continue standard HNSW with estimated ef
                else:
                    if len(W) < ef:
                        heapq.heappush(C, (d, nb))
                        heapq.heappush(W, (-d, nb))
                    else:
                        worst_dist = -W[0][0]
                        if d < worst_dist:
                            heapq.heappush(C, (d, nb))
                            heapq.heapreplace(W, (-d, nb))

        result = [(-neg_d, node_id) for (neg_d, node_id) in W]
        result.sort(key=lambda x: x[0])
        return [node_id for (_, node_id) in result]

    def _query_adaef(self, q_vec, K, target_recall):
        if self.entry_id is None:
            return []
        if not self.offline_ready:
            raise RuntimeError("Call build_adaef_offline(...) first")

        ep = self._entry_after_upper_layers(q_vec)
        W = self._search_layer_adaef(q_vec, ep, 0, target_recall)
        return W[:K]

    def search(self, Xq: np.ndarray, k: int, target_recall: float):
        """
        Public Ada-ef query API.
        Instead of efSearch, user provides target_recall.
        """
        Xq = np.asarray(Xq, dtype=np.float32)
        I = np.empty((Xq.shape[0], k), dtype=np.int32)
        D = np.empty((Xq.shape[0], k), dtype=np.float32)

        for i, q in enumerate(Xq):
            ids = self._query_adaef(q, K=k, target_recall=target_recall)

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