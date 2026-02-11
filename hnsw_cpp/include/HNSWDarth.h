// HNSWDarth.h
#pragma once

#include <vector>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstdint>

struct DarthParams {
  int k = 10;           // top-k
  float Rt = 0.9f;      // target recall
  int efSearch = 100;   // max search effort cap (soft cap)
  int ipi = 32;         // initial prediction interval (in distance calcs)
  int mpi = 4;          // minimum prediction interval
};

// Predictor interface (plug in LightGBM/GBDT/etc.)
struct IRecallPredictor {
  virtual ~IRecallPredictor() = default;
  // features size must be 11; return Rp in [0,1]
  virtual float predict_one(const float* features11) const = 0;
};

// --------------------
// Minimal expectations from your HNSW class:
// - vectors storage (id -> pointer/array)
// - layers adjacency (layer 0 used here)
// - dist(q, id) returning float (L2 squared or cosine etc.)
// - greedy descent in upper layers already done in your query()
// --------------------

// Utility: update prediction interval (Eq.1 in DARTH)  pi = mpi + (ipi-mpi)*(Rt-Rp)
inline int darth_update_pi(int ipi, int mpi, float Rt, float Rp) {
  float val = static_cast<float>(mpi) + (static_cast<float>(ipi - mpi)) * (Rt - Rp);
  if (val < static_cast<float>(mpi)) val = static_cast<float>(mpi);
  if (val > static_cast<float>(ipi)) val = static_cast<float>(ipi);
  int out = static_cast<int>(std::lround(val));
  return std::max(1, out);
}

// Compute stats on sorted distances: avg, var, median, p25, p75
inline void nn_stats(const std::vector<float>& dists_sorted,
                     float& avg, float& var, float& med, float& p25, float& p75) {
  if (dists_sorted.empty()) { avg=var=med=p25=p75=0.0f; return; }
  double sum = 0.0;
  for (float x : dists_sorted) sum += x;
  avg = static_cast<float>(sum / dists_sorted.size());

  double vsum = 0.0;
  for (float x : dists_sorted) { double dx = x - avg; vsum += dx*dx; }
  var = static_cast<float>(vsum / dists_sorted.size());

  auto pct = [&](float p)->float{
    if (dists_sorted.size()==1) return dists_sorted[0];
    float idx = p * (dists_sorted.size() - 1);
    size_t i0 = static_cast<size_t>(std::floor(idx));
    size_t i1 = std::min(i0 + 1, dists_sorted.size()-1);
    float t = idx - static_cast<float>(i0);
    return dists_sorted[i0]*(1.0f-t) + dists_sorted[i1]*t;
  };

  med = pct(0.5f);
  p25 = pct(0.25f);
  p75 = pct(0.75f);
}

// This is the DARTH base-layer search implementation.
// You call it AFTER upper-layer greedy descent gives you an entrypoint ep_id.
// You must provide callbacks to access neighbors and compute dist.
class DarthSearcher {
public:
  // Neighbor callback: returns adjacency list for a node in layer 0
  using NeighFn = const std::vector<int>& (*)(int node_id, void* ctx);
  // Dist callback: returns distance between query vector and node vector
  using DistFn  = float (*)(const float* q, int node_id, void* ctx);

  // ctx is your HNSW instance pointer (void* to avoid including your whole class here)
  static std::vector<int> search_layer0_darth(
      const float* q,
      int ep_id,
      const DarthParams& params,
      const IRecallPredictor& model,
      NeighFn neigh,
      DistFn dist,
      void* ctx
  );
};
