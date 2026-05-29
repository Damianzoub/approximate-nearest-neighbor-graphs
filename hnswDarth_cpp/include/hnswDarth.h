#pragma once 

#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <limits>
#include <string>
#include <cmath>
class HNSW_DARTH {
    public: 
        enum class Metric {L2,Cosine};
        HNSW_DARTH(int dim, int M, int efConstruction, Metric metric = Metric::L2, uint64_t seed=42);

        int insert(const std::vector<float>& vec, int node_id=-1);
        std::vector<int> query(const std::vector<float>& q, int k ,int efSearch) const;
        void search(const std::vector<std::vector<float>>& Xq, int k ,int efSearch, std::vector<std::vector<float>>& D, std::vector<std::vector<int>>& I) const;

        int max_level() const {return maxlevel_;}
        int entry_point() const {return entry_id_;}
        int size() const { return static_cast<int>(slot_to_id_.size()); }
        void set_use_heuristic(bool v) {use_heuristic_ = v;}
        bool use_heuristic() const {return use_heuristic_;}
        
        struct DarthFeatures{
            int ndis =0;
            int nstep=0;
            int ninserts=0;
            float firstNN = std::numeric_limits<float>::infinity();
            float closestNN  = std::numeric_limits<float>::infinity();
            float furthestNN = 0.0f;
            float meanNN     = 0.0f;
            float varNN      = 0.0f;
            float p25NN      = 0.0f;
            float p50NN      = 0.0f;
            float p75NN      = 0.0f;
        };

        struct IPredictor{
          virtual ~IPredictor() = default;
          virtual float predict(const DarthFeatures& features) const = 0;
        };

        // Callback invoked at each DARTH prediction step during data collection.
        // Arguments: features snapshot, top-k node IDs sorted by distance (closest first).
        using DarthLogCallback = std::function<void(const DarthFeatures&, const std::vector<int>&)>;

        std::vector<int> query_darth(const std::vector<float>& q,int k, int efSearch,float Rt,const IPredictor* predictor,int ipi=200,int mpi=0) const;

        void search_darth(const std::vector<std::vector<float>>& Xq, int k ,int efSearch,float Rt,const IPredictor* predictor,int ipi,int mpi,std::vector<std::vector<float>>& D,std::vector<std::vector<int>>& I) const;

        // Data-collection variant: runs full search (never exits early) and invokes
        // log_cb at each prediction interval with the current feature snapshot and
        // top-k node IDs so the caller can compute per-step recall.
        void search_darth_collect(const std::vector<std::vector<float>>& Xq,
                                  int k,
                                  int efSearch,
                                  int ipi,
                                  const DarthLogCallback& log_cb,
                                  std::vector<std::vector<float>>& D,
                                  std::vector<std::vector<int>>& I) const;
        
    private:
        int dim_;
        int M_;
        int M0_;
        int efConstruction_;
        Metric metric_;

        //fast continuous vector storage--
        std::vector<float> data_;
        std::vector<float> inv_norms_;
        std::unordered_map<int, int> id_to_slot_; // node_id -> (offset, length)
        std::vector<int> slot_to_id_; // slot index -> node_id
        
        //fast visted bookkeping
        mutable std::vector<uint32_t> visited_tag_;
        mutable uint32_t cur_tag_=1;

        //buffers
        struct SearchScratch{
            std::vector<std::pair<float,int>> C; //min-heap
            std::vector<std::pair<float,int>> W;  // max-heap
            std::vector<std::pair<float,int>> tmp; //for final sort
        };
        mutable SearchScratch scratch_;

        std::vector<std::vector<std::vector<int>>> layers_;
        std::unordered_map<int, std::vector<float>> vectors_;
        std::vector<int> level_;
        int maxlevel_;
        int entry_id_;
        double mL_;
        //RNG
        mutable std::mt19937_64 rng_;
        mutable std::uniform_real_distribution<double> unif_;
        bool use_heuristic_;

        void ensure_layers(int new_maxlevel);
        void ensure_node_capacity(int slot,int upto_level);
        int sample_level() const;

        // pointer helpers + distance helpers
        inline const float* vec_ptr_slot(int slot) const{
            return data_.data() + (size_t)slot *(size_t)dim_;
        }
        inline void begin_search_tag() const {
            ++cur_tag_;
            if (cur_tag_ == 0){ //overflow
                std::fill(visited_tag_.begin(), visited_tag_.end(), 0);
                cur_tag_ = 1;
            }
        }
        inline bool is_visited(int slot) const {return visited_tag_[slot] == cur_tag_;}
        inline void mark_visited(int slot) const {visited_tag_[slot] = cur_tag_;}

        // ---------- fast kernels (hot path) ----------
        static inline float l2_unrolled4(const float* __restrict a,
                                        const float* __restrict b,
                                        int dim) {
            float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
            int i = 0;
            int limit = dim - (dim % 4);

            for (; i < limit; i += 4) {
                float d0 = a[i]   - b[i];
                float d1 = a[i+1] - b[i+1];
                float d2 = a[i+2] - b[i+2];
                float d3 = a[i+3] - b[i+3];
                s0 += d0*d0; s1 += d1*d1; s2 += d2*d2; s3 += d3*d3;
            }
            float dist = s0 + s1 + s2 + s3;
            for (; i < dim; ++i) {
                float d = a[i] - b[i];
                dist += d*d;
            }
            return dist;
        }

        static inline float dot_unrolled4(const float* __restrict a,
                                        const float* __restrict b,
                                        int dim) {
            float s0=0.f, s1=0.f, s2=0.f, s3=0.f;
            int i = 0;
            int limit = dim - (dim % 4);

            for (; i < limit; i += 4) {
                s0 += a[i]   * b[i];
                s1 += a[i+1] * b[i+1];
                s2 += a[i+2] * b[i+2];
                s3 += a[i+3] * b[i+3];
            }
            float dot = s0 + s1 + s2 + s3;
            for (; i < dim; ++i) dot += a[i] * b[i];
            return dot;
        }

        static inline float inv_norm_ptr(const float* __restrict a, int dim) {
            // 1/||a|| with epsilon to avoid divide-by-zero
            float ss = dot_unrolled4(a, a, dim);
            float n = std::sqrt(ss);
            const float eps = 1e-12f;
            return 1.0f / (n + eps);
        }


        // Optimized distance between query pointer q and slot vector
        inline float dist_ptr(const float* __restrict q, int slot, float q_inv_norm = 1.0f) const {
            const float* __restrict v = vec_ptr_slot(slot);

            if (metric_ == Metric::L2) {
                return l2_unrolled4(q, v, dim_);
            } else {
                // cosine distance = 1 - dot(q,v) * inv||q|| * inv||v||
                // inv_norms_[slot] is precomputed at insert time
                float dot = dot_unrolled4(q, v, dim_);
                float sim = dot * q_inv_norm * inv_norms_[slot];
                return 1.0f - sim;
            }
        }


        inline float dist_slots(int a_slot, int b_slot) const {
            const float* __restrict a = vec_ptr_slot(a_slot);
            if (metric_ == Metric::L2) {
                return l2_unrolled4(a, vec_ptr_slot(b_slot), dim_);
            } else {
                float dot = dot_unrolled4(a, vec_ptr_slot(b_slot), dim_);
                float sim = dot * inv_norms_[a_slot] * inv_norms_[b_slot];
                return 1.0f - sim;
            }
        }
        int search_layer_greedy(const std::vector<float>& q,int ep, int layer) const;

        std::vector<int> search_layer_beam(const std::vector<float>& q, int ep, int layer, int ef ) const;

        std::vector<int> select_neighbors_simple(const std::vector<float>& q, const std::vector<int>& candidates, int layer, int Mmax) const;

        std::vector<int> select_neighbors_heuristic_paper_slot(int qslot,const std::vector<int>& candidates, int layer, int Mmax,bool extendCandidates,bool keepPruned) const;
        std::vector<int> query_slots(const std::vector<float>& q, int k, int efSearch) const;
        std::vector<int> prune_connection(int node_id, int layer, int Mmax);
        void check_dim(const std::vector<float>& v) const;  
                // ---------- DARTH internals ----------
        std::vector<int> search_layer_darth(const std::vector<float>& q_vec,
                                            int ep_id,
                                            int layer,
                                            int efSearch,
                                            int k,
                                            float Rt,
                                            const IPredictor* predictor,
                                            int ipi,
                                            int mpi,
                                            const DarthLogCallback* log_cb = nullptr) const;

        static DarthFeatures darth_extract_features(const std::vector<std::pair<float,int>>& result_topk,
                                                    int ndis,
                                                    int nstep,
                                                    float firstNN,
                                                    int ninserts);      
};