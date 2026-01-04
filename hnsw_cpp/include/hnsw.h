#pragma once
#include <vector>
#include <utility>
#include <unordered_set>

class HNSWIndex{
    public:
        HNSWIndex(int dim,int M, int efConstruction,int seed=42);
        void addPoint(const std::vector<float>& vec, int id);
        std::vector<std::pair<float,int>> searchKnn(
            const std::vector<float>& query, int k, int efSearch
        ) const;

        int size() const{return static_cast<int>(nodes_.size());}

    private: 
        struct Node{
            int id;
            std::vector<float> data;
            std::vector<std::vector<int>> neighbors; // neighbors per level
            int maxLevel;
        };
        int dim_;
        int M_;
        int efConstruction_;
        std::vector<Node> nodes_;
        int enterPoint_;
        int maxLevel_;
        float levelMult_;
        class LevelGeneratorImpl;
        LevelGeneratorImpl* levelGen_;

    private:
        std::vector<int> searchLayer(
            const std::vector<float>& query,
            int entry,
            int level,
            int ef) const;
        
};