#pragma once 
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

class LevelGenerator;

class HNSWIndex{
    public:
        HNSWIndex(int dim,int M, int efConstruction, int seed=42);
        int maxLevel() const {return maxLevel_;}
        int entryPoint() const {return entry_;}
        int addPoint(const std::vector<float>& vec, int id=-1);
        std::vector<int> query(const std::vector<float>& q, int K,  int efSearch) const;
    
    private:
        struct Node{
            int id;
            std::vector<float> data;
            int level;
        };

        int dim_;
        int M_;
        int M0_;
        int efConstruction_;

        std::vector<Node> nodes_;
        std::unordered_map<int,int> idToIndex_;
        std::vector<std::unordered_map<int, std::unordered_set<int>>> layers_;


        int maxLevel_;
        int entry_;
        float mL_;
        LevelGenerator* levelGen_;
    private:
        int searchLayerGreedy(const std::vector<float>& q, int entryIndex, int layer) const;
        std::vector<int> searchLayer(const std::vector<float>& q, int entryIndex,int layer, int ef) const;

        std::vector<int> selectNeighborsSimple(const std::vector<float>& q, const std::vector<int>& candidates, int Mmax) const;
        std::vector<int> selectNeighborsHeuristic(const std::vector<float>& q, const std::vector<int>& candidates,int Mmax) const;

        void pruneConnections(int nodeIndex, int layer, int Mmax);
};