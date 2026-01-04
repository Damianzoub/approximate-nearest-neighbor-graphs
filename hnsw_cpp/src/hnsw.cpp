#include "hnsw.h"
#include "distance.h"
#include "util.h"

#include <queue>
#include <limits>
#include <algorithm>
#include <stdexcept>

// small wrapper to avoid including utils.h in header
class HNSWIndex::LevelGenWrapper {
    public: 
        explicit LevelGenWrapper(int seed, float level_mult) : level_gen_(seed, level_mult) {} LevelGenWrapper gen;
};

HNSWIndex::HNSWIndex(int dim, int M, int efConstruction, int seed): dim_(dim), M_(M), efConstruction_(efConstruction),enterPoint_(-1),maxLevel_(-1){
    levelMult_ = 1.0f/ std::log(static_cast<float>(M_));
    levelGen_ = new LevelGeneratorImpl(seed,levelMult_);
}


void HNSWIndex::addPoint(const std::vector<float>& vec,int id){
    if (static_cast<int>(vec.size()) != dim_){
        throw std::runtime_error("Dimension mismatch");
    }
    int nodeLevel = levelGen_-> gen.sampleLevel();
    Node node;
    node.id = id;
    node.data = vec;
    node.maxLevel = nodeLevel;
    //node.links.resize(nodeLevel+1);
    int newIndex = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));

    if (enterPoint_ == -1){
        enterPoint_ = newIndex;
        maxLevel_ = nodeLevel;
        return;
    }

    // skeleton only
    if (nodeLevel > maxLevel_){
        enterPoint_ = newIndex;
        maxLevel_ = nodeLevel;
    }
}

std::vector<std::pair<float,int>> HNSWIndex::searchKnn(
    const std::vector<float>& query, int k, int efSearch
) const{
    if (static_cast<int>(query.size()) != dim_){
        throw std::runtime_error("Dimension mismatch");
    }
    if (enterPoint_ == -1) return {};

    std::vector<std::pair<float,int>> res;
    res.reserve(nodes_.size());

    for (int i =0; i < static_cast<int>(nodes_.size()); ++i){
        float dist = l2_sqr(query,nodes_[i].data,query);
        res.push_back({dist,nodes_[i].id});
    }

    std::sort(res.begin(),res.end(),[](auto& a,auto& b){return a.first < b.first;});
    if (static_cast<int>(res.size()) > k){
        res.resize(k);
    }
}

std::vector<int> HNSWIndex::searchLayer(const std::vector<float>& query, int entry, int level, int ef) const
{
    return {entry};
}
