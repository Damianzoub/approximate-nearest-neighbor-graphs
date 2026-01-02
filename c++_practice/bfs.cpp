#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <unordered_set>

struct Node{
    int id;
    std::vector<float> data;
    std::vector<int> neighbors;
};

float l2 (const std::vector<float>& a, const std::vector<float>& b){
    float sum =0.0f;
    for (size_t i=0; i <a.size();i++){
        float d = a[i]-b[i];
        sum += d*d;
    };
    return std::sqrt(sum);
}

int bfs(const std::vector<Node>& nodes,int entry,const std::vector<float>& query){
    using Candidate = std::pair<float,int>;
    std::priority_queue<Candidate,std::vector<Candidate>,std::greater<Candidate>> pq;
    std::unordered_set<int> visited;
    
    float entry_dist = l2(nodes[entry].data,query);
    pq.push({entry_dist,entry});
    visited.insert(entry);

    int best = entry;
    float best_dist = entry_dist;

    while(!pq.empty()){
        auto [dist,u] = pq.top();
        pq.pop();

        if(dist > best_dist){
            break;
        }

        for (int nb: nodes[u].neighbors){
            if (visited.count(nb)) continue;
            visited.insert(nb);
            float nb_dist = l2(nodes[nb].data,query);
            pq.push({nb_dist,nb});
            if (nb_dist < best_dist){
                best_dist = nb_dist;
                best = nb;
            }
        }
    }
    return best;
}