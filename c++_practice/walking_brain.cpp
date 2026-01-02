#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath> 

struct Node{
    int id;
    std::vector<float> data;
    std::vector<int> neighbors;
};

float l2 (const std::vector<float>& a, const std::vector<float>& b ){
    float sum =0.0f;
    for (size_t i = 0; i < a.size(); i++){
        float d = a[i] - b[i];
        sum += d*d;
    }
    return std::sqrt(sum);
}

std::vector<Node> nodes = {
    {0, {1.0f, 2.0f}, {1,2}},
    {1, {1.0f, 5.0f}, {0,2}},
    {2, {7.0f, 8.0f}, {0,1}},
    {3, {2.0f, 1.0f}, {0}}
};

int greedy_search (const std::vector<Node>& nodes, int entry, const std::vector<float>& query){
    int current = entry;
    float current_dist = l2(nodes[current].data,query);
    bool improved = true;

    while (improved){
        improved = false;
        for (int neighbor_id : nodes[current].neighbors){
            float d=  l2(nodes[neighbor_id].data,query);
            if (d < current_dist){
                current = neighbor_id;
                current_dist = d;
                improved = true;
            }
        }
    }
    return current;
}

int main(){
    std::vector<float> query = {3.0f, 3.0f};
    int result = greedy_search(nodes,0,query);
    std::cout<<"Closest node id: " << result << "\n";
    return 0;
}