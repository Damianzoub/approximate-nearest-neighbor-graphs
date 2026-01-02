#include <iostream>
#include <vector> 
#include <cmath>

float l2(const std::vector<float>& a, const std::vector<float>& b){
    float sum =0.0f;
    for (size_t i = 0; i< a.size(); i++){
        float d = a[i] - b[i];
        sum += d*d;
    }
    return std::sqrt(sum);

}

int main(){
    std::vector<std::vector<float>> points = {
        {1.0f, 2.0f},
        {4.0f, 5.0f},
        {7.0f, 8.0f}
    };
    std::vector<float> query = {0.8f,0.9f};

    int best =-1;
    float best_dist = 1e9f;

    for (size_t i = 0; i< points.size();i++){
        float dist = l2(points[i],query);
        if (dist < best_dist){
            best_dist = dist;
            best = static_cast<int>(i);
        }
    }
    std::cout << "Closest index: " << best << "\n";
    return 0;
}