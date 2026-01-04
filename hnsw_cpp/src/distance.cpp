#include "distance.h"
#include <cstddef>
#include <cmath>
float l2_sqr(const std::vector<float>& a,const std::vector<float>& b){
    float sum =0.0f;
    for (size_t i=0;i<a.size();i++){
        float diff = a[i]-b[i];
        sum += diff*diff;
    }
    return sum;
}

float cosine_similarity(const std::vector<float>& a,const std::vector<float>& b){
    float dot_product =0.0f;
    float norm_a =0.0f;
    float norm_b =0.0f;
    for (size_t i=0;i<a.size();i++){
        dot_product += a[i]*b[i];
        norm_a += a[i]*a[i];
        norm_b += b[i]*b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f; // Handle zero-vector case
    }
    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}