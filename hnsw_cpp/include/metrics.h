#pragma once 

#include <vector>
#include <chrono>
#include <unordered_set>

namespace eval{
    double recall_at_k(
        const std::vector<std::vector<int>>& ground_truth,
        const std::vector<std::vector<int>>& predictions,
        int k
    );

    template <typename Index>
    double measure_qps(
        Index& index,
        const std::vector<std::vector<float>>& queries,
        int k
    );

    template <typename Index>
    std::pair<double, double> measure_qps_recall(
        Index& index,
        const std::vector<std::vector<float>>& queries,
        int k,
        const std::vector<std::vector<int>>* ground_truth = nullptr
    );

}