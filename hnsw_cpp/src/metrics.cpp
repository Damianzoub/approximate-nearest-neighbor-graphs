#include "metrics.h";
namespace eval {

double recall_at_k(const std::vector<std::vector<int>>& gt,
                   const std::vector<std::vector<int>>& pred,
                   int k) {
    size_t correct = 0;
    size_t total = gt.size() * k;

    for (size_t i = 0; i < gt.size(); ++i) {
        std::unordered_set<int> truth(gt[i].begin(), gt[i].begin() + k);
        for (int j = 0; j < k; ++j) {
            if (truth.count(pred[i][j])) {
                correct++;
            }
        }
    }

    return static_cast<double>(correct) / total;
}

template <typename Index>
double measure_qps(Index& index,
                   const std::vector<std::vector<float>>& queries,
                   int k) {

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& q : queries) {
        index.search(q, k);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return queries.size() / elapsed.count();
}

template <typename Index>
std::pair<double, double> measure_qps_recall(
    Index& index,
    const std::vector<std::vector<float>>& queries,
    int k,
    const std::vector<std::vector<int>>* gt) {

    std::vector<std::vector<int>> preds;
    preds.reserve(queries.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& q : queries) {
        preds.push_back(index.search(q, k));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double qps = queries.size() / elapsed.count();

    if (gt) {
        double recall = recall_at_k(*gt, preds, k);
        return {qps, recall};
    }

    return {qps, -1.0};
}

}
