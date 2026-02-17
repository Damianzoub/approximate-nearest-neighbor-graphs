#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>

#include "../include/hnsw_new.h"   // <-- προσαρμόζεις path αν χρειάζεται

namespace py = pybind11;

static HNSW_NEW::Metric parse_metric(const std::string& metric) {
    if (metric == "l2") return HNSW_NEW::Metric::L2;
    if (metric == "cosine") return HNSW_NEW::Metric::Cosine;
    throw std::runtime_error("Unknown metric: " + metric);
}

class HNSWIndex {
public:
    HNSWIndex(int dim, int M, int efConstruction, const std::string& metric)
        : dim_(dim),
          index_(dim, M, efConstruction, parse_metric(metric)) {}

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> xb) {
        auto b = xb.request();
        if (b.ndim != 2) throw std::runtime_error("xb must be 2D");
        const int64_t nb  = (int64_t)b.shape[0];
        const int64_t dim = (int64_t)b.shape[1];
        if (dim != dim_) throw std::runtime_error("dim mismatch");

        const float* ptr = static_cast<const float*>(b.ptr);

        // C++ work (no python objects inside)
        py::gil_scoped_release release;
        std::vector<float> v(dim_);

        for (int64_t i = 0; i < nb; ++i) {
            const float* row = ptr + i * dim_;
            for (int d = 0; d < dim_; ++d) v[d] = row[d];
            index_.insert(v); // node_id auto
        }
    }

    std::pair<py::array_t<float>, py::array_t<int>> search(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        int k,
        int efSearch
    ) {
        auto qbuf = xq.request();
        if (qbuf.ndim != 2) throw std::runtime_error("xq must be 2D");
        const int64_t nq  = (int64_t)qbuf.shape[0];
        const int64_t dim = (int64_t)qbuf.shape[1];
        if (dim != dim_) throw std::runtime_error("dim mismatch");

        const float* qptr = static_cast<const float*>(qbuf.ptr);

        // Prepare std::vectors for output
        std::vector<std::vector<float>> Xqv(nq, std::vector<float>(dim_));
        for (int64_t i = 0; i < nq; ++i) {
            const float* row = qptr + i * dim_;
            for (int d = 0; d < dim_; ++d) Xqv[i][d] = row[d];
        }

        std::vector<std::vector<float>> Dv;
        std::vector<std::vector<int>>   Iv;

        {
            py::gil_scoped_release release;
            index_.search(Xqv, k, efSearch, Dv, Iv);
        }

        // Convert to numpy (GIL held here)
        py::array_t<int>   I({nq, (int64_t)k});
        py::array_t<float> D({nq, (int64_t)k});
        auto Iw = I.mutable_unchecked<2>();
        auto Dw = D.mutable_unchecked<2>();

        for (int64_t i = 0; i < nq; ++i) {
            for (int j = 0; j < k; ++j) {
                Iw(i, j) = (j < (int)Iv[i].size()) ? Iv[i][j] : -1;
                Dw(i, j) = (j < (int)Dv[i].size()) ? Dv[i][j] : std::numeric_limits<float>::infinity();
            }
        }

        return {D, I};
    }

private:
    int dim_;
    HNSW_NEW index_;
};

PYBIND11_MODULE(hnsw_cpp, m) {
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<int,int,int,const std::string&>(),
             py::arg("dim"), py::arg("M"), py::arg("efConstruction"), py::arg("metric")="l2")
        .def("add", &HNSWIndex::add, py::arg("xb"))
        .def("search", &HNSWIndex::search, py::arg("xq"), py::arg("k"), py::arg("efSearch"));
}
