#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <memory>

#include "hnswDarth.h"

namespace py = pybind11;

static HNSW_DARTH::Metric parse_metric(const std::string& metric) {
    if (metric == "l2") return HNSW_DARTH::Metric::L2;
    if (metric == "cosine") return HNSW_DARTH::Metric::Cosine;
    throw std::runtime_error("Unknown metric: " + metric);
}

struct PyPredictor final : public HNSW_DARTH::IPredictor {
    explicit PyPredictor(py::object fn) : fn_(std::move(fn)) {}

    float predict(const HNSW_DARTH::DarthFeatures& f) const override {
        py::gil_scoped_acquire gil;

        py::dict d;
        d["ndis"] = f.ndis;
        d["nstep"] = f.nstep;
        d["ninserts"] = f.ninserts;
        d["firstNN"] = f.firstNN;

        d["closestNN"] = f.closestNN;
        d["furthestNN"] = f.furthestNN;
        d["meanNN"] = f.meanNN;
        d["varNN"] = f.varNN;

        d["p25NN"] = f.p25NN;
        d["p50NN"] = f.p50NN;
        d["p75NN"] = f.p75NN;

        py::object out = fn_(d);
        return out.cast<float>();
    }

private:
    py::object fn_;
};

class HNSWDarthIndex {
public:
    HNSWDarthIndex(int dim, int M, int efConstruction, const std::string& metric)
        : dim_(dim),
          index_(dim, M, efConstruction, parse_metric(metric)) {}

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> xb) {
        auto b = xb.request();
        if (b.ndim != 2) throw std::runtime_error("xb must be 2D");
        const int64_t nb  = (int64_t)b.shape[0];
        const int64_t dim = (int64_t)b.shape[1];
        if ((int)dim != dim_) throw std::runtime_error("dim mismatch");

        const float* ptr = static_cast<const float*>(b.ptr);

        py::gil_scoped_release release;
        std::vector<float> v((size_t)dim_);

        for (int64_t i = 0; i < nb; ++i) {
            const float* row = ptr + i * dim_;
            for (int d = 0; d < dim_; ++d) v[(size_t)d] = row[d];
            index_.insert(v);
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
        if ((int)dim != dim_) throw std::runtime_error("dim mismatch");

        const float* qptr = static_cast<const float*>(qbuf.ptr);

        std::vector<std::vector<float>> Xqv((size_t)nq, std::vector<float>((size_t)dim_));
        for (int64_t i = 0; i < nq; ++i) {
            const float* row = qptr + i * dim_;
            for (int d = 0; d < dim_; ++d) Xqv[(size_t)i][(size_t)d] = row[d];
        }

        std::vector<std::vector<float>> Dv;
        std::vector<std::vector<int>>   Iv;

        {
            py::gil_scoped_release release;
            index_.search(Xqv, k, efSearch, Dv, Iv);
        }

        py::array_t<int>   I({nq, (int64_t)k});
        py::array_t<float> D({nq, (int64_t)k});
        auto Iw = I.mutable_unchecked<2>();
        auto Dw = D.mutable_unchecked<2>();

        for (int64_t i = 0; i < nq; ++i) {
            for (int j = 0; j < k; ++j) {
                Iw(i, j) = (j < (int)Iv[(size_t)i].size()) ? Iv[(size_t)i][(size_t)j] : -1;
                Dw(i, j) = (j < (int)Dv[(size_t)i].size())
                         ? Dv[(size_t)i][(size_t)j]
                         : std::numeric_limits<float>::infinity();
            }
        }

        return {D, I};
    }

    std::pair<py::array_t<float>, py::array_t<int>> search_darth(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        int k,
        int efSearch,
        float Rt,
        py::object predictor,
        int ipi,
        int mpi
    ) {
        auto qbuf = xq.request();
        if (qbuf.ndim != 2) throw std::runtime_error("xq must be 2D");
        const int64_t nq  = (int64_t)qbuf.shape[0];
        const int64_t dim = (int64_t)qbuf.shape[1];
        if ((int)dim != dim_) throw std::runtime_error("dim mismatch");

        const float* qptr = static_cast<const float*>(qbuf.ptr);

        std::vector<std::vector<float>> Xqv((size_t)nq, std::vector<float>((size_t)dim_));
        for (int64_t i = 0; i < nq; ++i) {
            const float* row = qptr + i * dim_;
            for (int d = 0; d < dim_; ++d) Xqv[(size_t)i][(size_t)d] = row[d];
        }

        std::vector<std::vector<float>> Dv;
        std::vector<std::vector<int>>   Iv;

        std::unique_ptr<PyPredictor> pyPred;
        const HNSW_DARTH::IPredictor* predPtr = nullptr;

        if (!predictor.is_none()) {
            pyPred = std::make_unique<PyPredictor>(predictor);
            predPtr = pyPred.get();
        }

        {
            py::gil_scoped_release release;
            index_.search_darth(Xqv, k, efSearch, Rt, predPtr, ipi, mpi, Dv, Iv);
        }

        py::array_t<int>   I({nq, (int64_t)k});
        py::array_t<float> D({nq, (int64_t)k});
        auto Iw = I.mutable_unchecked<2>();
        auto Dw = D.mutable_unchecked<2>();

        for (int64_t i = 0; i < nq; ++i) {
            for (int j = 0; j < k; ++j) {
                Iw(i, j) = (j < (int)Iv[(size_t)i].size()) ? Iv[(size_t)i][(size_t)j] : -1;
                Dw(i, j) = (j < (int)Dv[(size_t)i].size())
                         ? Dv[(size_t)i][(size_t)j]
                         : std::numeric_limits<float>::infinity();
            }
        }

        return {D, I};
    }

    // Compat aliases (kept from your old code)
    std::pair<py::array_t<float>, py::array_t<int>> search_layer(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        int k,
        int efSearch
    ) { return search(xq, k, efSearch); }

    std::pair<py::array_t<float>, py::array_t<int>> search_layer_darth(
        py::array_t<float, py::array::c_style | py::array::forcecast> xq,
        int k,
        int efSearch,
        float Rt,
        py::object predictor,
        int ipi,
        int mpi
    ) { return search_darth(xq, k, efSearch, Rt, predictor, ipi, mpi); }

private:
    int dim_;
    HNSW_DARTH index_;
};

PYBIND11_MODULE(hnswDarth_cpp, m) {
    py::class_<HNSWDarthIndex>(m, "HNSWDarthIndex")
        .def(py::init<int,int,int,const std::string&>(),
             py::arg("dim"), py::arg("M"), py::arg("efConstruction"), py::arg("metric")="l2")
        .def("add", &HNSWDarthIndex::add, py::arg("xb"))
        .def("search", &HNSWDarthIndex::search,
             py::arg("xq"), py::arg("k"), py::arg("efSearch"))
        .def("search_darth", &HNSWDarthIndex::search_darth,
             py::arg("xq"), py::arg("k"), py::arg("efSearch"),
             py::arg("Rt")=0.95f,
             py::arg("predictor")=py::none(),
             py::arg("ipi")=200,
             py::arg("mpi")=0)
        .def("search_layer", &HNSWDarthIndex::search_layer,
             py::arg("xq"), py::arg("k"), py::arg("efSearch"))
        .def("search_layer_darth", &HNSWDarthIndex::search_layer_darth,
             py::arg("xq"), py::arg("k"), py::arg("efSearch"),
             py::arg("Rt")=0.95f,
             py::arg("predictor")=py::none(),
             py::arg("ipi")=200,
             py::arg("mpi")=0);
}