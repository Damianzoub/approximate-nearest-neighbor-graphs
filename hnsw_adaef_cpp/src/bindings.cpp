#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hnsw_adaef.h"

#include <vector>
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// -------------------------
// Helpers
// -------------------------
static std::vector<std::vector<float>> numpy_to_vec2d_f32(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected a 2D float32 array");
    }

    const ssize_t n = buf.shape[0];
    const ssize_t d = buf.shape[1];
    const float* ptr = static_cast<float*>(buf.ptr);

    std::vector<std::vector<float>> out(n, std::vector<float>(d));
    for (ssize_t i = 0; i < n; ++i) {
        std::memcpy(out[i].data(), ptr + i * d, d * sizeof(float));
    }
    return out;
}

static py::array_t<float> vec2d_to_numpy_f32(const std::vector<std::vector<float>>& X) {
    ssize_t n = static_cast<ssize_t>(X.size());
    ssize_t d = (n == 0 ? 0 : static_cast<ssize_t>(X[0].size()));

    py::array_t<float> arr({n, d});
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);

    for (ssize_t i = 0; i < n; ++i) {
        if ((ssize_t)X[i].size() != d) {
            throw std::runtime_error("Inconsistent row sizes in float matrix");
        }
        std::memcpy(ptr + i * d, X[i].data(), d * sizeof(float));
    }
    return arr;
}

static py::array_t<int> vec2d_to_numpy_i32(const std::vector<std::vector<int>>& X) {
    ssize_t n = static_cast<ssize_t>(X.size());
    ssize_t d = (n == 0 ? 0 : static_cast<ssize_t>(X[0].size()));

    py::array_t<int> arr({n, d});
    auto buf = arr.request();
    int* ptr = static_cast<int*>(buf.ptr);

    for (ssize_t i = 0; i < n; ++i) {
        if ((ssize_t)X[i].size() != d) {
            throw std::runtime_error("Inconsistent row sizes in int matrix");
        }
        std::memcpy(ptr + i * d, X[i].data(), d * sizeof(int));
    }
    return arr;
}

// -------------------------
// Module
// -------------------------
PYBIND11_MODULE(hnswAdaEF_cpp, m) {
    m.doc() = "Pybind11 bindings for HNSW + Ada-ef";

    py::class_<HNSWAdaEFIndex>(m, "HNSWAdaEFIndex")
        .def(py::init<int, int, int, const std::string&, uint32_t, int, double, int>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("efConstruction") = 200,
             py::arg("metric") = "cosine",
             py::arg("seed") = 42,
             py::arg("adaef_bins") = 5,
             py::arg("adaef_delta") = 0.001,
             py::arg("adaef_sample_size") = 200)

        .def("add", [](HNSWAdaEFIndex& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> X) {
            self.add(numpy_to_vec2d_f32(X));
        }, py::arg("X"))

        .def("add_point", [](HNSWAdaEFIndex& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> x,
                             int node_id) {
            auto vv = numpy_to_vec2d_f32(x.reshape({static_cast<ssize_t>(1), x.size()}));
            return self.add_point(vv[0], node_id);
        }, py::arg("x"), py::arg("node_id") = -1)

        .def("build_adaef_offline",
             &HNSWAdaEFIndex::build_adaef_offline,
             py::arg("k"),
             py::arg("target_recall") = 0.95,
             py::arg("ef_values") = std::vector<int>{50, 75, 100, 150, 200, 300, 400, 500})

        .def("search", [](const HNSWAdaEFIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> Xq,
                          int k,
                          double target_recall) {
            auto queries = numpy_to_vec2d_f32(Xq);
            auto result = self.search(queries, k, target_recall);
            return py::make_tuple(
                vec2d_to_numpy_f32(result.first),
                vec2d_to_numpy_i32(result.second)
            );
        }, py::arg("Xq"), py::arg("k"), py::arg("target_recall"))

        .def("entry_point", &HNSWAdaEFIndex::entry_point)
        .def("max_level", &HNSWAdaEFIndex::max_level);
}