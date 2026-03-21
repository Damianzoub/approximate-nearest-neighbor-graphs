#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hnsw_pip.h"
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
PYBIND11_MODULE(hnswPip_cpp, m) {
    m.doc() = "Pybind11 bindings for HNSW + PiP";

    py::class_<HNSWPiPIndex>(m, "HNSWPiPIndex")
        .def(py::init<int, int, int, const std::string&, uint32_t, double, int>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("efConstruction") = 200,
             py::arg("metric") = "l2",
             py::arg("seed") = 42,
             py::arg("pip_gamma") = 95.0,
             py::arg("pip_delta") = 20)

        .def("add", [](HNSWPiPIndex& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> X) {
            self.add(numpy_to_vec2d_f32(X));
        }, py::arg("X"))

        .def("add_point", [](HNSWPiPIndex& self,
                        py::array_t<float, py::array::c_style | py::array::forcecast> x,
                        int node_id) {
                        auto buf = x.request();
                        if (buf.ndim != 1) {
                            throw std::runtime_error("Expected a 1D float32 array for add_point");
                        }

                        const ssize_t d = buf.shape[0];
                        const float* ptr = static_cast<float*>(buf.ptr);

                        std::vector<float> v(d);
                        std::memcpy(v.data(), ptr, d * sizeof(float));

                        return self.add_point(v, node_id);
                    }, py::arg("x"), py::arg("node_id") = -1)

        .def("search", [](const HNSWPiPIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> Xq,
                          int k,
                          int efSearch) {
            auto queries = numpy_to_vec2d_f32(Xq);
            auto result = self.search(queries, k, efSearch);
            return py::make_tuple(
                vec2d_to_numpy_f32(result.first),
                vec2d_to_numpy_i32(result.second)
            );
        }, py::arg("Xq"), py::arg("k"), py::arg("efSearch"))

        .def("entry_point", &HNSWPiPIndex::entry_point)
        .def("max_level", &HNSWPiPIndex::max_level);
}