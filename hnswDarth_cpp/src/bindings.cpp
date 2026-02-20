#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hnswDarth.h"

namespace py = pybind11;

class PyPredictor : public HNSW_DARTH::IPredictor {
    public:
        using HNSW_DARTH::IPredictor::IPredictor; // inherit constructors

        float predict(const HNSW_DARTH::DarthFeatures& f) const override{
            PYBIND11_OVERRIDE_PURE(
                float,          // Return type
                HNSW_DARTH::IPredictor, // Parent class
                predict,        // Name of function in C++ (must match Python name)
                f               // Argument(s)
            );
        }
};

PYBIND11_MODULE(hnsw_darth_cpp,m){
    m.doc() = "HNSW_DARTH C++ implementation with DARTH predictor interface";

    py::enum_<HNSW_DARTH::Metric>(m,"Metric")
    .value("L2", HNSW_DARTH::Metric::L2)
    .value("Cosine", HNSW_DARTH::Metric::Cosine)
    .export_values();

    py::class_<HNSW_DARTH::DarthFeatures>(m, "DarthFeatures")
        .def(py::init<>())
        .def_readwrite("ndis", &HNSW_DARTH::DarthFeatures::ndis)
        .def_readwrite("nstep", &HNSW_DARTH::DarthFeatures::nstep)
        .def_readwrite("ninserts", &HNSW_DARTH::DarthFeatures::ninserts)
        .def_readwrite("firstNN", &HNSW_DARTH::DarthFeatures::firstNN)
        .def_readwrite("closestNN", &HNSW_DARTH::DarthFeatures::closestNN)
        .def_readwrite("furthestNN", &HNSW_DARTH::DarthFeatures::furthestNN)
        .def_readwrite("meanNN", &HNSW_DARTH::DarthFeatures::meanNN)
        .def_readwrite("varNN", &HNSW_DARTH::DarthFeatures::varNN)
        .def_readwrite("p25NN", &HNSW_DARTH::DarthFeatures::p25NN)
        .def_readwrite("p50NN", &HNSW_DARTH::DarthFeatures::p50NN)
        .def_readwrite("p75NN", &HNSW_DARTH::DarthFeatures::p75NN);

    py::class_<HNSW_DARTH::IPredictor, PyPredictor>(m, "Predictor")
    .def(py::init<>())
    .def("predict", &HNSW_DARTH::IPredictor::predict);

    py::class_<HNSW_DARTH>(m, "HNSW_DARTH")
    .def(py::init<int,int,int,HNSW_DARTH::Metric,uint64_t>(),
    py::arg("dim"), py::arg("M"), py::arg("efConstruction"), py::arg("metric")=HNSW_DARTH::Metric::L2, py::arg("seed")=42)

    .def ("insert", &HNSW_DARTH::insert, py::arg("vec"), py::arg("node_id")=-1,"Insert a vector with optional node_id (auto-assigned if -1)")
    .def("query_darth", &HNSW_DARTH::query_darth, py::arg("q"), py::arg("k"), py::arg("efSearch"), py::arg("Rt"), py::arg("predictor"), py::arg("ipi")=200, py::arg("mpi")=20,
         "Query with DARTH: q=vector, k=num neighbors, efSearch=beam width, Rt=latency threshold, predictor=Predictor instance, ipi=initial pi, mpi=min pi")
    .def("max_level", &HNSW_DARTH::max_level, "Get maximum level of the graph")
    .def("entry_point", &HNSW_DARTH::entry_point, "Get entry point node id");

}