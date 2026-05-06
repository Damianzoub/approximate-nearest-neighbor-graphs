#pragma once

#include "hnswDarth.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal LightGBM C API declarations (subset we actually use).
// The full c_api.h is not bundled with the pip package, so we declare only
// what is needed and link against lib_lightgbm.dylib at build time.
//
// NOTE: LGBM_BoosterPredictForMatSingleRow segfaults when called directly
// (it requires internal thread-pool init that only runs through the standard
// Python booster path).  LGBM_BoosterPredictForMat with nrow=1 is identical
// in practice and does not have this requirement.
// ---------------------------------------------------------------------------
extern "C" {
    typedef void* BoosterHandle;

    int LGBM_BoosterCreateFromModelfile(const char*    filename,
                                        int*           out_num_iterations,
                                        BoosterHandle* out);
    int LGBM_BoosterFree(BoosterHandle handle);

    // Calculate number of output values for a given predict call
    int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                   int           num_row,
                                   int           predict_type,
                                   int           start_iteration,
                                   int           num_iteration,
                                   int64_t*      out_len);

    // Dense matrix prediction — works reliably for 1 row (unlike SingleRow variant)
    int LGBM_BoosterPredictForMat(BoosterHandle handle,
                                  const void*   data,
                                  int           data_type,
                                  int32_t       nrow,
                                  int32_t       ncol,
                                  int           is_row_major,
                                  int           predict_type,
                                  int           start_iteration,
                                  int           num_iteration,
                                  const char*   parameter,
                                  int64_t*      out_len,
                                  double*       out_result);

    const char* LGBM_GetLastError();
}

// LightGBM C API constants
static constexpr int LGBM_C_API_PREDICT_NORMAL = 0;
static constexpr int LGBM_C_API_DTYPE_FLOAT64  = 1;

// ---------------------------------------------------------------------------
// Number of features — must match FEATURE_COLUMNS in darth/predictor.py
// Order: nstep, ndis, ninserts, firstNN, closestNN, furthestNN,
//        meanNN, varNN, p25NN, p50NN, p75NN
// ---------------------------------------------------------------------------
static constexpr int DARTH_NUM_FEATURES = 11;

// ---------------------------------------------------------------------------
// LGBMPredictor — native C++ implementation of HNSW_DARTH::IPredictor.
//
// Loads a LightGBM model from a .txt file and runs prediction entirely in C++
// with no Python callbacks on the hot path.
// ---------------------------------------------------------------------------
class LGBMPredictor final : public HNSW_DARTH::IPredictor {
public:
    explicit LGBMPredictor(const std::string& model_path) {
        int num_iters = 0;
        if (LGBM_BoosterCreateFromModelfile(model_path.c_str(), &num_iters, &booster_) != 0)
            throw std::runtime_error(
                "LGBMPredictor: failed to load '" + model_path +
                "': " + LGBM_GetLastError());

        // Pre-allocate output buffer sized to what the model actually produces
        int64_t out_len = 0;
        if (LGBM_BoosterCalcNumPredict(booster_, 1,
                                        LGBM_C_API_PREDICT_NORMAL,
                                        0, -1, &out_len) != 0) {
            LGBM_BoosterFree(booster_);
            throw std::runtime_error(
                std::string("LGBMPredictor: CalcNumPredict failed: ") +
                LGBM_GetLastError());
        }
        out_buf_.assign(static_cast<size_t>(out_len), 0.0);
    }

    ~LGBMPredictor() override {
        if (booster_) LGBM_BoosterFree(booster_);
    }

    LGBMPredictor(const LGBMPredictor&)            = delete;
    LGBMPredictor& operator=(const LGBMPredictor&) = delete;

    float predict(const HNSW_DARTH::DarthFeatures& f) const override {
        // Feature vector — order must match FEATURE_COLUMNS in darth/predictor.py
        const double row[DARTH_NUM_FEATURES] = {
            static_cast<double>(f.nstep),
            static_cast<double>(f.ndis),
            static_cast<double>(f.ninserts),
            static_cast<double>(f.firstNN),
            static_cast<double>(f.closestNN),
            static_cast<double>(f.furthestNN),
            static_cast<double>(f.meanNN),
            static_cast<double>(f.varNN),
            static_cast<double>(f.p25NN),
            static_cast<double>(f.p50NN),
            static_cast<double>(f.p75NN),
        };

        int64_t actual_len = 0;
        LGBM_BoosterPredictForMat(
            booster_,
            row,
            LGBM_C_API_DTYPE_FLOAT64,
            1,                                       // nrow
            static_cast<int32_t>(DARTH_NUM_FEATURES),// ncol
            1,                                       // is_row_major (C order)
            LGBM_C_API_PREDICT_NORMAL,
            0,                                       // start_iteration
            -1,                                      // num_iteration (all trees)
            "",                                      // extra parameters
            &actual_len,
            out_buf_.data());

        double result = out_buf_[0];
        if (result < 0.0) result = 0.0;
        if (result > 1.0) result = 1.0;
        return static_cast<float>(result);
    }

private:
    BoosterHandle         booster_  = nullptr;
    mutable std::vector<double> out_buf_;
};
