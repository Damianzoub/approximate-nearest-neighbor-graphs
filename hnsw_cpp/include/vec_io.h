#pragma once 

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace vecio{
    std::vector<std::vector<float>> read_fvecs(const std::string& filename);
    std::vector<std::vector<int>> read_ivecs(const std::string& filename);
}