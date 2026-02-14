#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

std::vector<std::vector<int>> read_ivecs(const std::string& fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file)
        throw std::runtime_error("ivecs file not found: " + fname);

    // Read first dimension
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!file)
        throw std::runtime_error("ivecs file is empty or corrupted: " + fname);

    // Move back to beginning
    file.seekg(0, std::ios::beg);

    std::vector<std::vector<int>> vectors;

    while (true) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break;

        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!file)
            throw std::runtime_error("Error reading ivecs data from: " + fname);

        vectors.push_back(std::move(vec));
    }

    return vectors;
}

std::vector<std::vector<float>> read_fvecs(const std::string& fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file)
        throw std::runtime_error("fvecs file not found: " + fname);

    std::vector<std::vector<float>> vectors;

    while (true) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break;

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file)
            throw std::runtime_error("Error reading fvecs data from: " + fname);

        vectors.push_back(std::move(vec));
    }

    return vectors;
}
