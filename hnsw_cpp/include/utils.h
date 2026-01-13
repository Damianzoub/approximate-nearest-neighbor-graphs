#pragma once
#include <random>

class LevelGenerator {
public:
    LevelGenerator(int seed, float mL);
    int sampleLevel(); // returns level

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> unif_;
    float mL_;
};
