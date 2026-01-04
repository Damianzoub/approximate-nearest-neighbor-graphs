#pragma once
#include <random>

class LevelGenerator{
    public: 
        LevelGenerator(int seed,float levelMult);
        int sampleLevel();
    private:
        std::mt19937 rng_;
        std::uniform_real_distribution<float> unif_;
        float levelMult_;
};
