#include "util.h"
#include <cmath>

LevelGenerator::LevelGenerator(int seed,float level_mult) : rng_(seed), unif_(0.0f,1.0f),level_mult(level_mult){
    float u = unif_(rng);
    if (u < 1e-7f) u = 1e-7f;
    return static_cast<int>(-std::log(u)*level_mult)
}