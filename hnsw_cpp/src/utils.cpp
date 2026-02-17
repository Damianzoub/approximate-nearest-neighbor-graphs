#include "utils.h"
#include <cmath>
#include <algorithm>

LevelGenerator::LevelGenerator(int seed, float mL_):
rng_(seed),unif_(0.0f,1.0f), mL_(mL_){}

int LevelGenerator::sampleLevel(){
    float u = std::max(unif_(rng_),1e-12f);
    return static_cast<int>(-std::log(u)*mL_);
}