#include <random>

static std::random_device rd;
static std::mt19937 gen(rd());

float myrand()
{
    std::uniform_real_distribution<float> dis(-M_PI, M_PI);
    return dis(gen);
}