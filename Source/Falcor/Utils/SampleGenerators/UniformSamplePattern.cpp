#include "stdafx.h"
#include "UniformSamplePattern.h"

namespace Falcor
{
    UniformSamplePattern::SharedPtr UniformSamplePattern::create(uint32_t seed)
    {
        return SharedPtr(new UniformSamplePattern(seed));
    }

    UniformSamplePattern::UniformSamplePattern(uint32_t seed)
    {
        mRng = std::mt19937(seed);
    }

    void UniformSamplePattern::reset(uint32_t seed)
    {
        mRng = std::mt19937(seed);
    }

    float2 UniformSamplePattern::next()
    {
        auto dist = std::uniform_real_distribution<float>();
        auto u = [&](){ return dist(mRng); };
        return float2(u(), u()) - 0.5f;
    }
}
