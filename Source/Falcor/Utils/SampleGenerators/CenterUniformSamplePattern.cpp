#include "stdafx.h"
#include "CenterUniformSamplePattern.h"

namespace Falcor
{
    CenterUniformSamplePattern::SharedPtr CenterUniformSamplePattern::create(uint32_t seed, uint32_t sampleIndex)
    {
        return SharedPtr(new CenterUniformSamplePattern(seed, sampleIndex));
    }

    CenterUniformSamplePattern::CenterUniformSamplePattern(uint32_t seed, uint32_t sampleIndex)
    {
        mSampleIndex = sampleIndex;
        mRng = std::mt19937(seed);
    }

    void CenterUniformSamplePattern::reset(uint32_t seed)
    {
        mRng = std::mt19937(seed);
    }

    float2 CenterUniformSamplePattern::next()
    {
        if (mSampleIndex == 0)
        {
            // The first sample is always the center
            return float2(0.0f);
        }
        else
        {
            // Then other samples are random
            auto dist = std::uniform_real_distribution<float>();
            auto u = [&](){ return dist(mRng); };
            return float2(u(), u()) - 0.5f;
        }
    }
}
