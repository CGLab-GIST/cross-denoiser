#pragma once
#include "CPUSampleGenerator.h"
#include <random>

namespace Falcor
{
    /** Uniform random sample pattern generator.
    */
    class dlldecl UniformSamplePattern : public CPUSampleGenerator
    {
    public:
        using SharedPtr = std::shared_ptr<UniformSamplePattern>;

        virtual ~UniformSamplePattern() = default;

        static SharedPtr create(uint32_t seed = 0u);

        virtual uint32_t getSampleCount() const override { return 1; }
        virtual void reset(uint32_t seed = 0u) override;
        virtual float2 next() override;

    protected:
        UniformSamplePattern(uint32_t seed);

        std::mt19937 mRng;
    };
}
