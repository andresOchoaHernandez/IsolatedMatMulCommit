#include <limits>
#include <cmath>
#include <algorithm>

#include "LinearAlgebra.hpp"

namespace LinearAlgebra
{
    bool areFloatNearlyEqual(float a, float b) {
        const float normal_min = std::numeric_limits<float>::min();
        const float relative_error = 0.0001;
        if (!std::isfinite(a) || !std::isfinite(b))
        {
            return false;
        }

        float diff = std::abs(a - b);
        if (diff <= normal_min) 
            return true;

        float abs_a = std::abs(a);
        float abs_b = std::abs(b);

        return (diff / std::max(abs_a, abs_b)) <= relative_error;
    }
}