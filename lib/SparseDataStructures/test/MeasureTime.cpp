#include "MeasureTime.hpp"

#include <iostream>

namespace MeasureTime
{
    Timer::Timer(){}
    void Timer::begin()
    {
        _begin = std::chrono::steady_clock::now();
    }

    void Timer::end(const std::string& message)
    {
        _end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _begin).count();
        std::cout << message  << " took :" << duration << " ms" << std::endl;
    }
}