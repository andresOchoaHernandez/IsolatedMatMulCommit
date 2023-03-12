#pragma once

#include <chrono>
#include <string>

namespace MeasureTime
{
    class Timer
    {
        std::chrono::steady_clock::time_point _begin;
        std::chrono::steady_clock::time_point _end;

        public:
            Timer();
            void begin();
            void end(const std::string& message);
    };
}