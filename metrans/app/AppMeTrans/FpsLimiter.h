#pragma once

#include <chrono>
#include <thread>

class FpsLimiter {
public:
    FpsLimiter(int nFpsLimit) : nFpsLimit(nFpsLimit) {
        t0 = std::chrono::high_resolution_clock::now();
    }
    void CheckAndSleep() {
        if (!nFpsLimit) return;

        nFrame++;
        double t = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
        if (nFrame <= t * nFpsLimit) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * (nFrame - t * nFpsLimit) / nFpsLimit)));
    }

private:
    int nFrame = 0;
    int nFpsLimit;
    std::chrono::high_resolution_clock::time_point t0;
};
