#pragma once

#include "AdaptiveRdp.h"

#include <cstring>
#include <vector>

// C++ helpers callable from Cython (rdp.pyx).

inline void rdp_adaptive_fill_mask(
    const double* x,
    const double* y,
    int n,
    char* mask,
    bool fast_dist = true)
{
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        mask[0] = 1;
        return;
    }
    if (n == 2) {
        mask[0] = 1;
        mask[1] = 1;
        return;
    }

    std::memset(mask, 0, static_cast<size_t>(n));

    std::vector<std::size_t> keep;
    keep.reserve(static_cast<std::size_t>(n) / 4 + 8);
    keep.push_back(0);

    rdp_adaptive::ramer_douglas_peucker_adaptive(
        x,
        y,
        static_cast<std::size_t>(n),
        keep,
        fast_dist);

    for (std::size_t i = 0; i < keep.size(); ++i) {
        mask[keep[i]] = 1;
    }
}

inline bool rdp_supports_avx2()
{
    return rdp_adaptive::cpu_supports_avx2_fma();
}
