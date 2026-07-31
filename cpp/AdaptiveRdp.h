#pragma once

// Adaptive RDP: per-segment epsilon (algebraic compute_epsilon), SoA, iterative stack.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace rdp_adaptive {

// Runtime CPU check: binary may be built with -mavx2, but must not execute
// AVX2/FMA opcodes on CPUs that lack them (would SIGILL).
inline bool cpu_supports_avx2_fma()
{
#if defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
    // Cached: CPUID once per process.
    static const bool ok =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
#elif defined(__AVX2__) && defined(_MSC_VER)
    // MSVC: assume host supports AVX2 if the TU was compiled for it.
    // Prefer building AVX2 code in a separate TU with runtime dispatch for production Windows.
    return true;
#else
    return false;
#endif
}

// Algebraically equivalent to Cython compute_epsilon (no atan2/cos/sin/sqrt).
inline double compute_epsilon_sq(double dx, double dy)
{
    const double s2 = dx * dx + dy * dy;
    if (s2 == 0.0) {
        return 0.0;
    }
    const double inv_s2 = 1.0 / s2;
    const double t_max = (std::fabs(dx) + std::fabs(dy)) * inv_s2;
    const double poly = 1.0 - t_max + t_max * t_max;
    const double dphi1 = std::atan(std::fabs(dx + dy) * poly * inv_s2);
    const double dphi2 = std::atan(std::fabs(dy - dx) * poly * inv_s2);
    const double dphi = dphi1 > dphi2 ? dphi1 : dphi2;
    return s2 * dphi * dphi;
}

inline double point_to_segment_distance_sq_pre(
    double x, double y,
    double x1, double y1,
    double dx, double dy, double proj)
{
    // Legacy form (projection + residual) — kept for A/B.
    if (proj == 0.0) {
        const double ex = x - x1;
        const double ey = y - y1;
        return ex * ex + ey * ey;
    }

    const double px = x - x1;
    const double py = y - y1;
    double t = (px * dx + py * dy) / proj;
    if (t < 0.0) {
        t = 0.0;
    } else if (t > 1.0) {
        t = 1.0;
    }

    const double ox = x - (x1 + t * dx);
    const double oy = y - (y1 + t * dy);
    return ox * ox + oy * oy;
}

// Same clamped segment distance²; fewer deps via endpoint / cross branches.
inline double point_to_segment_distance_sq_fast(
    double x, double y,
    double x1, double y1,
    double x2, double y2,
    double dx, double dy, double proj)
{
    if (proj == 0.0) {
        const double ex = x - x1;
        const double ey = y - y1;
        return ex * ex + ey * ey;
    }

    const double px = x - x1;
    const double py = y - y1;
    const double t = (px * dx + py * dy) / proj;
    if (t <= 0.0) {
        return px * px + py * py;
    }
    if (t >= 1.0) {
        const double qx = x - x2;
        const double qy = y - y2;
        return qx * qx + qy * qy;
    }
    const double cross = dx * py - dy * px;
    return (cross * cross) / proj;
}

inline void find_most_distant_point_from_segment_soa_ref(
    const double* x,
    const double* y,
    std::size_t start_index,
    std::size_t end_index,
    double& max_dist2,
    std::size_t& max_dist_index)
{
    const double x1 = x[start_index];
    const double y1 = y[start_index];
    const double x2 = x[end_index];
    const double y2 = y[end_index];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double proj = dx * dx + dy * dy;

    max_dist2 = 0.0;
    max_dist_index = start_index;

    std::size_t i = start_index + 1;
    for (; i + 1 < end_index; i += 2) {
        const double d0 = point_to_segment_distance_sq_pre(x[i], y[i], x1, y1, dx, dy, proj);
        if (d0 > max_dist2) {
            max_dist2 = d0;
            max_dist_index = i;
        }
        const double d1 = point_to_segment_distance_sq_pre(x[i + 1], y[i + 1], x1, y1, dx, dy, proj);
        if (d1 > max_dist2) {
            max_dist2 = d1;
            max_dist_index = i + 1;
        }
    }
    if (i < end_index) {
        const double d = point_to_segment_distance_sq_pre(x[i], y[i], x1, y1, dx, dy, proj);
        if (d > max_dist2) {
            max_dist2 = d;
            max_dist_index = i;
        }
    }
}

#if defined(__AVX2__)
#include <immintrin.h>

inline void find_most_distant_point_from_segment_soa_avx2(
    const double* x,
    const double* y,
    std::size_t start_index,
    std::size_t end_index,
    double& max_dist2,
    std::size_t& max_dist_index)
{
    const double x1 = x[start_index];
    const double y1 = y[start_index];
    const double x2 = x[end_index];
    const double y2 = y[end_index];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double proj = dx * dx + dy * dy;

    max_dist2 = 0.0;
    max_dist_index = start_index;

    if (proj == 0.0) {
        for (std::size_t i = start_index + 1; i < end_index; ++i) {
            const double ex = x[i] - x1;
            const double ey = y[i] - y1;
            const double d = ex * ex + ey * ey;
            if (d > max_dist2) {
                max_dist2 = d;
                max_dist_index = i;
            }
        }
        return;
    }

    const __m256d vdx = _mm256_set1_pd(dx);
    const __m256d vdy = _mm256_set1_pd(dy);
    const __m256d vx1 = _mm256_set1_pd(x1);
    const __m256d vy1 = _mm256_set1_pd(y1);
    const __m256d vx2 = _mm256_set1_pd(x2);
    const __m256d vy2 = _mm256_set1_pd(y2);
    const __m256d inv_proj = _mm256_set1_pd(1.0 / proj);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);

    std::size_t i = start_index + 1;
    for (; i + 3 < end_index; i += 4) {
        const __m256d vx = _mm256_loadu_pd(x + i);
        const __m256d vy = _mm256_loadu_pd(y + i);
        const __m256d px = _mm256_sub_pd(vx, vx1);
        const __m256d py = _mm256_sub_pd(vy, vy1);
        const __m256d dot = _mm256_fmadd_pd(px, vdx, _mm256_mul_pd(py, vdy));
        const __m256d t = _mm256_mul_pd(dot, inv_proj);

        const __m256d dA = _mm256_fmadd_pd(px, px, _mm256_mul_pd(py, py));
        const __m256d qx = _mm256_sub_pd(vx, vx2);
        const __m256d qy = _mm256_sub_pd(vy, vy2);
        const __m256d dB = _mm256_fmadd_pd(qx, qx, _mm256_mul_pd(qy, qy));
        const __m256d cross = _mm256_fmsub_pd(vdx, py, _mm256_mul_pd(vdy, px));
        const __m256d dLine = _mm256_mul_pd(_mm256_mul_pd(cross, cross), inv_proj);

        const __m256d mask_lo = _mm256_cmp_pd(t, zero, _CMP_LE_OQ);
        const __m256d mask_hi = _mm256_cmp_pd(t, one, _CMP_GE_OQ);
        __m256d d2 = dLine;
        d2 = _mm256_blendv_pd(d2, dA, mask_lo);
        d2 = _mm256_blendv_pd(d2, dB, mask_hi);

        alignas(32) double dist[4];
        _mm256_store_pd(dist, d2);
        for (int k = 0; k < 4; ++k) {
            if (dist[k] > max_dist2) {
                max_dist2 = dist[k];
                max_dist_index = i + static_cast<std::size_t>(k);
            }
        }
    }
    for (; i < end_index; ++i) {
        const double d = point_to_segment_distance_sq_fast(
            x[i], y[i], x1, y1, x2, y2, dx, dy, proj);
        if (d > max_dist2) {
            max_dist2 = d;
            max_dist_index = i;
        }
    }
}
#endif

inline void find_most_distant_point_from_segment_soa_fast_scalar(
    const double* x,
    const double* y,
    std::size_t start_index,
    std::size_t end_index,
    double& max_dist2,
    std::size_t& max_dist_index)
{
    const double x1 = x[start_index];
    const double y1 = y[start_index];
    const double x2 = x[end_index];
    const double y2 = y[end_index];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double proj = dx * dx + dy * dy;

    max_dist2 = 0.0;
    max_dist_index = start_index;

    for (std::size_t i = start_index + 1; i < end_index; ++i) {
        const double d = point_to_segment_distance_sq_fast(
            x[i], y[i], x1, y1, x2, y2, dx, dy, proj);
        if (d > max_dist2) {
            max_dist2 = d;
            max_dist_index = i;
        }
    }
}

inline void find_most_distant_point_from_segment_soa_fast(
    const double* x,
    const double* y,
    std::size_t start_index,
    std::size_t end_index,
    double& max_dist2,
    std::size_t& max_dist_index)
{
#if defined(__AVX2__)
    if (cpu_supports_avx2_fma()) {
        find_most_distant_point_from_segment_soa_avx2(
            x, y, start_index, end_index, max_dist2, max_dist_index);
        return;
    }
#endif
    find_most_distant_point_from_segment_soa_fast_scalar(
        x, y, start_index, end_index, max_dist2, max_dist_index);
}

inline void find_most_distant_point_from_segment_soa(
    const double* x,
    const double* y,
    std::size_t start_index,
    std::size_t end_index,
    double& max_dist2,
    std::size_t& max_dist_index,
    bool fast_dist = true)
{
    if (fast_dist) {
        find_most_distant_point_from_segment_soa_fast(
            x, y, start_index, end_index, max_dist2, max_dist_index);
    } else {
        find_most_distant_point_from_segment_soa_ref(
            x, y, start_index, end_index, max_dist2, max_dist_index);
    }
}

inline void ramer_douglas_peucker_adaptive(
    const double* x,
    const double* y,
    std::size_t n_points,
    std::vector<std::size_t>& indices_to_keep,
    bool fast_dist = true)
{
    assert(n_points >= 2);
    assert(!indices_to_keep.empty() && indices_to_keep[0] == 0);

    std::vector<std::pair<std::size_t, std::size_t> > stack;
    stack.reserve(64);
    stack.push_back(std::make_pair(static_cast<std::size_t>(0), n_points - 1));

    while (!stack.empty()) {
        const std::size_t start_index = stack.back().first;
        const std::size_t end_index = stack.back().second;
        stack.pop_back();

        if (end_index <= start_index + 1) {
            indices_to_keep.push_back(end_index);
            continue;
        }

        const double dx = x[end_index] - x[start_index];
        const double dy = y[end_index] - y[start_index];
        const double epsilon_sq = compute_epsilon_sq(dx, dy);

        double max_dist2 = 0.0;
        std::size_t max_dist_index = start_index;
        find_most_distant_point_from_segment_soa(
            x, y, start_index, end_index, max_dist2, max_dist_index, fast_dist);

        if (max_dist2 > epsilon_sq) {
            stack.push_back(std::make_pair(max_dist_index, end_index));
            stack.push_back(std::make_pair(start_index, max_dist_index));
        } else {
            indices_to_keep.push_back(end_index);
        }
    }
}

}  // namespace rdp_adaptive
