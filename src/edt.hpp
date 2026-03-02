/*
 * Graph-First Euclidean Distance Transform (ND)
 *
 * Input: a labels array or a pre-built voxel connectivity graph
 *        (bit-encoded uint8 for 1-4D, uint16 for 5-8D, uint32 for 9-16D, uint64 for 17-32D).
 *
 * Pipeline (edtsq / edtsq_from_labels_fused):
 *   1. Build a compact connectivity graph: each voxel stores a bitmask of
 *      forward edges plus a foreground marker at bit 0.
 *   2. Run all EDT passes directly from the graph — no intermediate segment label array:
 *      - Pass 0 (innermost axis): Rosenfeld-Pfaltz scan detects boundaries from graph
 *        edge bits and writes squared 1D distances.
 *      - Passes 1..N-1: parabolic envelope reads graph edge bits per scanline in-place.
 *   O(N) per scanline, parallelized across scanlines.
 *   For edtsq_from_graph: step 1 is skipped (caller supplies the pre-built graph).
 *
 * See src/README.md for graph bit encoding, memory layout, and algorithm details.
 */

#ifndef EDT_HPP
#define EDT_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <unordered_map>
#include "threadpool.h"

namespace nd {

// Tuning parameter
static size_t ND_CHUNKS_PER_THREAD = 1;

inline void set_tuning(size_t chunks_per_thread) {
    if (chunks_per_thread > 0) ND_CHUNKS_PER_THREAD = chunks_per_thread;
}

// Shared thread pool keyed by thread count; created lazily on first use
inline ThreadPool& shared_pool_for(size_t threads) {
    static std::mutex mutex;
    static std::unordered_map<size_t, std::unique_ptr<ThreadPool>> pools;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = pools[threads];
    if (!entry) {
        entry = std::make_unique<ThreadPool>(threads);
    }
    return *entry;
}

// Per-pass thread cap: further limits threads based on work in a single EDT axis pass.
// This is a C++-level inner cap applied per axis pass; the caller-supplied `desired`
// is already capped at the Python level by _adaptive_thread_limit_nd.
inline size_t compute_threads(size_t desired, size_t total_lines, size_t axis_length) {
    if (desired <= 1 || total_lines <= 1) return 1;

    size_t threads = std::min<size_t>(desired, total_lines);

    // Further cap based on work per pass (total_work = voxels along this axis sweep)
    const size_t total_work = axis_length * total_lines;
    if (total_work <= 60000) {
        threads = std::min<size_t>(threads, 4);
    } else if (total_work <= 120000) {
        threads = std::min<size_t>(threads, 8);
    } else if (total_work <= 400000) {
        threads = std::min<size_t>(threads, 12);
    }

    return std::max<size_t>(1, threads);
}

// Distribute [0, total) into up to max_chunks chunks across threads.
// Calls work(begin, end) directly when threads==1; otherwise via shared pool.
// Blocks until all chunks complete.
template <typename F>
inline void dispatch_parallel(size_t threads, size_t total, size_t max_chunks, F work) {
    if (threads <= 1 || total == 0) {
        work(size_t(0), total);
        return;
    }
    const size_t n_chunks = std::min(max_chunks, total);
    const size_t chunk_sz = (total + n_chunks - 1) / n_chunks;
    ThreadPool& pool = shared_pool_for(threads);
    std::vector<std::future<void>> pending;
    pending.reserve(n_chunks);
    for (size_t start = 0; start < total; start += chunk_sz) {
        const size_t end = std::min(total, start + chunk_sz);
        pending.push_back(pool.enqueue([=]() { work(start, end); }));
    }
    for (auto& f : pending) f.get();
}

// Precomputed per-pass iteration layout for an EDT axis pass.
// Gathers all "other" (non-axis) dimensions and their strides, and
// exposes for_each_line() to iterate every scanline in a slice range.
struct AxisPassInfo {
    size_t num_other = 0;   // number of non-axis dims
    size_t other_ext[32];   // extents of non-axis dims (in shape order)
    size_t other_str[32];   // strides of non-axis dims
    size_t total_lines = 1; // product of all other extents
    size_t first_ext  = 1;  // extent of first other dim  (parallelized over)
    size_t first_str  = 0;  // stride of first other dim
    size_t rest_prod  = 1;  // product of other_ext[1..num_other-1]

    AxisPassInfo(const size_t* shape, const size_t* strides,
                 size_t dims, size_t axis) {
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            other_ext[num_other] = shape[d];
            other_str[num_other] = strides[d];
            total_lines *= shape[d];
            num_other++;
        }
        if (num_other > 0) {
            first_ext = other_ext[0];
            first_str = other_str[0];
            for (size_t d = 1; d < num_other; d++)
                rest_prod *= other_ext[d];
        }
    }

    // Call fn(base) for every scanline starting offset whose first-other-dim
    // index falls in [begin, end).  Handles 1D and ND sub-iteration.
    //
    // For the ND branch, coords[1..num_other-1] are guaranteed to return to
    // all-zeros after exactly rest_prod inner iterations, so they are
    // initialized once and not re-initialized per i0 row.
    template <typename F>
    void for_each_line(size_t begin, size_t end, F fn) const {
        if (num_other <= 1) {
            // Simple path: one scanline per first-dim row
            for (size_t i0 = begin; i0 < end; i0++)
                fn(i0 * first_str);
        } else {
            // ND path: iterate the inner dims with a multi-dim counter.
            // coords reused across i0 rows; invariant: all-zero at start of each row.
            size_t coords[32] = {};
            for (size_t i0 = begin; i0 < end; i0++) {
                size_t base = i0 * first_str;
                for (size_t i = 0; i < rest_prod; i++) {
                    fn(base);
                    for (size_t d = 1; d < num_other; d++) {
                        coords[d]++;
                        base += other_str[d];
                        if (coords[d] < other_ext[d]) break;
                        base -= coords[d] * other_str[d];
                        coords[d] = 0;
                    }
                }
            }
        }
    }
};

template <typename T>
inline float sq(T x) { return float(x) * float(x); }

/*
 * Pass 0 from Graph
 *
 * Reads the voxel connectivity graph and computes the Rosenfeld-Pfaltz
 * 1D EDT (pass 0) directly. Does not write segment labels.
 */
template <typename GRAPH_T>
inline void squared_edt_1d_from_graph_direct(
    const GRAPH_T* graph,
    float* d,
    const int n,
    const int64_t stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border
) {
    if (n <= 0) return;

    const float anis_sq = anisotropy * anisotropy;
    int i = 0;

    while (i < n) {
        const int64_t base_idx = i * stride;

        // Check if this voxel is background (graph == 0)
        if (graph[base_idx] == 0) {
            d[base_idx] = 0.0f;
            i++;
            continue;
        }

        // Foreground: find segment extent using connectivity bits
        const int seg_start = i;
        GRAPH_T edge = graph[base_idx];
        i++;

        // Follow connectivity along axis
        while (i < n && (edge & axis_bit)) {
            edge = graph[i * stride];
            if (edge == 0) break;
            i++;
        }
        const int seg_len = i - seg_start;

        // Compute squared EDT for this segment.
        // Store squared distances directly to avoid a separate squaring pass.
        const bool left_boundary = (seg_start > 0) || black_border;
        const bool right_boundary = (i < n) || black_border;

        // Forward pass: squared distance from left boundary
        if (left_boundary) {
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = anis_sq * sq(k + 1);
            }
        } else {
            const float inf = std::numeric_limits<float>::infinity();
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = inf;
            }
        }

        // Backward pass: take min with squared distance from right boundary
        if (right_boundary) {
            for (int k = seg_len - 1; k >= 0; k--) {
                const float v_sq = anis_sq * sq(seg_len - k);
                const int64_t idx = (seg_start + k) * stride;
                if (v_sq < d[idx]) {
                    d[idx] = v_sq;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Pass 0 from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edt_pass0_from_graph_direct_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    const int n = int(shape[axis]);
    const int64_t stride_ax = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, n);

    auto process_range = [&](size_t begin, size_t end) {
        info.for_each_line(begin, end, [&](size_t base) {
            squared_edt_1d_from_graph_direct<GRAPH_T>(
                graph + base, output + base,
                n, stride_ax, axis_bit, anisotropy, black_border
            );
        });
    };

    dispatch_parallel(threads, info.first_ext, threads, process_range);
}

/*
 * Parabolic Pass from Graph
 *
 * Reads voxel connectivity graph directly; no separate segment label
 * building step.
 */
template <typename GRAPH_T>
inline void squared_edt_1d_parabolic_from_graph_ws(
    const GRAPH_T* graph,
    float* f,
    const int n,
    const int64_t stride,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    int* v,
    float* ff,
    float* ranges
) {
    if (n <= 0) return;

    constexpr int SMALL_THRESHOLD = 8;
    const float w2 = anisotropy * anisotropy;

    // Fast path for small segments: O(n²) brute force
    auto process_small_run = [&](int start, int len, bool left_border, bool right_border) {
        float original[SMALL_THRESHOLD];
        for (int q = 0; q < len; ++q) {
            original[q] = f[(start + q) * stride];
        }
        for (int j = 0; j < len; ++j) {
            float best = original[j];
            if (left_border) {
                const float cap_left = w2 * sq(j + 1);
                if (cap_left < best) best = cap_left;
            }
            if (right_border) {
                const float cap_right = w2 * sq(len - j);
                if (cap_right < best) best = cap_right;
            }
            for (int q = 0; q < len; ++q) {
                const float candidate = original[q] + w2 * sq(j - q);
                if (candidate < best) best = candidate;
            }
            f[(start + j) * stride] = best;
        }
    };

    // Parabolic envelope for larger segments
    auto process_large_run = [&](int start, int len, bool left_border, bool right_border) {
        // Copy to workspace
        for (int i = 0; i < len; i++) {
            ff[i] = f[(start + i) * stride];
        }

        // Skip INF-valued sources when building the parabolic envelope.
        // INF sources never win the minimum, and INF - INF = NaN corrupts
        // the intersection formula, leaving all subsequent ranges as NaN
        // and preventing the output pass from ever advancing k.
        int first_src = 0;
        while (first_src < len && std::isinf(ff[first_src])) first_src++;

        int k = 0;
        // If all sources are INF, fall back to v[0]=0 with ff[0]=INF so
        // the output pass correctly produces INF (borders still applied).
        v[0] = (first_src < len) ? first_src : 0;
        ranges[0] = -std::numeric_limits<float>::infinity();
        ranges[1] = std::numeric_limits<float>::infinity();

        // Intersection of the two parabolas centered at ff[a] and ff[b].
        // Use double arithmetic to avoid catastrophic cancellation when
        // ff[b] - ff[a] is tiny relative to the large squared-distance values.
        auto intersect = [&](int a, int b) -> float {
            const double d1 = double(b - a) * double(w2);
            return float((double(ff[b]) - double(ff[a]) + d1 * double(a + b)) / (2.0 * d1));
        };

        float s;
        const int loop_start = (first_src < len) ? first_src + 1 : len;
        for (int i = loop_start; i < len; i++) {
            if (std::isinf(ff[i])) continue;  // INF never wins the minimum

            s = intersect(v[k], i);
            while (k > 0 && s <= ranges[k]) {
                k--;
                s = intersect(v[k], i);
            }

            k++;
            v[k] = i;
            ranges[k] = s;
            ranges[k + 1] = std::numeric_limits<float>::infinity();
        }

        // Output pass - use specialized loops to avoid per-iteration conditionals
        k = 0;
        if (left_border && right_border) {
            // Both borders: take min of border distances and parabolic result
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float result = w2 * sq(i - v[k]) + ff[v[k]];
                const float envelope = w2 * std::fminf(sq(i + 1), sq(len - i));
                f[(start + i) * stride] = std::fminf(envelope, result);
            }
        } else if (left_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(w2 * sq(i + 1), w2 * sq(i - v[k]) + ff[v[k]]);
            }
        } else if (right_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(w2 * sq(len - i), w2 * sq(i - v[k]) + ff[v[k]]);
            }
        } else {
            // No borders - just parabolic result
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = w2 * sq(i - v[k]) + ff[v[k]];
            }
        }
    };

    // Scan graph to find foreground segments - single loop
    // Key insight: segment boundary when prev didn't connect forward (!(prev & axis_bit))
    // Background has graph=0, so axis_bit check handles both cases

    // Skip leading background
    int i = 0;
    while (i < n && graph[i * stride] == 0) i++;
    if (i >= n) return;

    int seg_start = i;
    GRAPH_T g = graph[i * stride];
    i++;

    while (i < n) {
        const GRAPH_T prev_g = g;
        g = graph[i * stride];

        // Boundary if previous didn't connect forward
        // Note: axis_bit encodes connectivity, so if current is background,
        // previous won't have axis_bit set (labels differ). No need for g==0 check.
        if (!(prev_g & axis_bit)) {
            // Process segment [seg_start, i)
            const int seg_len = i - seg_start;
            const bool left_border = (black_border || seg_start > 0);
            if (seg_len <= SMALL_THRESHOLD) {
                process_small_run(seg_start, seg_len, left_border, true);
            } else {
                process_large_run(seg_start, seg_len, left_border, true);
            }

            // Skip background, find next segment start
            while (i < n && graph[i * stride] == 0) i++;
            if (i >= n) return;
            seg_start = i;
            g = graph[i * stride];
        }
        i++;
    }

    // Final segment
    const int seg_len = n - seg_start;
    const bool left_border = (black_border || seg_start > 0);
    if (seg_len <= SMALL_THRESHOLD) {
        process_small_run(seg_start, seg_len, left_border, black_border);
    } else {
        process_large_run(seg_start, seg_len, left_border, black_border);
    }
}

//-----------------------------------------------------------------------------
// Parabolic Pass from Graph (parallel dispatch)
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edt_pass_parabolic_from_graph_fused_parallel(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const size_t* strides,
    const size_t dims,
    const size_t axis,
    const GRAPH_T axis_bit,
    const float anisotropy,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    const int n = int(shape[axis]);
    const int64_t stride_ax = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, n);

    auto process_range = [&](size_t begin, size_t end) {
        std::vector<int>   v(n);
        std::vector<float> ff(n), ranges(n + 1);
        info.for_each_line(begin, end, [&](size_t base) {
            squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                graph + base, output + base, n, stride_ax, axis_bit,
                anisotropy, black_border, v.data(), ff.data(), ranges.data()
            );
        });
    };

    dispatch_parallel(threads, info.first_ext, threads, process_range);
}

//-----------------------------------------------------------------------------
// Full EDT from Voxel Graph
//-----------------------------------------------------------------------------

template <typename GRAPH_T>
inline void edtsq_from_graph(
    const GRAPH_T* graph,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // Compute total voxels and C-order strides (stack array, supports up to 32D)
    size_t total = 1;
    size_t strides[32];
    for (size_t d = dims; d-- > 0;) {
        strides[d] = total;
        total *= shape[d];
    }
    if (total == 0) return;

    // Axis bit encoding: bit 0 = foreground; axis a -> bit (2*(dims-1-a)+1).
    // For 2D: axis 0 -> bit 3, axis 1 -> bit 1
    // For 3D: axis 0 -> bit 5, axis 1 -> bit 3, axis 2 -> bit 1

    // Process axes innermost-to-outermost for cache efficiency.
    // The innermost axis (axis = dims-1, stride=1) uses pass 0 (Rosenfeld-Pfaltz);
    // all remaining axes use the parabolic envelope pass.

    // Pass 0: innermost axis (always bit 1 in the graph encoding)
    {
        const size_t axis = dims - 1;
        edt_pass0_from_graph_direct_parallel<GRAPH_T>(
            graph, output,
            shape, strides, dims, axis, GRAPH_T(2),
            anisotropy[axis], black_border, parallel
        );
    }

    // Parabolic passes: axes dims-2 down to 0
    for (size_t axis = dims - 1; axis-- > 0;) {
        const GRAPH_T axis_bit = GRAPH_T(1) << (2 * (dims - 1 - axis) + 1);
        edt_pass_parabolic_from_graph_fused_parallel<GRAPH_T>(
            graph, output,
            shape, strides, dims, axis, axis_bit,
            anisotropy[axis], black_border, parallel
        );
    }
}

//-----------------------------------------------------------------------------
// Build graph from labels - SINGLE-PASS, unified ND algorithm
//
// 1D: dedicated linear scan.
// 2D+: unified ND path (chunk-based background skipping on innermost dim).
// Fixed internal arrays support up to 32D.
//-----------------------------------------------------------------------------

template <typename T, typename GRAPH_T = uint8_t>
inline void build_connectivity_graph(
    const T* labels,
    GRAPH_T* graph,
    const size_t* shape,
    const size_t dims,
    const int parallel
) {
    if (dims == 0) return;

    size_t voxels = 1;
    for (size_t d = 0; d < dims; d++) voxels *= shape[d];
    if (voxels == 0) return;

    const int threads = std::max(1, parallel);
    constexpr GRAPH_T FG = 0b00000001;  // Foreground bit (bit 0)

    //-------------------------------------------------------------------------
    // 1D path: simple linear scan
    //-------------------------------------------------------------------------
    if (dims == 1) {
        const int64_t n = shape[0];
        constexpr GRAPH_T BIT = 0b00000010;  // axis 0 bit for 1D

        auto process_1d = [=](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                const T label = labels[i];
                GRAPH_T g = (label != 0) ? FG : 0;
                if (label != 0 && i + 1 < (size_t)n && labels[i + 1] == label) {
                    g |= BIT;
                }
                graph[i] = g;
            }
        };
        dispatch_parallel((size_t)threads, (size_t)n, (size_t)threads, process_1d);
        return;
    }

    //-------------------------------------------------------------------------
    // Unified ND path for 2D+ - parallelize over first dimension with
    // chunk-based background skipping on the inner loop
    //-------------------------------------------------------------------------
    int64_t strides[32];
    int64_t shape64[32];
    GRAPH_T axis_bits[32];
    {
        int64_t s = 1;
        for (size_t d = dims; d-- > 0;) {
            strides[d] = s;
            shape64[d] = shape[d];
            s *= shape64[d];
        }
        for (size_t d = 0; d < dims; d++) {
            axis_bits[d] = GRAPH_T(1) << (2 * (dims - 1 - d) + 1);
        }
    }

    const int64_t first_extent = shape64[0];
    const int64_t first_stride = strides[0];
    const int64_t last_extent = shape64[dims - 1];
    const GRAPH_T last_bit = axis_bits[dims - 1];
    const GRAPH_T first_bit = axis_bits[0];

    // Middle dimensions product (dims 1 to dims-2); = 1 for 2D (empty product)
    int64_t mid_product = 1;
    for (size_t d = 1; d + 1 < dims; d++) {
        mid_product *= shape64[d];
    }

    // Number of middle dimensions (dims between first and last); 0 for 2D, 1 for 3D, etc.
    const size_t num_mid = dims - 2;

    constexpr int64_t CHUNK = 8;  // chunk size for background-skipping in inner loop

    // Process range of first dimension (outer loop) for 2D+
    auto process_dim0_range = [&](int64_t d0_start, int64_t d0_end) {
        // Thread-local storage for precomputed middle dimension info
        const T* mid_neighbor_row[30];  // Neighbor row pointers for middle dims (max 30 for 32D)
        bool mid_can_check[30];         // Whether we can check each mid neighbor
        GRAPH_T mid_bits[30];           // Bit to set for each mid dimension (constant per call)
        for (size_t md = 0; md < num_mid; md++)
            mid_bits[md] = axis_bits[md + 1];

        for (int64_t d0 = d0_start; d0 < d0_end; d0++) {
            const int64_t base0 = d0 * first_stride;
            const bool can_d0 = (d0 + 1 < first_extent);

            // Iterate middle dimensions (dims 1 to dims-2)
            int64_t mid_coords[30] = {0};  // For dims 1..dims-2 (max 30 for 32D)
            int64_t mid_offset = 0;

            for (int64_t m = 0; m < mid_product; m++) {
                const int64_t base = base0 + mid_offset;

                // Precompute row pointers for tight inner loop
                const T* row = labels + base;
                GRAPH_T* rowg = graph + base;
                const T* row_d0_next = can_d0 ? (labels + base + first_stride) : nullptr;

                // Precompute middle dimension neighbor info BEFORE inner loop
                for (size_t md = 0; md < num_mid; md++) {
                    const size_t d = md + 1;  // Actual dimension index
                    mid_can_check[md] = (mid_coords[md] + 1 < shape64[d]);
                    mid_neighbor_row[md] = mid_can_check[md] ? (labels + base + strides[d]) : nullptr;
                }

                // Inner loop over last dimension with chunk-based background skipping
                int64_t x = 0;
                const int64_t chunk_end = last_extent - (last_extent % CHUNK);
                for (; x < chunk_end; x += CHUNK) {
                    T any_fg = row[x]   | row[x+1] | row[x+2] | row[x+3] |
                               row[x+4] | row[x+5] | row[x+6] | row[x+7];
                    if (any_fg == 0) {
                        std::memset(rowg + x, 0, CHUNK * sizeof(GRAPH_T));
                    } else {
                        for (int64_t i = 0; i < CHUNK; i++) {
                            const int64_t xi = x + i;
                            const T label = row[xi];
                            GRAPH_T g = (label != 0) ? FG : 0;
                            if (label != 0) {
                                if (xi + 1 < last_extent && row[xi + 1] == label) g |= last_bit;
                                if (can_d0 && row_d0_next[xi] == label) g |= first_bit;
                                for (size_t md = 0; md < num_mid; md++) {
                                    if (mid_can_check[md] && mid_neighbor_row[md][xi] == label) g |= mid_bits[md];
                                }
                            }
                            rowg[xi] = g;
                        }
                    }
                }
                for (; x < last_extent; x++) {
                    const T label = row[x];
                    GRAPH_T g = (label != 0) ? FG : 0;
                    if (label != 0) {
                        if (x + 1 < last_extent && row[x + 1] == label) g |= last_bit;
                        if (can_d0 && row_d0_next[x] == label) g |= first_bit;
                        for (size_t md = 0; md < num_mid; md++) {
                            if (mid_can_check[md] && mid_neighbor_row[md][x] == label) g |= mid_bits[md];
                        }
                    }
                    rowg[x] = g;
                }

                // Increment mid coords; skip on last m iteration
                // (mid_coords is re-initialized for each d0 row, so
                //  the final increment before that reset is always wasted)
                if (m + 1 < mid_product) {
                    for (size_t d = dims - 2; d >= 1; d--) {
                        mid_coords[d - 1]++;
                        mid_offset += strides[d];
                        if (mid_coords[d - 1] < shape64[d]) break;
                        mid_offset -= mid_coords[d - 1] * strides[d];
                        mid_coords[d - 1] = 0;
                    }
                }
            }
        }
    };

    dispatch_parallel((size_t)threads, (size_t)first_extent, (size_t)threads,
        [&](size_t s, size_t e) { process_dim0_range((int64_t)s, (int64_t)e); });
}

//-----------------------------------------------------------------------------
// Fused labels-to-EDT: Build graph internally, run EDT, free graph
// This is more efficient than separate Python calls because:
// 1. No Python/Cython overhead between build and EDT
// 2. Graph memory is allocated and freed in C++ (faster)
// 3. Thread pool is already warm from graph build
//-----------------------------------------------------------------------------

// Internal: allocate graph of type GRAPH_T, build connectivity, run EDT.
// `total` (precomputed by caller) is passed to avoid recomputing for the allocation.
template <typename T, typename GRAPH_T>
inline void _edtsq_fused_typed(
    const T* labels, float* output, const size_t* shape,
    const float* anisotropy, const size_t dims,
    const bool black_border, const int parallel, const size_t total
) {
    std::unique_ptr<GRAPH_T[]> graph(new GRAPH_T[total]);
    build_connectivity_graph<T, GRAPH_T>(labels, graph.get(), shape, dims, parallel);
    edtsq_from_graph<GRAPH_T>(graph.get(), output, shape, anisotropy, dims, black_border, parallel);
}

template <typename T>
inline void edtsq_from_labels_fused(
    const T* labels,
    float* output,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;
    size_t total = 1;
    for (size_t d = 0; d < dims; d++) total *= shape[d];
    if (total == 0) return;

    // Graph type: smallest unsigned integer fitting 2*(dims-1)+1 bits.
    // uint8 <=4D (max bit 7), uint16 <=8D (max bit 15),
    // uint32 <=16D (max bit 31), uint64 <=32D (max bit 63).
    if      (dims <=  4) _edtsq_fused_typed<T, uint8_t> (labels, output, shape, anisotropy, dims, black_border, parallel, total);
    else if (dims <=  8) _edtsq_fused_typed<T, uint16_t>(labels, output, shape, anisotropy, dims, black_border, parallel, total);
    else if (dims <= 16) _edtsq_fused_typed<T, uint32_t>(labels, output, shape, anisotropy, dims, black_border, parallel, total);
    else                 _edtsq_fused_typed<T, uint64_t>(labels, output, shape, anisotropy, dims, black_border, parallel, total);
}

//-----------------------------------------------------------------------------
// Parabolic EDT with argmin tracking (for expand_labels/feature_transform)
//-----------------------------------------------------------------------------

// Workspace arrays (v_ws[n], ff_ws[n], ranges_ws[n+1]) must be caller-allocated.
// Callers should allocate once per chunk and reuse across scanlines to avoid
// repeated heap churn inside tight per-line loops.
// Workspace does not need to be initialized by the caller; this function sets
// all required entries before use.
inline void squared_edt_1d_parabolic_with_arg_stride(
    float* f,
    const size_t n,
    const size_t stride,
    const float anisotropy,
    const bool black_border,
    size_t* arg_out,
    int* v_ws,
    float* ff_ws,
    float* ranges_ws
) {
    if (n == 0) return;
    const int nn = int(n);
    const float w2 = anisotropy * anisotropy;

    int k = 0;
    int* v = v_ws;
    float* ff = ff_ws;
    float* ranges = ranges_ws;
    v[0] = 0;  // invariant: envelope starts at position 0
    for (int i = 0; i < nn; i++) ff[i] = f[i * stride];
    ranges[0] = -std::numeric_limits<float>::infinity();
    ranges[1] = std::numeric_limits<float>::infinity();

    // Use double arithmetic for the same cancellation-resistance as process_large_run.
    auto intersect = [&](int a, int b) -> float {
        const double d1 = double(b - a) * double(w2);
        return float((double(ff[b]) - double(ff[a]) + d1 * double(a + b)) / (2.0 * d1));
    };

    float s;
    for (int i = 1; i < nn; i++) {
        s = intersect(v[k], i);
        while (k > 0 && s <= ranges[k]) {
            k--;
            s = intersect(v[k], i);
        }
        k++;
        v[k] = i;
        ranges[k] = s;
        ranges[k + 1] = std::numeric_limits<float>::infinity();
    }

    // Output pass: hoisted outside loop to avoid per-iteration branch.
    k = 0;
    if (black_border) {
        for (int i = 0; i < nn; i++) {
            while (ranges[k + 1] < i) k++;
            const float parabola = w2 * sq(i - v[k]) + ff[v[k]];
            const float border   = w2 * std::fminf(sq(i + 1), sq(nn - i));
            f[i * stride] = std::fminf(border, parabola);
            arg_out[i] = v[k];
        }
    } else {
        for (int i = 0; i < nn; i++) {
            while (ranges[k + 1] < i) k++;
            f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
            arg_out[i] = v[k];
        }
    }
}

//-----------------------------------------------------------------------------
// Expand labels helpers (for ND expand_labels/feature_transform)
//-----------------------------------------------------------------------------

template <typename INDEX=size_t>
inline void _nd_expand_init_bases(
    const uint8_t* seeds,
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    INDEX* feat_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);

    auto process_chunk = [&](size_t start, size_t end) {
        std::vector<size_t> arg(n);
        std::vector<int>   v_ws(n);
        std::vector<float> ff_ws(n), ranges_ws(n + 1);
        for (size_t i = start; i < end; ++i) {
            const size_t base = bases[i];
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) {
                const bool seeded = (seeds[base + j * s] != 0);
                dist[base + j * s] = seeded ? 0.0f : (std::numeric_limits<float>::max() / 4.0f);
                any_nonseed |= (!seeded);
            }
            if (any_nonseed) {
                squared_edt_1d_parabolic_with_arg_stride(
                    dist + base, n, s, anis,
                    black_border,
                    arg.data(),
                    v_ws.data(), ff_ws.data(), ranges_ws.data());
                for (size_t j = 0; j < n; ++j)
                    feat_out[base + j * s] = (INDEX)(base + arg[j] * s);
            } else {
                // All seeds: each position is its own nearest seed
                for (size_t j = 0; j < n; ++j)
                    feat_out[base + j * s] = (INDEX)(base + j * s);
            }
        }
    };

    dispatch_parallel(threads, num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

template <typename INDEX=size_t>
inline void _nd_expand_parabolic_bases(
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    INDEX* feat,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);

    auto process_chunk = [&](size_t start, size_t end) {
        std::vector<size_t> arg(n);
        std::vector<INDEX> feat_line(n);
        std::vector<int>   v_ws(n);
        std::vector<float> ff_ws(n), ranges_ws(n + 1);
        for (size_t i = start; i < end; ++i) {
            const size_t base = bases[i];
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) any_nonseed |= (dist[base + j * s] != 0.0f);
            if (!any_nonseed) continue;  // all seeds: dist=0 everywhere, feat already holds self-references
            for (size_t j = 0; j < n; ++j) feat_line[j] = feat[base + j * s];
            squared_edt_1d_parabolic_with_arg_stride(
                dist + base, n, s, anis,
                black_border,
                arg.data(),
                v_ws.data(), ff_ws.data(), ranges_ws.data());
            for (size_t j = 0; j < n; ++j) {
                feat[base + j * s] = feat_line[arg[j]];
            }
        }
    };

    dispatch_parallel(threads, num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

inline void _nd_expand_init_labels_bases(
    const uint8_t* seeds,
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    const uint32_t* labels_in,
    uint32_t* label_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);

    auto process_chunk = [&](size_t start, size_t end) {
        std::vector<size_t> arg(n);
        std::vector<int>   v_ws(n);
        std::vector<float> ff_ws(n), ranges_ws(n + 1);
        for (size_t i = start; i < end; ++i) {
            const size_t base = bases[i];
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) {
                const bool seeded = (seeds[base + j * s] != 0);
                dist[base + j * s] = seeded ? 0.0f : (std::numeric_limits<float>::max() / 4.0f);
                any_nonseed |= (!seeded);
            }
            if (any_nonseed) {
                squared_edt_1d_parabolic_with_arg_stride(
                    dist + base, n, s, anis,
                    black_border,
                    arg.data(),
                    v_ws.data(), ff_ws.data(), ranges_ws.data());
                for (size_t j = 0; j < n; ++j)
                    label_out[base + j * s] = labels_in[base + arg[j] * s];
            } else {
                // All seeds: copy labels unchanged (each position is its own nearest seed)
                for (size_t j = 0; j < n; ++j)
                    label_out[base + j * s] = labels_in[base + j * s];
            }
        }
    };

    dispatch_parallel(threads, num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

inline void _nd_expand_parabolic_labels_bases(
    float* dist,
    const size_t* bases,
    const size_t num_lines,
    const size_t n,
    const size_t s,
    const float anis,
    const bool black_border,
    const uint32_t* labels_in,
    uint32_t* label_out,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const int threads = std::max(1, parallel);

    auto process_chunk = [&](size_t start, size_t end) {
        std::vector<size_t> arg(n);
        std::vector<int>   v_ws(n);
        std::vector<float> ff_ws(n), ranges_ws(n + 1);
        for (size_t i = start; i < end; ++i) {
            const size_t base = bases[i];
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) any_nonseed |= (dist[base + j * s] != 0.0f);
            if (any_nonseed) {
                squared_edt_1d_parabolic_with_arg_stride(
                    dist + base, n, s, anis,
                    black_border,
                    arg.data(),
                    v_ws.data(), ff_ws.data(), ranges_ws.data());
                for (size_t j = 0; j < n; ++j) {
                    label_out[base + j * s] = labels_in[base + arg[j] * s];
                }
            } else {
                // All seeds: copy labels unchanged (self-reference)
                for (size_t j = 0; j < n; ++j) {
                    label_out[base + j * s] = labels_in[base + j * s];
                }
            }
        }
    };

    dispatch_parallel(threads, num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

//-----------------------------------------------------------------------------
// Fused expand_labels orchestration (C++ replacement for Cython orchestration)
//
// expand_labels_fused       - labels-only output
// expand_labels_features_fused - labels + feature indices output
//
// The four _nd_expand_*_bases helpers above do the per-axis work;
// these functions supply the axis loop, base computation, and memory.
//-----------------------------------------------------------------------------

// Fill ax_ord[0..count-1] with all axis indices except exclude_ax, sorted by
// stride (innermost/smallest first), longer axis as tiebreaker.
// count = dims-1 when exclude_ax is a valid axis; dims when exclude_ax==dims (no exclusion).
inline void _expand_fill_sort_ax_ord(
    size_t* ax_ord,
    const size_t exclude_ax,
    const size_t* shape,
    const size_t* strides,
    const size_t dims
) {
    size_t op = 0;
    for (size_t ii = 0; ii < dims; ++ii)
        if (ii != exclude_ax) ax_ord[op++] = ii;
    for (size_t i = 1; i < op; ++i) {
        size_t key = ax_ord[i];
        int j = (int)i - 1;
        while (j >= 0 && (strides[ax_ord[j]] > strides[key] ||
               (strides[ax_ord[j]] == strides[key] && shape[ax_ord[j]] < shape[key]))) {
            ax_ord[j + 1] = ax_ord[j];
            --j;
        }
        ax_ord[j + 1] = key;
    }
}

// Sort all dims axis indices into paxes[0..dims-1], innermost first.
// Delegates to _expand_fill_sort_ax_ord with no exclusion (exclude_ax=dims).
inline void _expand_sort_axes(
    size_t* paxes,
    const size_t* shape,
    const size_t* strides,
    const size_t dims
) {
    _expand_fill_sort_ax_ord(paxes, dims, shape, strides, dims);
}

// Compute start offsets for each of `num_lines` scanlines, given the axes
// in ax_ord[0..ord_len-1], filling bases[0..num_lines-1].
// Uses a carry-ripple counter (ax_ord[0] varies fastest) to avoid integer
// division; amortized O(1) per line instead of O(ord_len) divisions.
inline void _expand_compute_bases(
    size_t* bases,
    const size_t* ax_ord,
    const size_t* shape,
    const size_t* strides,
    const size_t ord_len,
    const size_t num_lines
) {
    size_t coords[32] = {};
    size_t base = 0;
    for (size_t il = 0; il < num_lines; ++il) {
        bases[il] = base;
        for (size_t j = 0; j < ord_len; ++j) {
            coords[j]++;
            base += strides[ax_ord[j]];
            if (coords[j] < shape[ax_ord[j]]) break;
            base -= coords[j] * strides[ax_ord[j]];
            coords[j] = 0;
        }
    }
}

// Collect seed positions and pairwise midpoints for a 1D expand_labels pass.
// Returns false (seeds/mids left empty) when no seeds exist.
template <typename T>
inline bool _expand_1d_setup(
    const T* data, const size_t n,
    std::vector<size_t>& seeds, std::vector<double>& mids
) {
    for (size_t i = 0; i < n; ++i)
        if (data[i] != 0) seeds.push_back(i);
    if (seeds.empty()) return false;
    mids.resize(seeds.size() - 1);
    for (size_t i = 0; i < mids.size(); ++i)
        mids[i] = (seeds[i] + seeds[i + 1]) * 0.5;
    return true;
}

// Compute C-order strides, axis processing order, total element count,
// and maximum scanline count for expand_labels ND passes.
// strides[] and paxes[] must be caller-allocated arrays of size >= dims.
// If total == 0, max_lines is set to 0 and strides/paxes are left uninitialized.
inline void _expand_nd_compute_layout(
    const size_t* shape, const size_t dims,
    size_t* strides, size_t* paxes,
    size_t& total, size_t& max_lines
) {
    total = 1;
    size_t s = 1;
    for (size_t d = dims; d-- > 0;) {
        strides[d] = s;
        s *= shape[d];
        total *= shape[d];
    }
    if (total == 0) { max_lines = 0; return; }
    _expand_sort_axes(paxes, shape, strides, dims);
    size_t min_shape = shape[0];
    for (size_t d = 1; d < dims; ++d)
        if (shape[d] < min_shape) min_shape = shape[d];
    max_lines = total / min_shape;  // max scanlines over all axis passes
}

// labels-only mode
template <typename T>
inline void expand_labels_fused(
    const T* data,
    uint32_t* labels_out,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // 1D path
    if (dims == 1) {
        const size_t n = shape[0];
        if (n == 0) return;
        std::vector<size_t> seeds;
        std::vector<double> mids;
        if (!_expand_1d_setup(data, n, seeds, mids)) {
            std::fill(labels_out, labels_out + n, uint32_t(0));
            return;
        }
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            while (k < mids.size() && (double)i >= mids[k]) ++k;
            const size_t si = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                // Border acts as a virtual seed at distance min(i+1, n-i).
                // If the border is at least as close as the real seed, label = 0.
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= si) ? (i - si) : (si - i);
                if (border_dist <= seed_dist) { labels_out[i] = 0; continue; }
            }
            labels_out[i] = (uint32_t)data[si];
        }
        return;
    }

    // ND path
    size_t total, max_lines;
    size_t strides[32], paxes[32];
    _expand_nd_compute_layout(shape, dims, strides, paxes, total, max_lines);
    if (total == 0) return;

    // Heap allocations
    std::vector<float>    dist(total);
    std::vector<size_t>   bases(max_lines);
    std::vector<uint32_t> lab_prev(total);
    std::vector<uint32_t> lab_next(total);

    // Build seeds_flat and labels_flat from data
    std::vector<uint8_t>  seeds_flat(total);
    std::vector<uint32_t> labels_flat(total);
    for (size_t i = 0; i < total; ++i) {
        seeds_flat[i]  = (data[i] != 0) ? 1 : 0;
        labels_flat[i] = (uint32_t)data[i];
    }

    size_t ax_ord[32];  // stack; dims-1 entries used per axis pass
    for (size_t a = 0; a < dims; ++a) {
        const size_t ax  = paxes[a];
        const size_t n0  = shape[ax];
        const size_t s0  = strides[ax];
        const size_t lines = total / n0;
        const float  anis  = anisotropy[ax];

        _expand_fill_sort_ax_ord(ax_ord, ax, shape, strides, dims);
        _expand_compute_bases(bases.data(), ax_ord, shape, strides, dims - 1, lines);

        if (a == 0) {
            _nd_expand_init_labels_bases(
                seeds_flat.data(), dist.data(), bases.data(),
                lines, n0, s0, anis, black_border,
                labels_flat.data(), lab_prev.data(), parallel);
        } else {
            _nd_expand_parabolic_labels_bases(
                dist.data(), bases.data(),
                lines, n0, s0, anis, black_border,
                lab_prev.data(), lab_next.data(), parallel);
            std::swap(lab_prev, lab_next);
        }
    }

    std::copy(lab_prev.begin(), lab_prev.end(), labels_out);
}

// labels + feature indices mode
template <typename T, typename INDEX>
inline void expand_labels_features_fused(
    const T* data,
    uint32_t* labels_out,
    INDEX* features_out,
    const size_t* shape,
    const float* anisotropy,
    const size_t dims,
    const bool black_border,
    const int parallel
) {
    if (dims == 0) return;

    // 1D path
    if (dims == 1) {
        const size_t n = shape[0];
        if (n == 0) return;
        std::vector<size_t> seeds;
        std::vector<double> mids;
        if (!_expand_1d_setup(data, n, seeds, mids)) {
            std::fill(labels_out, labels_out + n, uint32_t(0));
            std::fill(features_out, features_out + n, INDEX(0));
            return;
        }
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            while (k < mids.size() && (double)i >= mids[k]) ++k;
            const size_t si = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                // Border acts as a virtual seed at distance min(i+1, n-i).
                // If the border is at least as close as the real seed, label = 0.
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= si) ? (i - si) : (si - i);
                if (border_dist <= seed_dist) {
                    labels_out[i]   = 0;
                    features_out[i] = INDEX(si);  // nearest real seed, same as ND path
                    continue;
                }
            }
            labels_out[i]   = (uint32_t)data[si];
            features_out[i] = INDEX(si);
        }
        return;
    }

    // ND path
    size_t total, max_lines;
    size_t strides[32], paxes[32];
    _expand_nd_compute_layout(shape, dims, strides, paxes, total, max_lines);
    if (total == 0) return;

    std::vector<float>    dist(total);
    std::vector<size_t>   bases(max_lines);
    std::vector<uint8_t>  seeds_flat(total);
    std::vector<uint32_t> labels_flat(total);

    for (size_t i = 0; i < total; ++i) {
        seeds_flat[i]  = (data[i] != 0) ? 1 : 0;
        labels_flat[i] = (uint32_t)data[i];
    }

    size_t ax_ord[32];
    for (size_t a = 0; a < dims; ++a) {
        const size_t ax  = paxes[a];
        const size_t n0  = shape[ax];
        const size_t s0  = strides[ax];
        const size_t lines = total / n0;
        const float  anis  = anisotropy[ax];

        _expand_fill_sort_ax_ord(ax_ord, ax, shape, strides, dims);
        _expand_compute_bases(bases.data(), ax_ord, shape, strides, dims - 1, lines);

        if (a == 0) {
            _nd_expand_init_bases<INDEX>(
                seeds_flat.data(), dist.data(), bases.data(),
                lines, n0, s0, anis, black_border,
                features_out, parallel);
        } else {
            _nd_expand_parabolic_bases<INDEX>(
                dist.data(), bases.data(),
                lines, n0, s0, anis, black_border,
                features_out, parallel);
        }
    }

    // Resolve feature indices to labels
    for (size_t i = 0; i < total; ++i)
        labels_out[i] = labels_flat[(size_t)features_out[i]];
}

} // namespace nd

#endif // EDT_HPP
