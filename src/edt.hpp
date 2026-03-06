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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "threadpool.h"

namespace nd {

// Tuning parameter: more chunks = better load balancing with atomic work-stealing
static size_t ND_CHUNKS_PER_THREAD = 4;

inline void set_tuning(size_t chunks_per_thread) {
    if (chunks_per_thread > 0) ND_CHUNKS_PER_THREAD = chunks_per_thread;
}

// Shared fork-join pool keyed by thread count; created lazily on first use
inline ForkJoinPool& shared_pool_for(size_t threads) {
    static std::mutex mutex;
    static std::unordered_map<size_t, std::unique_ptr<ForkJoinPool>> pools;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = pools[threads];
    if (!entry) {
        entry = std::make_unique<ForkJoinPool>(threads);
    }
    return *entry;
}

// Per-pass thread cap: further limits threads based on work in a single EDT axis pass.
// This is a C++-level inner cap applied per axis pass; the caller-supplied `desired`
// is already capped at the Python level by _adaptive_thread_limit_nd.
inline size_t compute_threads(size_t desired, size_t total_lines, size_t axis_len) {
    if (desired <= 1 || total_lines <= 1) return 1;

    size_t threads = std::min<size_t>(desired, total_lines);

    // Further cap based on work per pass (total_work = voxels along this axis sweep)
    const size_t total_work = axis_len * total_lines;
    if (total_work <= 60000) {
        threads = std::min<size_t>(threads, 4);   // small pass: diminishing returns above 4T
    } else if (total_work <= 120000) {
        threads = std::min<size_t>(threads, 8);   // medium pass: cap at 8T
    } else if (total_work <= 400000) {
        threads = std::min<size_t>(threads, 12);  // large pass: cap at 12T
    }

    return std::max<size_t>(1, threads);
}

// Static buffer cache for expand_labels — avoids repeated allocation/page-fault
// overhead on repeated calls (like ncolor's module-level np.empty() globals).
// Each slot independently tracks its allocation size and reuses if sufficient.
struct ExpandBufCache {
    static constexpr int N_SLOTS = 8;
    void* bufs[N_SLOTS] = {};
    size_t sizes[N_SLOTS] = {};

    void* get(int slot, size_t bytes) {
        if (bytes <= sizes[slot]) return bufs[slot];
        std::free(bufs[slot]);
        bufs[slot] = std::malloc(bytes);
        sizes[slot] = bytes;
        return bufs[slot];
    }
    ~ExpandBufCache() {
        for (int i = 0; i < N_SLOTS; i++) std::free(bufs[i]);
    }
};

inline ExpandBufCache& expand_cache() {
    static ExpandBufCache cache;
    return cache;
}

// Distribute [0, total) into up to max_chunks chunks across threads.
// Calls work(begin, end) directly when threads==1; otherwise via shared pool.
// Uses atomic work-stealing: each thread claims chunks via fetch_add.
// Blocks until all chunks complete.
template <typename F>
inline void dispatch_parallel(size_t threads, size_t total, size_t max_chunks, F work) {
    if (threads <= 1 || total == 0) {
        work(size_t(0), total);
        return;
    }
    const size_t n_chunks = std::min(max_chunks, total);
    const size_t chunk_sz = (total + n_chunks - 1) / n_chunks;
    std::atomic<size_t> next{0};
    ForkJoinPool& pool = shared_pool_for(threads);
    pool.parallel([&]() {
        size_t idx;
        while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const size_t begin = idx * chunk_sz;
            const size_t end = std::min(total, begin + chunk_sz);
            work(begin, end);
        }
    });
}

// Precomputed per-pass iteration layout for an EDT axis pass.
// Gathers all "other" (non-axis) dimensions and their strides, and
// exposes for_each_line() to iterate every scanline in a slice range.
struct AxisPassInfo {
    size_t num_other = 0;   // number of non-axis dims
    size_t other_extents[32];   // extents of non-axis dims (in shape order)
    size_t other_strides[32];   // strides of non-axis dims
    size_t total_lines = 1; // product of all other extents
    size_t first_extent  = 1;  // extent of first other dim  (parallelized over)
    size_t first_stride  = 0;  // stride of first other dim
    size_t rest_prod  = 1;  // product of other_extents[1..num_other-1]

    AxisPassInfo(const size_t* shape, const size_t* strides,
                 size_t dims, size_t axis) {
        for (size_t d = 0; d < dims; d++) {
            if (d == axis) continue;
            other_extents[num_other] = shape[d];
            other_strides[num_other] = strides[d];
            total_lines *= shape[d];
            num_other++;
        }
        if (num_other > 0) {
            first_extent = other_extents[0];
            first_stride = other_strides[0];
            for (size_t d = 1; d < num_other; d++)
                rest_prod *= other_extents[d];
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
                fn(i0 * first_stride);
        } else {
            // ND path: iterate the inner dims with a multi-dim counter.
            // coords reused across i0 rows; invariant: all-zero at start of each row.
            size_t coords[32] = {};
            for (size_t i0 = begin; i0 < end; i0++) {
                size_t base = i0 * first_stride;
                for (size_t i = 0; i < rest_prod; i++) {
                    fn(base);
                    for (size_t d = 1; d < num_other; d++) {
                        coords[d]++;
                        base += other_strides[d];
                        if (coords[d] < other_extents[d]) break;
                        base -= coords[d] * other_strides[d];
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

    const float wsq = anisotropy * anisotropy;
    int i = 0;

    while (i < n) {
        // Check if this voxel is background (graph == 0)
        if (graph[i * stride] == 0) {
            d[i * stride] = 0.0f;
            i++;
            continue;
        }

        // Foreground: find segment extent using connectivity bits
        const int seg_start = i;
        GRAPH_T edge = graph[i * stride];
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
        const bool left_border = (seg_start > 0) || black_border;
        const bool right_border = (i < n) || black_border;

        // Forward pass: squared distance from left border
        if (left_border) {
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = wsq * sq(k + 1);
            }
        } else {
            const float inf = std::numeric_limits<float>::infinity();
            for (int k = 0; k < seg_len; k++) {
                d[(seg_start + k) * stride] = inf;
            }
        }

        // Backward pass: take min with squared distance from right border
        if (right_border) {
            for (int k = seg_len - 1; k >= 0; k--) {
                const float v_sq = wsq * sq(seg_len - k);
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
    const int64_t axis_stride = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, (size_t)n);

    auto process_range = [&](size_t begin, size_t end) {
        info.for_each_line(begin, end, [&](size_t base) {
            squared_edt_1d_from_graph_direct<GRAPH_T>(
                graph + base, output + base,
                n, axis_stride, axis_bit, anisotropy, black_border
            );
        });
    };

    dispatch_parallel(threads, info.first_extent, threads, process_range);
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
    const float wsq = anisotropy * anisotropy;

    // Fast path for small segments: O(n²) brute force
    auto process_small_run = [&](int start, int len, bool left_border, bool right_border) {
        float original[SMALL_THRESHOLD];
        for (int q = 0; q < len; ++q) {
            original[q] = f[(start + q) * stride];
        }
        for (int j = 0; j < len; ++j) {
            float best = original[j];
            if (left_border) {
                const float cap_left = wsq * sq(j + 1);
                if (cap_left < best) best = cap_left;
            }
            if (right_border) {
                const float cap_right = wsq * sq(len - j);
                if (cap_right < best) best = cap_right;
            }
            for (int q = 0; q < len; ++q) {
                const float candidate = original[q] + wsq * sq(j - q);
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
            const double d1 = double(b - a) * double(wsq);
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

        // Output pass: use specialized loops to avoid per-iteration conditionals
        k = 0;
        if (left_border && right_border) {
            // Both borders: take min of border distances and parabolic result
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                const float parabola = wsq * sq(i - v[k]) + ff[v[k]];
                const float border   = wsq * std::fminf(sq(i + 1), sq(len - i));
                f[(start + i) * stride] = std::fminf(border, parabola);
            }
        } else if (left_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(wsq * sq(i + 1), wsq * sq(i - v[k]) + ff[v[k]]);
            }
        } else if (right_border) {
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = std::fminf(wsq * sq(len - i), wsq * sq(i - v[k]) + ff[v[k]]);
            }
        } else {
            // No borders - just parabolic result
            for (int i = 0; i < len; i++) {
                while (ranges[k + 1] < i) k++;
                f[(start + i) * stride] = wsq * sq(i - v[k]) + ff[v[k]];
            }
        }
    };

    // Scan graph to find foreground segments (single pass)
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
            const bool left_border = (seg_start > 0) || black_border;
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
    const bool left_border = (seg_start > 0) || black_border;
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
    const int64_t axis_stride = strides[axis];
    if (n == 0) return;

    const AxisPassInfo info(shape, strides, dims, axis);
    const size_t threads = compute_threads(parallel, info.total_lines, (size_t)n);

    auto process_range = [&](size_t begin, size_t end) {
        std::vector<int>   v(n);
        std::vector<float> ff(n), ranges(n + 1);
        info.for_each_line(begin, end, [&](size_t base) {
            squared_edt_1d_parabolic_from_graph_ws<GRAPH_T>(
                graph + base, output + base, n, axis_stride, axis_bit,
                anisotropy, black_border, v.data(), ff.data(), ranges.data()
            );
        });
    };

    dispatch_parallel(threads, info.first_extent, threads, process_range);
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
        const GRAPH_T axis_bit = GRAPH_T(1) << 1;  // innermost axis: bit 1 of graph encoding
        edt_pass0_from_graph_direct_parallel<GRAPH_T>(
            graph, output,
            shape, strides, dims, axis, axis_bit,
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
// Build connectivity graph from labels (single-pass, unified ND algorithm)
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

    size_t total = 1;
    for (size_t d = 0; d < dims; d++) total *= shape[d];
    if (total == 0) return;

    const int threads = std::max(1, parallel);
    constexpr GRAPH_T fg_bit = 0b00000001;  // Foreground bit (bit 0)

    //-------------------------------------------------------------------------
    // 1D path: simple linear scan
    //-------------------------------------------------------------------------
    if (dims == 1) {
        const size_t n = shape[0];
        constexpr GRAPH_T axis_bit = 0b00000010;  // axis 0 bit for 1D

        auto process_1d = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                const T label = labels[i];
                GRAPH_T g = (label != 0) ? fg_bit : 0;
                if (label != 0 && i + 1 < n && labels[i + 1] == label) {
                    g |= axis_bit;
                }
                graph[i] = g;
            }
        };
        dispatch_parallel((size_t)threads, n, (size_t)threads, process_1d);
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
    // Safe: dims >= 2 is guaranteed by the dims == 1 early return above.
    const size_t num_mid = dims - 2;

    constexpr int64_t CHUNK = 8;  // chunk size for background-skipping in inner loop

    // Process range of first dimension (outer loop) for 2D+
    auto process_dim0_range = [&](int64_t d0_start, int64_t d0_end) {
        // Thread-local storage for precomputed middle dimension info
        const T* mid_neighbor_row[30];  // Neighbor row pointers for middle dims (max 30 for 32D)
        bool mid_can_check[30];         // Whether we can check each mid neighbor
        GRAPH_T mid_bits[30];           // Bit to set for each mid dimension (constant per call)
        for (size_t mid = 0; mid < num_mid; mid++)
            mid_bits[mid] = axis_bits[mid + 1];

        for (int64_t d0 = d0_start; d0 < d0_end; d0++) {
            const int64_t base0 = d0 * first_stride;
            const bool can_d0 = (d0 + 1 < first_extent);

            // Iterate middle dimensions (dims 1 to dims-2)
            int64_t mid_coords[30] = {0};  // For dims 1..dims-2 (max 30 for 32D)
            int64_t mid_offset = 0;

            for (int64_t mid = 0; mid < mid_product; mid++) {
                const int64_t base = base0 + mid_offset;

                // Precompute row pointers for tight inner loop
                const T* row = labels + base;
                GRAPH_T* rowg = graph + base;
                const T* row_d0_next = can_d0 ? (labels + base + first_stride) : nullptr;

                // Precompute middle dimension neighbor info BEFORE inner loop
                for (size_t mid = 0; mid < num_mid; mid++) {
                    const size_t d = mid + 1;  // Actual dimension index
                    mid_can_check[mid] = (mid_coords[mid] + 1 < shape64[d]);
                    mid_neighbor_row[mid] = mid_can_check[mid] ? (labels + base + strides[d]) : nullptr;
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
                            GRAPH_T g = (label != 0) ? fg_bit : 0;
                            if (label != 0) {
                                if (xi + 1 < last_extent && row[xi + 1] == label) g |= last_bit;
                                if (can_d0 && row_d0_next[xi] == label) g |= first_bit;
                                for (size_t mid = 0; mid < num_mid; mid++) {
                                    if (mid_can_check[mid] && mid_neighbor_row[mid][xi] == label) g |= mid_bits[mid];
                                }
                            }
                            rowg[xi] = g;
                        }
                    }
                }
                for (; x < last_extent; x++) {
                    const T label = row[x];
                    GRAPH_T g = (label != 0) ? fg_bit : 0;
                    if (label != 0) {
                        if (x + 1 < last_extent && row[x + 1] == label) g |= last_bit;
                        if (can_d0 && row_d0_next[x] == label) g |= first_bit;
                        for (size_t mid = 0; mid < num_mid; mid++) {
                            if (mid_can_check[mid] && mid_neighbor_row[mid][x] == label) g |= mid_bits[mid];
                        }
                    }
                    rowg[x] = g;
                }

                // Increment mid coords; skip on last mid iteration
                // (mid_coords is re-initialized for each d0 row, so
                //  the final increment before that reset is always wasted)
                if (mid + 1 < mid_product) {
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
        [&](size_t begin, size_t end) { process_dim0_range((int64_t)begin, (int64_t)end); });
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

//=============================================================================
// Expand labels: gather/scatter pipeline with seed-skipping
//=============================================================================

// Sort all axes by stride ascending (innermost first)
inline void _expand_sort_axes(
    size_t* paxes,
    const size_t* shape,
    const size_t* strides,
    const size_t dims
) {
    for (size_t d = 0; d < dims; ++d) paxes[d] = d;
    for (size_t i = 1; i < dims; ++i) {
        size_t key = paxes[i];
        int j = (int)i - 1;
        while (j >= 0 && (strides[paxes[j]] > strides[key] ||
               (strides[paxes[j]] == strides[key] && shape[paxes[j]] < shape[key]))) {
            paxes[j + 1] = paxes[j];
            --j;
        }
        paxes[j + 1] = key;
    }
}

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

//-----------------------------------------------------------------------------
// Pass 0: seed-skipping + midpoint optimization (L2)
// All seeds have dist=0, so all intersections are midpoints (a+b)/2.
//-----------------------------------------------------------------------------

inline void _expand_pass0(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wsq = anis * anis;
    const float HUGE_DIST = std::numeric_limits<float>::max() / 4.0f;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int> v(n);
        std::vector<uint32_t> lbl_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;

            int n_seeds = 0;
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) {
                if (ll[j] != 0) {
                    dd[j] = 0.0f;
                    v[n_seeds++] = (int)j;
                } else {
                    dd[j] = HUGE_DIST;
                    any_nonseed = true;
                }
            }
            if (!any_nonseed) continue;
            if (n_seeds == 0) {
                // No seeds: with black_border, fill dist with border distances
                // so subsequent passes see realistic distances. Labels stay 0.
                if (black_border) {
                    for (size_t i = 0; i < n; ++i)
                        dd[i] = wsq * std::fminf(sq((int)i + 1), sq((int)n - (int)i));
                }
                continue;
            }

            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));

            int k = 0;
            if (black_border) {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    const float envelope = wsq * sq((int)i - v[k]);
                    const float border = wsq * std::fminf(sq((int)i + 1), sq((int)n - (int)i));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                }
            } else {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    dd[i] = wsq * sq((int)i - v[k]);
                    ll[i] = lbl_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

template <typename INDEX>
inline void _expand_pass0_feat(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    INDEX* __restrict__ feat,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wsq = anis * anis;
    const float HUGE_DIST = std::numeric_limits<float>::max() / 4.0f;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int> v(n);
        std::vector<uint32_t> lbl_save(n);
        std::vector<INDEX> feat_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;
            INDEX*    ff = feat + line * n;

            int n_seeds = 0;
            bool any_nonseed = false;
            for (size_t j = 0; j < n; ++j) {
                if (ll[j] != 0) {
                    dd[j] = 0.0f;
                    v[n_seeds++] = (int)j;
                } else {
                    dd[j] = HUGE_DIST;
                    any_nonseed = true;
                }
            }
            if (!any_nonseed) continue;
            if (n_seeds == 0) {
                if (black_border) {
                    for (size_t i = 0; i < n; ++i)
                        dd[i] = wsq * std::fminf(sq((int)i + 1), sq((int)n - (int)i));
                }
                continue;
            }

            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));
            std::memcpy(feat_save.data(), ff, n * sizeof(INDEX));

            int k = 0;
            if (black_border) {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    const float envelope = wsq * sq((int)i - v[k]);
                    const float border = wsq * std::fminf(sq((int)i + 1), sq((int)n - (int)i));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                    ff[i] = feat_save[v[k]];
                }
            } else {
                for (size_t i = 0; i < n; ++i) {
                    while (k + 1 < n_seeds &&
                           (double)i > (double)(v[k] + v[k + 1]) * 0.5) ++k;
                    dd[i] = wsq * sq((int)i - v[k]);
                    ll[i] = lbl_save[v[k]];
                    ff[i] = feat_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

//-----------------------------------------------------------------------------
// Passes 1+: standard L2 envelope on contiguous (num_lines, n) data.
//-----------------------------------------------------------------------------

inline void _expand_parabolic(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wsq = anis * anis;
    const int nn = (int)n;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int>      v(n);
        std::vector<float>    ff(n), ranges(n + 1);
        std::vector<uint32_t> lbl_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;

            bool any_nonzero = false;
            for (size_t j = 0; j < n; ++j) {
                if (dd[j] != 0.0f) { any_nonzero = true; break; }
            }
            if (!any_nonzero) continue;

            std::memcpy(ff.data(), dd, n * sizeof(float));
            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));

            // Build lower envelope (L2 closed-form intersect, float precision)
            int k = 0;
            v[0] = 0;
            ranges[0] = -std::numeric_limits<float>::infinity();
            ranges[1] = std::numeric_limits<float>::infinity();

            // Float-precision intersect using difference-of-squares factorization
            // to minimize catastrophic cancellation.
            auto intersect = [&](int a, int b) -> float {
                const float denom = 2.0f * wsq * float(b - a);
                return (ff[b] - ff[a] + wsq * float((b + a) * (b - a))) / denom;
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

            // Output pass
            k = 0;
            if (black_border) {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    const float envelope = wsq * sq(i - v[k]) + ff[v[k]];
                    const float border = wsq * std::fminf(sq(i + 1), sq(nn - i));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                }
            } else {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    dd[i] = wsq * sq(i - v[k]) + ff[v[k]];
                    ll[i] = lbl_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

template <typename INDEX>
inline void _expand_parabolic_feat(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    INDEX* __restrict__ feat,
    const size_t n,
    const size_t num_lines,
    const float anis,
    const bool black_border,
    const int parallel
) {
    if (n == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, n);
    const float wsq = anis * anis;
    const int nn = (int)n;

    auto process_chunk = [&](size_t begin, size_t end) {
        std::vector<int>      v(n);
        std::vector<float>    ff(n), ranges(n + 1);
        std::vector<uint32_t> lbl_save(n);
        std::vector<INDEX>    feat_save(n);

        for (size_t line = begin; line < end; ++line) {
            uint32_t* ll = lbl + line * n;
            float*    dd = dist + line * n;
            INDEX*    ft = feat + line * n;

            bool any_nonzero = false;
            for (size_t j = 0; j < n; ++j) {
                if (dd[j] != 0.0f) { any_nonzero = true; break; }
            }
            if (!any_nonzero) continue;

            std::memcpy(ff.data(), dd, n * sizeof(float));
            std::memcpy(lbl_save.data(), ll, n * sizeof(uint32_t));
            std::memcpy(feat_save.data(), ft, n * sizeof(INDEX));

            int k = 0;
            v[0] = 0;
            ranges[0] = -std::numeric_limits<float>::infinity();
            ranges[1] = std::numeric_limits<float>::infinity();

            auto intersect = [&](int a, int b) -> float {
                const float denom = 2.0f * wsq * float(b - a);
                return (ff[b] - ff[a] + wsq * float((b + a) * (b - a))) / denom;
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

            k = 0;
            if (black_border) {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    const float envelope = wsq * sq(i - v[k]) + ff[v[k]];
                    const float border = wsq * std::fminf(sq(i + 1), sq(nn - i));
                    dd[i] = std::fminf(border, envelope);
                    ll[i] = lbl_save[v[k]];
                    ft[i] = feat_save[v[k]];
                }
            } else {
                for (int i = 0; i < nn; i++) {
                    while (ranges[k + 1] < i) k++;
                    dd[i] = wsq * sq(i - v[k]) + ff[v[k]];
                    ll[i] = lbl_save[v[k]];
                    ft[i] = feat_save[v[k]];
                }
            }
        }
    };
    dispatch_parallel(threads, num_lines, threads * ND_CHUNKS_PER_THREAD, process_chunk);
}

//-----------------------------------------------------------------------------
// Blocked transpose with streaming stores for non-contiguous axis processing.
// Uses non-temporal stores for the strided writes to avoid read-for-ownership
// cache line fetches, which cause 16x bandwidth amplification on x86.
// 3 barriers per axis (transpose → process → transpose back).
//-----------------------------------------------------------------------------

constexpr size_t TRANSPOSE_BLOCK = 64;

// Transpose A planes of (rows × cols) → (cols × rows), one array.
// Read-sequential (inner loop over c) with strided writes using a small
// register-resident tile to amortize write-combining. Block size 64.
template <typename T>
inline void _transpose_planes_nt(
    const T* __restrict__ src, T* __restrict__ dst,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const T* sp = src + a * rows * cols;
                T* dp = dst + a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c)
                        dp[c * rows + r] = sp[r * cols + c];
            }
        }
    );
}

// Fused transpose of two arrays
template <typename T1, typename T2>
inline void _transpose_planes_2_nt(
    const T1* __restrict__ s1, T1* __restrict__ d1,
    const T2* __restrict__ s2, T2* __restrict__ d2,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const size_t plane = a * rows * cols;
                const size_t tplane = a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c) {
                        d1[tplane + c * rows + r] = s1[plane + r * cols + c];
                        d2[tplane + c * rows + r] = s2[plane + r * cols + c];
                    }
            }
        }
    );
}

// Fused transpose of three arrays
template <typename T1, typename T2, typename T3>
inline void _transpose_planes_3_nt(
    const T1* __restrict__ s1, T1* __restrict__ d1,
    const T2* __restrict__ s2, T2* __restrict__ d2,
    const T3* __restrict__ s3, T3* __restrict__ d3,
    const size_t A, const size_t rows, const size_t cols,
    const size_t threads
) {
    const size_t ncb = (cols + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t nrb = (rows + TRANSPOSE_BLOCK - 1) / TRANSPOSE_BLOCK;
    const size_t bpp = nrb * ncb;
    const size_t total = A * bpp;

    dispatch_parallel(threads, total, threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const size_t a   = idx / bpp;
                const size_t blk = idx % bpp;
                const size_t rb  = blk / ncb;
                const size_t cb  = blk % ncb;
                const size_t r0 = rb * TRANSPOSE_BLOCK, r1 = std::min(r0 + TRANSPOSE_BLOCK, rows);
                const size_t c0 = cb * TRANSPOSE_BLOCK, c1 = std::min(c0 + TRANSPOSE_BLOCK, cols);
                const size_t plane = a * rows * cols;
                const size_t tplane = a * cols * rows;
                for (size_t r = r0; r < r1; ++r)
                    for (size_t c = c0; c < c1; ++c) {
                        const size_t si = plane + r * cols + c;
                        const size_t di = tplane + c * rows + r;
                        d1[di] = s1[si];
                        d2[di] = s2[si];
                        d3[di] = s3[si];
                    }
            }
        }
    );
}

//-----------------------------------------------------------------------------
// Strided variants: streaming transpose → contiguous process → streaming transpose back.
//-----------------------------------------------------------------------------

inline void _expand_pass0_strided(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    uint32_t* __restrict__ ws_lbl,
    float* __restrict__ ws_dist,
    const size_t B, const size_t C, const size_t A,
    const float anis, const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_nt(lbl, ws_lbl, A, B, C, threads);
    _expand_pass0(ws_lbl, ws_dist, B, num_lines, anis, black_border, parallel);
    _transpose_planes_2_nt(ws_lbl, lbl, ws_dist, dist, A, C, B, threads);
}

inline void _expand_parabolic_strided(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    uint32_t* __restrict__ ws_lbl,
    float* __restrict__ ws_dist,
    const size_t B, const size_t C, const size_t A,
    const float anis, const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_2_nt(lbl, ws_lbl, dist, ws_dist, A, B, C, threads);
    _expand_parabolic(ws_lbl, ws_dist, B, num_lines, anis, black_border, parallel);
    _transpose_planes_2_nt(ws_lbl, lbl, ws_dist, dist, A, C, B, threads);
}

template <typename INDEX>
inline void _expand_pass0_feat_strided(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    INDEX* __restrict__ feat,
    uint32_t* __restrict__ ws_lbl,
    float* __restrict__ ws_dist,
    INDEX* __restrict__ ws_feat,
    const size_t B, const size_t C, const size_t A,
    const float anis, const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_2_nt(lbl, ws_lbl, feat, ws_feat, A, B, C, threads);
    _expand_pass0_feat(ws_lbl, ws_dist, ws_feat, B, num_lines, anis, black_border, parallel);
    _transpose_planes_3_nt(ws_lbl, lbl, ws_dist, dist, ws_feat, feat, A, C, B, threads);
}

template <typename INDEX>
inline void _expand_parabolic_feat_strided(
    uint32_t* __restrict__ lbl,
    float* __restrict__ dist,
    INDEX* __restrict__ feat,
    uint32_t* __restrict__ ws_lbl,
    float* __restrict__ ws_dist,
    INDEX* __restrict__ ws_feat,
    const size_t B, const size_t C, const size_t A,
    const float anis, const bool black_border, const int parallel
) {
    const size_t num_lines = A * C;
    if (B == 0 || num_lines == 0) return;
    const size_t threads = compute_threads(parallel, num_lines, B);

    _transpose_planes_3_nt(lbl, ws_lbl, dist, ws_dist, feat, ws_feat, A, B, C, threads);
    _expand_parabolic_feat(ws_lbl, ws_dist, ws_feat, B, num_lines, anis, black_border, parallel);
    _transpose_planes_3_nt(ws_lbl, lbl, ws_dist, dist, ws_feat, feat, A, C, B, threads);
}

//=============================================================================
// Expand labels orchestrators (transpose pipeline with cached buffers)
//=============================================================================

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
            const size_t seed_idx = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= seed_idx) ? (i - seed_idx) : (seed_idx - i);
                if (border_dist <= seed_dist) { labels_out[i] = 0; continue; }
            }
            labels_out[i] = (uint32_t)data[seed_idx];
        }
        return;
    }

    // ND path: streaming transpose pipeline with cached buffers
    size_t total = 1;
    size_t strides[32], paxes[32];
    for (size_t d = dims; d-- > 0;) { strides[d] = total; total *= shape[d]; }
    if (total == 0) return;

    _expand_sort_axes(paxes, shape, strides, dims);

    // Slots: 0=lbl, 1=dist, 2=ws_lbl, 3=ws_dist
    auto& cache = expand_cache();
    uint32_t* lbl     = (uint32_t*)cache.get(0, total * sizeof(uint32_t));
    float*    dist    = (float*)cache.get(1, total * sizeof(float));
    uint32_t* ws_lbl  = (uint32_t*)cache.get(2, total * sizeof(uint32_t));
    float*    ws_dist = (float*)cache.get(3, total * sizeof(float));

    const size_t par_threads = compute_threads(parallel, total, 1);
    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                lbl[i] = (uint32_t)data[i];
        });

    for (size_t pass = 0; pass < dims; ++pass) {
        const size_t axis     = paxes[pass];
        const size_t axis_len = shape[axis];
        const float  anis     = anisotropy[axis];

        if (strides[axis] == 1) {
            const size_t num_lines = total / axis_len;
            if (pass == 0)
                _expand_pass0(lbl, dist, axis_len, num_lines, anis, black_border, parallel);
            else
                _expand_parabolic(lbl, dist, axis_len, num_lines, anis, black_border, parallel);
        } else {
            const size_t C = strides[axis];
            const size_t B = axis_len;
            const size_t A = total / (B * C);
            if (pass == 0)
                _expand_pass0_strided(lbl, dist, ws_lbl, ws_dist, B, C, A, anis, black_border, parallel);
            else
                _expand_parabolic_strided(lbl, dist, ws_lbl, ws_dist, B, C, A, anis, black_border, parallel);
        }
    }

    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            std::memcpy(labels_out + begin, lbl + begin, (end - begin) * sizeof(uint32_t));
        });
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
            const size_t seed_idx = seeds[std::min(k, seeds.size() - 1)];
            if (black_border) {
                const size_t border_dist = std::min(i + 1, n - i);
                const size_t seed_dist   = (i >= seed_idx) ? (i - seed_idx) : (seed_idx - i);
                if (border_dist <= seed_dist) {
                    labels_out[i]   = 0;
                    features_out[i] = INDEX(seed_idx);
                    continue;
                }
            }
            labels_out[i]   = (uint32_t)data[seed_idx];
            features_out[i] = INDEX(seed_idx);
        }
        return;
    }

    // ND path: streaming transpose pipeline with feature tracking
    size_t total = 1;
    size_t strides[32], paxes[32];
    for (size_t d = dims; d-- > 0;) { strides[d] = total; total *= shape[d]; }
    if (total == 0) return;

    _expand_sort_axes(paxes, shape, strides, dims);

    // Slots: 0=lbl, 1=dist, 2=ws_lbl, 3=ws_dist
    auto& cache = expand_cache();
    uint32_t* lbl     = (uint32_t*)cache.get(0, total * sizeof(uint32_t));
    float*    dist    = (float*)cache.get(1, total * sizeof(float));
    uint32_t* ws_lbl  = (uint32_t*)cache.get(2, total * sizeof(uint32_t));
    float*    ws_dist = (float*)cache.get(3, total * sizeof(float));

    // Feat/ws_feat use separate malloc (template type can't easily cache)
    INDEX* feat    = (INDEX*)std::malloc(total * sizeof(INDEX));
    INDEX* ws_feat = (INDEX*)std::malloc(total * sizeof(INDEX));

    const size_t par_threads = compute_threads(parallel, total, 1);
    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                lbl[i]  = (uint32_t)data[i];
                feat[i] = (INDEX)i;
            }
        });

    for (size_t pass = 0; pass < dims; ++pass) {
        const size_t axis     = paxes[pass];
        const size_t axis_len = shape[axis];
        const float  anis     = anisotropy[axis];

        if (strides[axis] == 1) {
            const size_t num_lines = total / axis_len;
            if (pass == 0)
                _expand_pass0_feat(lbl, dist, feat, axis_len, num_lines, anis, black_border, parallel);
            else
                _expand_parabolic_feat(lbl, dist, feat, axis_len, num_lines, anis, black_border, parallel);
        } else {
            const size_t C = strides[axis];
            const size_t B = axis_len;
            const size_t A = total / (B * C);
            if (pass == 0)
                _expand_pass0_feat_strided(lbl, dist, feat, ws_lbl, ws_dist, ws_feat, B, C, A, anis, black_border, parallel);
            else
                _expand_parabolic_feat_strided(lbl, dist, feat, ws_lbl, ws_dist, ws_feat, B, C, A, anis, black_border, parallel);
        }
    }

    dispatch_parallel(par_threads, total, par_threads * ND_CHUNKS_PER_THREAD,
        [&](size_t begin, size_t end) {
            std::memcpy(labels_out + begin, lbl + begin, (end - begin) * sizeof(uint32_t));
            std::memcpy(features_out + begin, feat + begin, (end - begin) * sizeof(INDEX));
        });
    std::free(feat);
    std::free(ws_feat);
}

} // namespace nd

#endif // EDT_HPP
