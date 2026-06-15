/*
Fork-Join Thread Pool for parallel dispatch.

Sense-reversing centralized barrier with a wait-on-address idle path:
  * **Spin** (FORKJOIN_SPIN_PAUSES pauses) catches tight back-to-back
    fork-join cycles within microseconds.
  * After the spin window expires, workers wait on the barrier sense
    via a kernel wait-on-address primitive:
      - macOS:  ``__ulock_wait(UL_COMPARE_AND_WAIT, ...)``
      - Linux:  ``futex(FUTEX_WAIT_PRIVATE, ...)``
      - Windows:``WaitOnAddress``
    The last participant to arrive at the barrier wakes all sleeping
    workers via the matching wake primitive. Wake latency is ~µs on
    all platforms.

Why wait-on-address rather than the older ``sleep_for(5ms)`` macOS
fallback or ``yield`` Linux loop: the sleep path regressed small-job
multi-T perf 60-100× on macOS (each ``parallel()`` had to pay ~2.5 ms
mean wake delay because nanosleep below the ~10 ms scheduler tick is
clamped); the yield path caused a ``swtch_pri`` storm (~1700%% CPU
across 19 idle workers). wait-on-address gives the µs-class wake of
spin/yield AND the ~0%% idle CPU of sleep — strictly better.

Original ThreadPool: Copyright (c) 2012 Jakob Progsch, Václav Zeman (zlib license).
Rewritten by William Silversmith and Kevin Cutler, 2025-2026.
*/

#ifndef EDT_THREADPOOL_H
#define EDT_THREADPOOL_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// ---- Platform wait-on-address primitives ---------------------------------
#if defined(__APPLE__)
  // Darwin private but ABI-stable since macOS 10.12. Used by libc's
  // os_unfair_lock and friends. xnu source defines the codes; we
  // forward-declare here.
  extern "C" {
    int __ulock_wait(uint32_t operation, void* addr, uint64_t value,
                     uint32_t timeout_us);
    int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value);
  }
  // UL_COMPARE_AND_WAIT = 1; ULF_WAKE_ALL = 0x100.
  #define FORKJOIN_UL_COMPARE_AND_WAIT 1
  #define FORKJOIN_ULF_WAKE_ALL        0x00000100
#elif defined(__linux__)
  #include <linux/futex.h>
  #include <sys/syscall.h>
  #include <unistd.h>
  #include <climits>
#elif defined(_WIN32)
  // Prevent windows.h from defining `min`/`max` macros that break
  // std::numeric_limits<>::max() etc. in downstream code.
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #pragma comment(lib, "Synchronization.lib")
#endif

namespace edt {
namespace threadpool_detail {

// Sleep until the value at *addr is no longer equal to ``expected``.
// Spurious wake-ups are allowed; the caller must re-check the value.
inline void wait_on_value(int* addr, int expected) {
#if defined(__APPLE__)
    __ulock_wait(FORKJOIN_UL_COMPARE_AND_WAIT,
                  reinterpret_cast<void*>(addr),
                  static_cast<uint64_t>(static_cast<uint32_t>(expected)),
                  /*timeout_us=*/0);
#elif defined(__linux__)
    ::syscall(SYS_futex, reinterpret_cast<int*>(addr),
              FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#elif defined(_WIN32)
    int local_expected = expected;
    WaitOnAddress(reinterpret_cast<volatile void*>(addr),
                   &local_expected, sizeof(int), INFINITE);
#else
    // Fallback: tight spin (shouldn't happen on any supported platform).
    while (*reinterpret_cast<volatile int*>(addr) == expected) {}
#endif
}

// Wake every waiter on this address.
inline void wake_all(int* addr) {
#if defined(__APPLE__)
    __ulock_wake(FORKJOIN_UL_COMPARE_AND_WAIT | FORKJOIN_ULF_WAKE_ALL,
                  reinterpret_cast<void*>(addr), 0);
#elif defined(__linux__)
    ::syscall(SYS_futex, reinterpret_cast<int*>(addr),
              FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#elif defined(_WIN32)
    WakeByAddressAll(reinterpret_cast<void*>(addr));
#endif
}

}  // namespace threadpool_detail
}  // namespace edt

// Cross-platform spin-pause hint
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #define FORKJOIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
  #ifdef _MSC_VER
    #include <intrin.h>
    #define FORKJOIN_PAUSE() __yield()
  #else
    #define FORKJOIN_PAUSE() __asm__ __volatile__("yield")
  #endif
#else
  #define FORKJOIN_PAUSE() ((void)0)
#endif

// How many CPU pauses to spin before falling back to the kernel
// wait-on-address. ~10 ns/pause on Apple Silicon, ~30 ns on x86_64 --
// long enough to cover tight back-to-back fork-join cycles within
// microseconds, short enough that genuinely idle pools fall through
// to the kernel wait (which costs ~0%% CPU once parked). Spin sweep
// {4096, 8192, 16384} on AMD + Intel (2026-05-23) showed all within
// ~5 %% of optimal; default stays at the conservative middle value.
#ifndef FORKJOIN_SPIN_PAUSES
#define FORKJOIN_SPIN_PAUSES 8192
#endif

namespace edt {

class ForkJoinPool {
public:
    explicit ForkJoinPool(size_t num_threads)
        : num_participants_(num_threads > 0 ? num_threads : 1),
          num_workers_(num_participants_ - 1),
          alive_(true),
          bar_count_(0),
          bar_sense_(0)
    {
        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&ForkJoinPool::worker_main_, this);
        }
    }

    // Execute fn on all workers + calling thread, block until all complete.
    // fn must be safe to call from multiple threads concurrently.
    template <typename F>
    void parallel(F&& fn) {
        if (num_workers_ == 0) {
            fn();
            return;
        }
        // Serialize driver threads. The shared work_fn_ slot and the
        // centralized barrier (sized to exactly num_participants_) tolerate
        // only ONE thread driving a fork-join cycle at a time. This pool is a
        // process-global singleton keyed by thread count, so two Python
        // threads running a parallel op (edtsq / expand_labels) concurrently
        // would otherwise race on work_fn_ and corrupt the barrier count ->
        // SIGSEGV. Within-call parallelism (the workers running fn) is
        // unaffected; only cross-thread overlap of parallel() is serialized.
        std::lock_guard<std::mutex> drive(driver_mu_);
        work_fn_ = std::forward<F>(fn);
        barrier_wait_();   // release workers (they're waiting at start barrier)
        work_fn_();        // main thread participates
        barrier_wait_();   // wait for all workers to finish
    }

    ~ForkJoinPool() {
        alive_.store(false, std::memory_order_relaxed);
        // Release workers from their start-barrier wait so they can see alive_==false
        barrier_wait_();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Non-copyable, non-movable
    ForkJoinPool(const ForkJoinPool&) = delete;
    ForkJoinPool& operator=(const ForkJoinPool&) = delete;

private:
    void worker_main_() {
        for (;;) {
            barrier_wait_();   // wait for work to be posted
            if (!alive_.load(std::memory_order_relaxed)) return;
            work_fn_();        // execute work
            barrier_wait_();   // signal completion
        }
    }

    // Sense-reversing centralized barrier with kernel wait-on-address
    // idle path. Spin covers tight back-to-back fork-join cycles; once
    // the spin window expires, workers park in the kernel until the
    // last participant arrives and calls wake_all.
    void barrier_wait_() {
        const int local_sense = 1 - bar_sense_.load(std::memory_order_relaxed);
        const size_t arrived = bar_count_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (arrived == num_participants_) {
            // Last to arrive: reset count, flip sense, wake any sleepers.
            // wake_all is cheap when no one is waiting (just a syscall
            // that returns immediately); cheap enough to call
            // unconditionally rather than tracking a "someone is parked"
            // flag.
            bar_count_.store(0, std::memory_order_relaxed);
            bar_sense_.store(local_sense, std::memory_order_release);
            threadpool_detail::wake_all(
                reinterpret_cast<int*>(&bar_sense_));
            return;
        }

        // Phase 1: spin. Covers tight back-to-back fork-join cycles.
        for (int i = 0; i < FORKJOIN_SPIN_PAUSES; ++i) {
            if (bar_sense_.load(std::memory_order_acquire) == local_sense) {
                return;
            }
            FORKJOIN_PAUSE();
        }

        // Phase 2: park in the kernel until the sense flips. Spurious
        // wake-ups are possible (wait returns "earlier than expected"
        // is documented behavior on every platform), so re-check in a
        // loop. The wait primitive does its own compare-then-park
        // atomically, so it's race-free: if the sense already flipped
        // after the spin loop, wait returns immediately (EAGAIN /
        // ERROR_TIMEOUT-equivalent), and we exit on the next load.
        while (bar_sense_.load(std::memory_order_acquire) != local_sense) {
            // Wait while sense is still the OLD value (1 - local_sense).
            threadpool_detail::wait_on_value(
                reinterpret_cast<int*>(&bar_sense_),
                1 - local_sense);
        }
    }

    const size_t num_participants_;  // workers + 1 (main thread)
    const size_t num_workers_;
    std::atomic<bool> alive_;

    // Barrier state
    std::atomic<size_t> bar_count_;
    std::atomic<int> bar_sense_;

    // Current work function (set by parallel(), read by workers)
    std::function<void()> work_fn_;

    // Serializes external driver threads (see parallel()). One mutex per pool,
    // so pools of different thread counts never contend with each other.
    std::mutex driver_mu_;

    std::vector<std::thread> workers_;
};

}  // namespace edt

#endif // EDT_THREADPOOL_H
