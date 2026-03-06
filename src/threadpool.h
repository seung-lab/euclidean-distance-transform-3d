/*
Fork-Join Thread Pool for parallel dispatch.

Replaces the original queue+future ThreadPool with a lightweight spin-based
fork-join pool. Workers spin-wait on a sense-reversing barrier, waking
instantly when work arrives. Work is distributed via atomic counter
(no mutex, no queue, no futures on the hot path).

Original ThreadPool: Copyright (c) 2012 Jakob Progsch, Václav Zeman (zlib license).
Rewritten by William Silversmith and Kevin Cutler, 2025-2026.
*/

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

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

    // Sense-reversing centralized barrier.
    // All num_participants_ threads (workers + main) must call this.
    // Last thread to arrive flips the sense and releases everyone.
    void barrier_wait_() {
        const int local_sense = 1 - bar_sense_.load(std::memory_order_relaxed);
        const size_t arrived = bar_count_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (arrived == num_participants_) {
            // Last to arrive: reset count and flip sense
            bar_count_.store(0, std::memory_order_relaxed);
            bar_sense_.store(local_sense, std::memory_order_release);
        } else {
            // Spin-wait until sense flips (hybrid: spin then yield)
            int spins = 0;
            while (bar_sense_.load(std::memory_order_acquire) != local_sense) {
                if (++spins < 1024) {
                    FORKJOIN_PAUSE();
                } else {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
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

    std::vector<std::thread> workers_;
};

#endif // THREAD_POOL_H
