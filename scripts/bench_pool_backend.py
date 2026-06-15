"""A/B benchmark of the threading backend, in isolation.

Routes the *same* EDT algorithm and *same* chunk decomposition through
different worker mechanisms via the EDT_POOL_BACKEND env var (see
``dispatch_parallel`` in src/edt.hpp):

    forkjoin   persistent spinning fork-join pool (production)
    stdthread  std::thread spawned + joined on every dispatch (no pool)
    serial     all chunks on the calling thread (baseline)

Because only the worker mechanism changes, the forkjoin-vs-stdthread ratio is
the threading model's contribution, with the algorithm held constant -- unlike
``edt`` vs ``edt_legacy``, which also changes the algorithm.

Backend is chosen once per process (static init from the env var), so each
backend runs in its own subprocess.

The alternate backends are gated out of production builds, so first rebuild the
extension with them enabled, then run:

    EDT_BENCH_BACKENDS=1 python setup.py build_ext --inplace
    python scripts/bench_pool_backend.py

If the extension was NOT built with EDT_BENCH_BACKENDS=1, EDT_POOL_BACKEND is
ignored (every backend silently runs forkjoin) and the script warns.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

# Make the in-place built extension importable regardless of CWD or how the
# worker subprocess is launched: build_ext --inplace puts edt*.so at the repo
# root, but `python scripts/...` roots sys.path at scripts/. (Harmless when edt
# is an editable/site install.)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

BACKENDS = ["forkjoin", "taskqueue", "stdthread", "serial"]


def make_labels(shape, n_seeds, seed=0):
    # unique seeds: distinct positions (no collisions) and distinct labels
    rng = np.random.default_rng(seed)
    a = np.zeros(shape, dtype=np.uint32)
    flat = a.reshape(-1)
    n = min(n_seeds, flat.size)
    idx = rng.choice(flat.size, size=n, replace=False)
    flat[idx] = rng.permutation(n).astype(np.uint32) + 1
    return a


# (label, shape, n_seeds). Sized so parallelism is on the critical path (vs the
# overhead-bound regime where serial ~ 16T and the threading model can't show).
CASES = [
    ("2D 5000^2", (5000, 5000), 20000),
    ("3D 384^3", (384, 384, 384), 40000),
]

# Burst regime: many small transforms back-to-back. This is where a persistent
# pool should dominate -- stdthread pays thread create+join on every call, the
# pool pays it once. Realistic for batch-processing many small images/volumes.
BURST_SHAPE = (192, 192)
BURST_CALLS = 2000


def thread_counts():
    cpu = os.cpu_count() or 4
    base = [1, 2, 4, 8, 16, 32, 64]
    return [t for t in base if t <= 2 * cpu] or [1, 2]


def bench_worker():
    """Run inside a subprocess with EDT_POOL_BACKEND already set."""
    import edt

    backend = os.environ.get("EDT_POOL_BACKEND", "forkjoin")
    rows = []
    for label, shape, n_seeds in CASES:
        a = make_labels(shape, n_seeds, seed=1)
        for t in thread_counts():
            # warmup (spin up pool, page in working buffers)
            for _ in range(2):
                edt.edtsq(a, parallel=t)
            samples = []
            for _ in range(7):
                t0 = time.perf_counter()
                edt.edtsq(a, parallel=t)
                samples.append((time.perf_counter() - t0) * 1e3)  # ms
            rows.append(
                {"case": label, "threads": t, "ms": round(statistics.median(samples), 3)}
            )

    # Burst throughput: BURST_CALLS small transforms at a fixed thread count.
    cpu = os.cpu_count() or 4
    burst = {}
    a = make_labels(BURST_SHAPE, 200, seed=2)
    for t in sorted({8, cpu}):
        for _ in range(3):
            edt.edtsq(a, parallel=t)
        t0 = time.perf_counter()
        for _ in range(BURST_CALLS):
            edt.edtsq(a, parallel=t)
        burst[t] = round(time.perf_counter() - t0, 3)
    print(json.dumps({"backend": backend, "rows": rows, "burst": burst}))


def drive():
    import platform

    results = {}
    burst = {}
    for backend in BACKENDS:
        env = dict(os.environ, EDT_POOL_BACKEND=backend)
        proc = subprocess.run(
            [sys.executable, __file__, "--worker"],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[{backend}] FAILED rc={proc.returncode}\n{proc.stderr[-1500:]}")
            continue
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        for r in data["rows"]:
            results[(r["case"], r["threads"], backend)] = r["ms"]
        for t, secs in data.get("burst", {}).items():
            burst[(int(t), backend)] = secs

    cpu = os.cpu_count() or 4

    # Guard: if the extension lacks the bench backends, EDT_POOL_BACKEND is a
    # no-op and "serial" actually ran forkjoin -- so serial ~ forkjoin even on a
    # large array at max threads (where true serial would be many-fold slower).
    label0, tmax = CASES[0][0], thread_counts()[-1]
    fj0 = results.get((label0, tmax, "forkjoin"))
    se0 = results.get((label0, tmax, "serial"))
    if fj0 and se0 and se0 < 1.5 * fj0:
        print("WARNING: 'serial' is not slower than 'forkjoin' at "
              f"{tmax} threads -- the extension was likely NOT built with "
              "EDT_BENCH_BACKENDS=1, so EDT_POOL_BACKEND is a no-op and every\n"
              "row below is really forkjoin. Rebuild:\n"
              "  EDT_BENCH_BACKENDS=1 python setup.py build_ext --inplace\n")

    print(f"\n# Threading backend A/B  ({platform.system()} {platform.machine()}, "
          f"{cpu} cores)\n")
    for label, shape, _ in CASES:
        print(f"## {label}")
        print("| threads | forkjoin/new (ms) | taskqueue/old (ms) | "
              "stdthread (ms) | serial (ms) | new vs old |")
        print("|--------:|------------------:|-------------------:|"
              "---------------:|------------:|:----------:|")
        for t in thread_counts():
            fj = results.get((label, t, "forkjoin"))
            tq = results.get((label, t, "taskqueue"))
            st = results.get((label, t, "stdthread"))
            se = results.get((label, t, "serial"))
            ratio = (f"{tq / fj:.2f}x" if (fj and tq and fj > 0) else "-")
            print(f"| {t} | {fj if fj is not None else '-'} | "
                  f"{tq if tq is not None else '-'} | "
                  f"{st if st is not None else '-'} | "
                  f"{se if se is not None else '-'} | {ratio} |")
        print()

    burst_threads = sorted({t for (t, _b) in burst})
    if burst_threads:
        print(f"## Burst: {BURST_CALLS} x edtsq({BURST_SHAPE[0]}^2) total seconds")
        print("| threads | forkjoin/new (s) | taskqueue/old (s) | "
              "stdthread (s) | new vs old |")
        print("|--------:|-----------------:|------------------:|"
              "--------------:|:----------:|")
        for t in burst_threads:
            fj = burst.get((t, "forkjoin"))
            tq = burst.get((t, "taskqueue"))
            st = burst.get((t, "stdthread"))
            ratio = (f"{tq / fj:.1f}x" if (fj and tq and fj > 0) else "-")
            print(f"| {t} | {fj if fj is not None else '-'} | "
                  f"{tq if tq is not None else '-'} | "
                  f"{st if st is not None else '-'} | {ratio} |")
        print()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        bench_worker()
    else:
        drive()
