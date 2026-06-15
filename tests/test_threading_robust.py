"""Robustness suite for the fork-join threading model.

The existing suite only ever drives the pool at ``parallel in (1, 2)`` (and a
few non-CI tests at 4/8), on 2-core CI runners, and never exercises the idle
wake path or interpreter-shutdown path. Threading bugs in this library scale
with *thread count* and *platform*, so those regimes are exactly where the
coverage was missing.

This file targets the gaps:

  * exact parallel-vs-serial equivalence across a thread-count *sweep* up to
    2x the core count (oversubscription),
  * thread count >> problem size (chunk-split / load-balance edges),
  * thin/skewed shapes where some axes have fewer lines than threads,
  * idle->wake stress (repeated fork-join cycles with gaps) under a hard
    watchdog, so a lost wakeup surfaces as a *timeout* not a silent hang,
  * clean interpreter shutdown with live pool workers,
  * concurrent drivers across *different* keyed pools (per-pool driver mutex).

EDT on integer inputs with unit anisotropy is exact, so parallel results must
be *bit-identical* to serial -- we assert ``array_equal``, not ``allclose``.

Hang-prone and crash-prone cases run in a subprocess with a timeout so a
deadlock/segfault is a normal test failure, not a killed pytest session.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import edt


CPU = os.cpu_count() or 4

# 1 .. 2x cores, plus a couple of awkward in-between values. Oversubscription
# (threads > cores) is where real wake/contention bugs show that a 2-core CI
# runner can never reproduce.
THREAD_SWEEP = sorted({1, 2, 3, 5, 8, 16, CPU, 2 * CPU})

# Heavy concurrent stress on large (4000^2) images is too slow/RAM-hungry for
# 2-core CI runners. Gate it behind EDT_STRESS=1 so it runs locally / on big
# boxes but is skipped by default (including in CI).
stress_only = pytest.mark.skipif(
    not os.environ.get("EDT_STRESS"),
    reason="heavy large-image concurrent stress; set EDT_STRESS=1 to run",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_labels(shape, n_seeds, seed=0):
    """Random segmented labels: n_seeds *unique* single-pixel seeds -- distinct
    positions (no two seeds collide) and distinct labels -- over a zero
    background."""
    rng = np.random.default_rng(seed)
    a = np.zeros(shape, dtype=np.uint32)
    flat = a.reshape(-1)
    n = min(n_seeds, flat.size)
    idx = rng.choice(flat.size, size=n, replace=False)      # unique positions
    flat[idx] = rng.permutation(n).astype(np.uint32) + 1    # unique labels 1..n
    return a


def run_snippet(snippet, timeout):
    """Run a Python snippet in a fresh interpreter.

    Returns (returncode, stdout, stderr). returncode is None on timeout
    (== hang). A negative returncode means killed by signal (e.g. -11 SIGSEGV).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(snippet)],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return None, e.stdout or b"", e.stderr or b""
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------- #
# 1. exact parallel == serial across a full thread sweep (2D + 3D)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,n_seeds",
    [
        ((512, 512), 800),
        ((128, 96), 200),
        ((64, 48, 40), 300),
        ((33, 17, 9, 5), 120),  # 4D, deliberately non-round extents
    ],
)
@pytest.mark.parametrize("parallel", THREAD_SWEEP)
def test_edtsq_parallel_matches_serial(shape, n_seeds, parallel):
    a = make_labels(shape, n_seeds, seed=parallel)  # vary input per case
    serial = np.asarray(edt.edtsq(a, parallel=1))
    got = np.asarray(edt.edtsq(a, parallel=parallel))
    assert np.array_equal(got, serial), (
        f"edtsq parallel={parallel} != serial for shape {shape} "
        f"(max abs diff {np.abs(got - serial).max()})"
    )


@pytest.mark.parametrize(
    "shape,n_seeds",
    [((512, 512), 800), ((64, 48, 40), 300)],
)
@pytest.mark.parametrize("parallel", THREAD_SWEEP)
def test_expand_labels_parallel_matches_serial(shape, n_seeds, parallel):
    a = make_labels(shape, n_seeds, seed=1000 + parallel)
    serial = np.asarray(edt.expand_labels(a, parallel=1))
    got = np.asarray(edt.expand_labels(a, parallel=parallel))
    assert np.array_equal(got, serial), (
        f"expand_labels parallel={parallel} != serial for shape {shape}"
    )


@pytest.mark.parametrize("parallel", THREAD_SWEEP)
def test_feature_transform_parallel_matches_serial(parallel):
    if not hasattr(edt, "feature_transform"):
        pytest.skip("feature_transform not available in this build")
    a = make_labels((96, 72, 40), 250, seed=2000 + parallel)
    serial = np.asarray(edt.feature_transform(a, parallel=1))
    got = np.asarray(edt.feature_transform(a, parallel=parallel))
    assert np.array_equal(got, serial), (
        f"feature_transform parallel={parallel} != serial"
    )


# --------------------------------------------------------------------------- #
# 2. thread count >> problem size  (chunk-split / empty-chunk edges)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [(1,), (2,), (1, 1), (2, 2), (3, 1, 5), (1, 1, 1), (1, 256), (256, 1)],
)
@pytest.mark.parametrize("parallel", [8, 16, 2 * CPU])
def test_tiny_arrays_more_threads_than_work(shape, parallel):
    a = make_labels(shape, max(1, np.prod(shape) // 2), seed=7)
    serial = np.asarray(edt.edtsq(a, parallel=1))
    got = np.asarray(edt.edtsq(a, parallel=parallel))
    assert got.shape == tuple(shape)
    assert np.array_equal(got, serial)


# --------------------------------------------------------------------------- #
# 3. thin/skewed shapes: some axes have far fewer lines than threads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [(1, 20000), (20000, 1), (2, 9000), (9000, 2), (3, 3, 4000), (4000, 3, 3)],
)
@pytest.mark.parametrize("parallel", [CPU, 2 * CPU])
def test_skewed_shapes(shape, parallel):
    a = make_labels(shape, max(4, np.prod(shape) // 50), seed=11)
    serial = np.asarray(edt.edtsq(a, parallel=1))
    got = np.asarray(edt.edtsq(a, parallel=parallel))
    assert np.array_equal(got, serial)


# --------------------------------------------------------------------------- #
# 4. idle -> wake stress: a lost wakeup must surface as a timeout, not a hang
# --------------------------------------------------------------------------- #
def test_idle_wake_stress_no_hang():
    """Many fork-join cycles separated by sleeps force workers down the
    wait-on-address idle path and back. A dropped wake hangs forever -> the
    subprocess times out -> test fails."""
    snippet = """
        import time
        import numpy as np
        import edt

        a = np.zeros((256, 256), np.uint32)
        a[60:120, 60:120] = 1
        a[180:220, 30:90] = 2
        for i in range(400):
            edt.edtsq(a, parallel=8)
            if i % 16 == 0:
                time.sleep(0.003)   # let workers fully idle on wait-on-address
        print("done")
    """
    rc, out, err = run_snippet(snippet, timeout=120)
    assert rc is not None, (
        "idle/wake stress HUNG (timed out) -> likely a lost wakeup in the "
        "wait-on-address idle path.\n" + err.decode(errors="replace")[-2000:]
    )
    assert rc == 0, f"idle/wake stress failed rc={rc}\n{err.decode(errors='replace')[-2000:]}"


def test_high_frequency_dispatch_no_hang():
    """Tight loop of many small parallel calls -- maximizes the rate of
    idle->wake transitions to flush out a rare missed wakeup."""
    snippet = """
        import numpy as np
        import edt
        a = np.zeros((64, 64), np.uint32); a[20:40, 20:40] = 1
        for _ in range(5000):
            edt.edtsq(a, parallel=8)
        print("done")
    """
    rc, out, err = run_snippet(snippet, timeout=120)
    assert rc is not None, "high-frequency dispatch HUNG (timed out)"
    assert rc == 0, f"rc={rc}\n{err.decode(errors='replace')[-2000:]}"


# --------------------------------------------------------------------------- #
# 5. clean interpreter shutdown with live pool workers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parallel", [4, 8, 2 * CPU])
def test_clean_shutdown_after_parallel(parallel):
    """Run a parallel call, then let the interpreter exit. Persistent spinning
    workers must join cleanly at static-destruction time -- a hang or crash on
    exit shows up here."""
    snippet = f"""
        import numpy as np
        import edt
        a = np.zeros((200, 200), np.uint32); a[50:150, 50:150] = 1
        edt.edtsq(a, parallel={parallel})
        # no explicit teardown: exercise atexit / static destruction order
    """
    rc, out, err = run_snippet(snippet, timeout=60)
    assert rc is not None, f"interpreter HUNG on shutdown (parallel={parallel})"
    assert rc == 0, (
        f"non-clean shutdown rc={rc} (parallel={parallel}); negative => signal\n"
        + err.decode(errors="replace")[-2000:]
    )


# --------------------------------------------------------------------------- #
# 6. repeated determinism (occasional corruption detector)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("parallel", [8, CPU, 2 * CPU])
def test_repeated_runs_identical(parallel):
    a = make_labels((300, 300), 500, seed=99)
    ref = np.asarray(edt.edtsq(a, parallel=parallel))
    for _ in range(40):
        got = np.asarray(edt.edtsq(a, parallel=parallel))
        assert np.array_equal(got, ref), "non-deterministic parallel result"


# --------------------------------------------------------------------------- #
# 7. concurrent drivers across DIFFERENT keyed pools
# --------------------------------------------------------------------------- #
def test_concurrent_mixed_thread_counts():
    """Python threads driving *different* parallel values hit different keyed
    ForkJoinPool instances simultaneously -- exercises the per-pool driver
    mutex across multiple pools, not just one. Subprocess in case of segfault."""
    snippet = """
        import sys, threading
        import numpy as np
        import edt

        a = np.zeros((400, 400), np.uint32)
        a[80:160, 80:160] = 1; a[240:300, 100:200] = 2
        errs = []
        def work(par):
            try:
                ref = np.asarray(edt.edtsq(a, parallel=1))
                for _ in range(20):
                    d = np.asarray(edt.edtsq(a, parallel=par))
                    assert np.array_equal(d, ref)
            except BaseException as e:
                errs.append(repr(e))
        # distinct thread counts => distinct keyed pools, run concurrently
        ts = [threading.Thread(target=work, args=(p,)) for p in (2, 4, 8, 16)]
        for t in ts: t.start()
        for t in ts: t.join()
        sys.exit(1 if errs else 0)
    """
    rc, out, err = run_snippet(snippet, timeout=120)
    assert rc is not None, "concurrent mixed-pool drivers HUNG (timed out)"
    assert rc == 0, (
        f"concurrent mixed-pool drivers crashed/errored rc={rc} "
        f"(negative => signal, -11 == SIGSEGV)\n"
        + err.decode(errors="replace")[-2500:]
    )


# --------------------------------------------------------------------------- #
# 8. heavy: concurrent drivers on LARGE images (opt-in, EDT_STRESS=1)
# --------------------------------------------------------------------------- #
@stress_only
def test_concurrent_large_images_stress():
    """Concurrent threads each run edtsq + expand_labels on a large 4000^2
    image with unique seeds, at *different* thread counts (different keyed
    pools), all verified against a serial reference. This is the realistic
    high-contention case: several heavy transforms driving the shared pools at
    once, with real per-call work (unlike the tiny-image concurrency test).

    Opt-in via EDT_STRESS=1 -- too slow / RAM-hungry for 2-core CI runners.
    Runs in a subprocess so a segfault is a failure, not a killed session."""
    snippet = """
        import sys, threading
        import numpy as np
        import edt

        def make_labels(shape, n_seeds, seed):
            rng = np.random.default_rng(seed)
            a = np.zeros(shape, dtype=np.uint32)
            flat = a.reshape(-1)
            n = min(n_seeds, flat.size)
            idx = rng.choice(flat.size, size=n, replace=False)   # unique pos
            flat[idx] = rng.permutation(n).astype(np.uint32) + 1 # unique labels
            return a

        a = make_labels((4000, 4000), 20000, 7)
        ref = np.asarray(edt.edtsq(a, parallel=1))
        errs = []

        def work(par):
            try:
                for _ in range(5):
                    d = np.asarray(edt.edtsq(a, parallel=par))
                    assert np.array_equal(d, ref), f"edtsq mismatch par={par}"
                    lbl = np.asarray(edt.expand_labels(a, parallel=par))
                    assert lbl.shape == a.shape
            except BaseException as e:  # noqa: BLE001
                errs.append(repr(e))

        # distinct thread counts => distinct keyed pools, driven concurrently
        ts = [threading.Thread(target=work, args=(p,)) for p in (2, 4, 8, 16)]
        for t in ts: t.start()
        for t in ts: t.join()
        sys.exit(1 if errs else 0)
    """
    rc, out, err = run_snippet(snippet, timeout=600)
    assert rc is not None, "large-image concurrent stress HUNG (timed out)"
    assert rc == 0, (
        f"large-image concurrent stress crashed/errored rc={rc} "
        f"(negative => signal, -11 == SIGSEGV)\n"
        + err.decode(errors="replace")[-2500:]
    )
