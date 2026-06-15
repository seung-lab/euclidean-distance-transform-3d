"""Thread-safety regression for the parallel EDT entry points.

The fork-join thread pool is a process-global singleton, keyed by thread
count (``shared_pool_for`` in ``src/edt.hpp``) and backed by a single C++
``ForkJoinPool``. Each ``ForkJoinPool`` has one shared ``work_fn_`` slot and
one centralized barrier sized to exactly ``num_participants_``. Two Python
threads entering a parallel EDT call (``edtsq`` / ``edt`` / ``expand_labels``
with ``parallel > 1``) at the same time drive that one pool concurrently:
they race on the ``work_fn_`` assignment and both bump the barrier counter,
which breaks the ``arrived == num_participants_`` release and hard-crashes
the process (``SIGSEGV``). A single call from any thread is always fine --
only cross-thread *overlap* corrupts the shared pool.

A segfault kills the whole interpreter (it would take pytest down with it),
so the concurrent stress runs in a SUBPROCESS and the test asserts the child
exits cleanly. A regression therefore shows up as a normal test failure
(non-zero return code), not a crashed test session.
"""
import subprocess
import sys
import textwrap

import numpy as np

import edt


# Concurrent stress: N threads each repeatedly run edtsq + expand_labels with
# parallel>1 (so every call drives the shared pool). Same thread count from
# every worker => they all contend on the *same* keyed pool. Exits 1 on any
# worker error, else 0. A pre-fix interpreter SIGSEGVs here => child return
# code is negative (killed by signal).
_STRESS = textwrap.dedent(
    """
    import sys, threading
    import numpy as np
    import edt

    def make(seed, n=1024, k=800):
        rng = np.random.default_rng(seed)
        a = np.zeros((n, n), np.uint32)
        ys = rng.integers(0, n, k); xs = rng.integers(0, n, k)
        for i, (y, x) in enumerate(zip(ys, xs), 1):
            a[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = i
        return a

    THREADS, REPS, PAR = 4, 8, 8
    segs = [make(t) for t in range(THREADS)]
    errs = []

    def work(seg):
        try:
            for _ in range(REPS):
                d = np.asarray(edt.edtsq(seg, parallel=PAR))
                assert d.shape == seg.shape
                # expand_labels shares the same singleton pool
                lbl = np.asarray(edt.expand_labels(seg, parallel=PAR))
                assert lbl.shape == seg.shape
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(s,)) for s in segs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    sys.exit(1 if errs else 0)
    """
)


def test_concurrent_edt_does_not_crash():
    """``edtsq``/``expand_labels`` must survive concurrent parallel calls."""
    proc = subprocess.run(
        [sys.executable, "-c", _STRESS],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "concurrent parallel EDT crashed or errored "
        f"(returncode={proc.returncode}; negative => killed by signal, "
        "e.g. -11 = SIGSEGV from the unguarded shared ForkJoinPool).\n"
        f"stderr tail:\n{proc.stderr.decode(errors='replace')[-3000:]}"
    )


def test_concurrent_edtsq_matches_serial():
    """Serialized concurrent calls must still produce correct results:
    each thread's edtsq matches a plain serial call on the same input."""
    import threading

    rng = np.random.default_rng(0)
    arr = np.zeros((256, 256), dtype=np.uint32)
    for i in range(1, 200):
        y, x = rng.integers(0, 256, 2)
        arr[max(0, y - 5):y + 5, max(0, x - 5):x + 5] = i

    serial = np.asarray(edt.edtsq(arr, parallel=8))
    results = [None] * 8

    def work(i):
        results[i] = np.asarray(edt.edtsq(arr, parallel=8))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    for r in results:
        assert r is not None
        assert np.array_equal(r, serial)
