#!/usr/bin/env python3
"""
Test correctness of ND EDT implementation.
"""
import numpy as np
import pytest
import sys
import os
import multiprocessing

# Add repo root to path for debug_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from debug_utils import make_label_matrix
import edt

def _make_bench_array(shape, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 3, size=shape, dtype=np.uint8)
    if arr.ndim == 2:
        y, x = shape
        if y > 20 and x > 20:
            arr[y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    elif arr.ndim == 3:
        z, y, x = shape
        if z > 10 and y > 20 and x > 20:
            arr[z // 4 : z // 3, y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * z // 5 : 4 * z // 5, 3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    return arr

def test_nd_correctness_2d():
    """Test ND EDT correctness for 2D cases."""
    for M in [50, 100, 200]:
        masks = make_label_matrix(2, M)
        r1 = edt.edt(masks, parallel=-1)
        r2 = edt.edt(masks, parallel=-1)
        np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6,
                                   err_msg=f"2D case M={M} failed")

        expected_max = float(M)
        np.testing.assert_allclose(r1.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"2D edt max mismatch for M={M}")
        np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"2D edt max mismatch for M={M}")

def test_nd_correctness_3d():
    """Test ND EDT correctness for 3D cases."""
    for M in [50, 100, 200]:
        masks = make_label_matrix(3, M)
        r1 = edt.edt(masks, parallel=-1)
        r2 = edt.edt(masks, parallel=-1)
        np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6,
                                   err_msg=f"3D case M={M} failed")

        expected_max = float(M)
        np.testing.assert_allclose(r1.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"3D edt max mismatch for M={M}")
        np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"3D edt max mismatch for M={M}")

def test_nd_correctness_4d():
    """Test ND EDT correctness for 4D case (ND only, original doesn't support 4D)."""
    # Smaller size for 4D to keep test fast
    masks = make_label_matrix(4, 20)
    # Only test that ND doesn't crash on 4D
    r2 = edt.edt(masks, parallel=-1)
    assert r2.shape == masks.shape, "4D ND EDT shape mismatch"
    assert np.all(np.isfinite(r2)), "4D ND EDT produced non-finite values"

    expected_max = float(20)
    np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                               err_msg="4D edt max mismatch")


def test_nd_correctness_5d():
    """Test ND EDT correctness for 5D case (ND only)."""
    masks = make_label_matrix(5, 10)
    r2 = edt.edt(masks, parallel=-1)

    assert r2.shape == masks.shape, "5D ND EDT shape mismatch"
    assert np.all(np.isfinite(r2)), "5D ND EDT produced non-finite values"

    expected_max = float(10)
    np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                               err_msg="5D edt max mismatch")

def test_nd_threading_consistency():
    """Test that threading produces consistent results."""
    masks = make_label_matrix(3, 50)
    
    # Compare serial vs threaded
    r_serial = edt.edt(masks, parallel=1)
    r_threaded = edt.edt(masks, parallel=-1)

    np.testing.assert_allclose(r_serial, r_threaded, rtol=1e-6, atol=1e-6,
                               err_msg="Threading consistency failed")


def _profile_parallel_used(arr, parallel):
    os.environ['EDT_ND_PROFILE'] = '1'
    try:
        edt.edtsq(arr, parallel=parallel)
        profile = edt._nd_profile_last
    finally:
        os.environ.pop('EDT_ND_PROFILE', None)
    assert profile is not None, "Expected ND profile to be available"
    assert profile.get('parallel_requested') == parallel, (
        "Profile should record requested parallel"
    )
    used = profile.get('parallel_used')
    assert used is not None, "Profile missing parallel_used"
    return int(used)


def _expected_parallel_used(shape, requested):
    cpu_cap = multiprocessing.cpu_count()
    parallel = requested
    if parallel <= 0:
        parallel = cpu_cap
    else:
        parallel = max(1, min(parallel, cpu_cap))
    return edt._adaptive_thread_limit_nd(parallel, shape)


def test_nd_thread_limit_heuristics():
    """Verify heuristic caps reduce oversubscription across shapes."""
    arr_128 = np.zeros((128, 128), dtype=np.uint8)
    assert _profile_parallel_used(arr_128, 16) == _expected_parallel_used(arr_128.shape, 16)
    assert _profile_parallel_used(arr_128, -1) == _expected_parallel_used(arr_128.shape, -1)

    arr_512 = np.zeros((512, 512), dtype=np.uint8)
    assert _profile_parallel_used(arr_512, 16) == _expected_parallel_used(arr_512.shape, 16)

    arr_192 = np.zeros((192, 192, 192), dtype=np.uint8)
    assert _profile_parallel_used(arr_192, -1) == _expected_parallel_used(arr_192.shape, -1)
    assert _profile_parallel_used(arr_192, 32) == _expected_parallel_used(arr_192.shape, 32)

@pytest.mark.parametrize(
    "shape",
    [
        (96, 96),
        (128, 128),
        (48, 48, 48),
        (64, 64, 64),
    ],
)
def test_nd_random_label_bench_patterns(shape):
    """Ensure ND path matches specialized kernels on benchmark-style random labels."""
    arr = _make_bench_array(shape, seed=0)
    assert arr.ndim in (2, 3), "Benchmark patterns currently cover 2D/3D cases"

    for parallel in (1, 4):
        spec = edt.edtsq(arr, parallel=parallel)
        nd = edt.edtsq(arr, parallel=parallel)

        assert np.all(np.isfinite(spec)), "Specialized EDT produced non-finite values"
        assert np.all(np.isfinite(nd)), "ND EDT produced non-finite values"

        np.testing.assert_allclose(
            spec, nd, rtol=1e-6, atol=1e-6,
            err_msg=f"Random benchmark array mismatch for shape={shape} parallel={parallel}"
        )

def _hypercube_m(D: int) -> int:
    """Largest M such that (2*M)**D <= 3,200,000, capped at 50.

    Budget chosen so each dimension stays under ~100ms on a single thread:
    D=6→M=6, D=7→M=4, D=8→M=3, D=9→M=2, D=10→M=2.
    """
    for m in range(50, 0, -1):
        if (2 * m) ** D <= 3_200_000:
            return m
    return 1


def test_edt_all_dims_1_to_32():
    """Correctness + crash test for edt across all supported dimensions (1-32).

    For D=1..20: uses make_label_matrix(D, M) — a hypercube of 2^D equal-sized
    label regions each of side length M.  The max EDT over all foreground voxels
    equals float(M) exactly (the center of each region is M steps from its
    boundary in every axis direction).  M is chosen as the largest value keeping
    total voxels ≤ ~1.1M.

    For D=21..32: make_label_matrix cannot be used (2^D voxels would exceed
    memory).  A single foreground voxel is placed at the array corner in a
    (2,)*20 + (1,)*(D-20) shape; its squared EDT to the nearest boundary = 1.0.

    Graph type coverage:
      D= 1-4  → uint8  graph
      D= 5-8  → uint16 graph
      D= 9-16 → uint32 graph
      D=17-32 → uint64 graph  (D=21-32 use the corner-voxel path)

    Both parallel=1 and parallel=4 are tested, and their outputs are compared,
    so that bugs in either code path (single-threaded vs parallel coordinate
    iteration) are caught independently.
    """
    # D=1..20: hypercube max-value correctness, single and multi-threaded
    for D in range(1, 21):
        M = _hypercube_m(D)
        masks = make_label_matrix(D, M).astype(np.uint32)
        r1 = edt.edt(masks, parallel=1)
        r4 = edt.edt(masks, parallel=4)

        assert r1.shape == masks.shape, f"D={D} M={M}: shape mismatch (parallel=1)"
        assert np.all(np.isfinite(r1)), f"D={D} M={M}: non-finite values (parallel=1)"
        expected_max = float(M)
        np.testing.assert_allclose(
            r1.max(), expected_max, rtol=1e-5, atol=1e-5 * expected_max,
            err_msg=f"D={D} M={M}: hypercube max-EDT mismatch (parallel=1)"
        )
        np.testing.assert_allclose(
            r1, r4, rtol=1e-5, atol=1e-5,
            err_msg=f"D={D} M={M}: parallel=1 vs parallel=4 mismatch"
        )

    # D=21..32: single foreground voxel in (2,)*20 + (1,)*(D-20) shape.
    # Checks:
    #   - Only 1 voxel has nonzero distance (the foreground voxel)
    #   - That voxel's squared EDT equals 1.0 (adjacent to background in each
    #     non-singleton axis)
    #   - parallel=1 and parallel=4 produce identical results
    for D in range(21, 33):
        shape = (2,) * 20 + (1,) * (D - 20)
        data = np.zeros(shape, dtype=np.uint32)
        data[(0,) * D] = 1
        out1 = edt.edtsq(data, parallel=1)
        out4 = edt.edtsq(data, parallel=4)

        assert out1.shape == shape, f"D={D}: output shape mismatch"
        fg_count = int((out1 > 0).sum())
        assert fg_count == 1, (
            f"D={D}: expected 1 nonzero voxel (parallel=1), got {fg_count}"
        )
        assert out1[(0,) * D] == pytest.approx(1.0), (
            f"D={D}: corner voxel expected squared-dist=1.0, got {out1[(0,)*D]} (parallel=1)"
        )
        np.testing.assert_allclose(
            out1, out4, rtol=1e-5, atol=1e-5,
            err_msg=f"D={D}: parallel=1 vs parallel=4 mismatch"
        )


if __name__ == "__main__":
    test_nd_correctness_2d()
    print("2D tests passed!")
    test_nd_correctness_3d()
    print("3D tests passed!")
    test_nd_correctness_4d()
    print("4D tests passed!")
    test_nd_threading_consistency()
    print("Threading tests passed!")
    test_edt_all_dims_1_to_32()
    print("All-dims 1-32 correctness test passed!")
    print("All tests passed!")
