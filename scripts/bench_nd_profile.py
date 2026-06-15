#!/usr/bin/env python3
"""
Benchmark ND vs specialized paths with configurable autotune/thread-cap settings
and capture detailed ND profiling data.

Usage examples:
  ./scripts/bench_nd_profile.py --parallels 4,8,16 --output benchmarks/nd_profile_runs.csv
"""
# PEP 563: keep PEP 604 (X | None) and PEP 585 (list[...]) annotations as lazy
# strings so this module imports under Python 3.8/3.9 (CI runs the full matrix).
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
import edt  # noqa: E402


def _require_legacy():
    legacy = getattr(edt, 'legacy', None)
    if legacy is None:
        raise ImportError(
            "The legacy edt.legacy extension is required for benchmarking. "
            "Please build/install the legacy module (e.g. `pip install -e .`)."
        )
    return legacy


def resolve_specialized(dims: int):
    legacy = _require_legacy()

    if dims == 1:
        def spec(arr, anisotropy, black_border, parallel):
            scalar = anisotropy[0] if isinstance(anisotropy, (tuple, list)) else float(anisotropy)
            return legacy.edt1dsq(arr, anisotropy=scalar, black_border=black_border)

        return spec, (1.0,)

    if dims == 2:
        def spec(arr, anisotropy, black_border, parallel):
            return legacy.edt2dsq(arr, anisotropy=anisotropy, black_border=black_border, parallel=parallel)

        return spec, (1.0, 1.0)

    if dims == 3:
        def spec(arr, anisotropy, black_border, parallel):
            return legacy.edt3dsq(arr, anisotropy=anisotropy, black_border=black_border, parallel=parallel)

        return spec, (1.0, 1.0, 1.0)
    raise ValueError(f"No specialized EDT available for {dims}D.")


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def default_shapes(dims: Sequence[int]) -> List[Tuple[int, ...]]:
    shapes: List[Tuple[int, ...]] = []
    if 1 in dims:
        shapes.extend([(256,), (1024,), (4096,)])
    if 2 in dims:
        shapes.extend([
            (96, 96),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ])
    if 3 in dims:
        shapes.extend([
            (48, 48, 48),
            (64, 64, 64),
            (96, 96, 96),
            (128, 128, 128),
            (256, 256, 256),
            (384, 384, 384),
        ])
    return shapes


def make_array(rng: np.random.Generator, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    arr = rng.integers(0, 3, size=shape, dtype=dtype)
    if arr.ndim == 1:
        length = shape[0]
        if length > 8:
            arr[: length // 4] = 0
            arr[length // 4 : length // 2] = 1
            arr[3 * length // 4 :] = 2
    elif arr.ndim == 2:
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


@dataclass
class BenchmarkSample:
    spec_times: list[float]
    nd_times: list[float]
    max_diff: float
    profile: dict | None


def run_once(
    arr: np.ndarray,
    parallel: int,
    reps: int,
    spec_fn,
    anis: Tuple[float, ...],
) -> BenchmarkSample:
    # warmup
    spec_fn(arr, anisotropy=anis, black_border=False, parallel=parallel)
    edt.edtsq(arr, parallel=parallel)

    spec_times: list[float] = []
    nd_times: list[float] = []
    max_diff = 0.0
    last_profile: dict | None = None

    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        spec_tmp = spec_fn(arr, anisotropy=anis, black_border=False, parallel=parallel)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        nd_tmp = edt.edtsq(arr, parallel=parallel)
        t3 = time.perf_counter()

        spec_times.append(t1 - t0)
        nd_times.append(t3 - t2)

        diff = float(np.max(np.abs(spec_tmp - nd_tmp)))
        if diff > max_diff:
            max_diff = diff

        if os.environ.get('EDT_ND_PROFILE'):
            last_profile = edt._nd_profile_last

    return BenchmarkSample(spec_times, nd_times, max_diff, last_profile)


@contextmanager
def temporary_env(overrides: dict[str, str | None]):
    sentinel = object()
    previous: dict[str, object] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key, sentinel)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old  # type: ignore[arg-type]


def measure_variant(
    arr: np.ndarray,
    parallel: int,
    reps: int,
    spec_fn,
    anis: Tuple[float, ...],
    min_samples: int,
    min_time: float,
    max_time: float,
    overrides: dict[str, str | None],
) -> Tuple[float, float, float, dict]:
    def run_sample() -> BenchmarkSample:
        with temporary_env(overrides):
            return run_once(arr, parallel, reps, spec_fn, anis)

    sample = run_sample()
    spec_times = list(sample.spec_times)
    nd_times = list(sample.nd_times)
    max_diff = sample.max_diff
    profile = sample.profile

    nd_time = min(nd_times) if nd_times else float('inf')
    repeat_count = adaptive_repeat(nd_time, min_samples=min_samples, min_time=min_time, max_time=max_time)
    if repeat_count > 1:
        for _ in range(repeat_count - 1):
            sample_r = run_sample()
            spec_times.extend(sample_r.spec_times)
            nd_times.extend(sample_r.nd_times)
            if sample_r.max_diff > max_diff:
                max_diff = sample_r.max_diff
            if sample_r.profile:
                profile = sample_r.profile
        spec_time_val = float(np.mean(spec_times))
        nd_time_val = float(np.mean(nd_times))
    else:
        spec_time_val = min(spec_times)
        nd_time_val = nd_time

    if profile is None:
        with temporary_env(overrides):
            profile = edt._nd_profile_last or {}
    else:
        profile = profile or {}
    return spec_time_val, nd_time_val, max_diff, profile


def adaptive_repeat(time_s: float, min_samples: int, min_time: float, max_time: float) -> int:
    if time_s <= 0:
        return min_samples
    reps = max(min_samples, int(min_time / time_s))
    reps = min(reps, int(max_time / max(time_s, 1e-9)))
    if reps < 1:
        reps = 1
    return reps


def extract_axes(profile: dict) -> str:
    axes = profile.get('axes', [])
    parts = []
    for entry in axes:
        parts.append(f"{entry.get('kind')}@{entry.get('axis')}:{float(entry.get('time', 0.0)):.6f}")
    return ';'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ND profiling scenarios")
    parser.add_argument('--parallels', default='4,8,16', help='Comma separated list of parallel values')
    parser.add_argument('--dims', default='1,2,3', help='Comma separated dimensions to test (e.g. "1,2,3")')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions per timing')
    parser.add_argument('--dtype', default='uint8', help='NumPy dtype for test arrays')
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'nd_profile_runs.csv'), help='Output CSV path')
    parser.add_argument('--no-header', action='store_true', help='Skip CSV header in the output file')
    parser.add_argument('--disable-tuning', action='store_true', help='Disable adaptive tuning for specialized and ND paths')
    args = parser.parse_args()

    parallels = parse_int_list(args.parallels)
    dims_requested = parse_int_list(args.dims)
    dtype = np.dtype(args.dtype)
    shapes = default_shapes(dims_requested)
    # Use quicker defaults so a single sweep stays snappy unless overridden by env vars.
    min_time = float(os.environ.get('EDT_BENCH_MIN_TIME', '0.05'))
    max_time = float(os.environ.get('EDT_BENCH_MAX_TIME', '1.0'))
    min_samples = int(os.environ.get('EDT_BENCH_MIN_REPEAT', '1'))
    rng = np.random.default_rng(0)

    if args.disable_tuning:
        os.environ['EDT_ADAPTIVE_THREADS'] = '0'
        os.environ['EDT_ND_AUTOTUNE'] = '0'
        os.environ['EDT_ND_THREAD_CAP'] = '0'
    else:
        os.environ.pop('EDT_ADAPTIVE_THREADS', None)
        os.environ.pop('EDT_ND_AUTOTUNE', None)
        os.environ.pop('EDT_ND_THREAD_CAP', None)
    os.environ['EDT_ND_PROFILE'] = '1'

    rows = []
    adaptive_overrides = {
        'EDT_ADAPTIVE_THREADS': None,
        'EDT_ND_AUTOTUNE': None,
        'EDT_ND_THREAD_CAP': None,
    }
    exact_overrides = {
        'EDT_ADAPTIVE_THREADS': '0',
        'EDT_ND_AUTOTUNE': '0',
        'EDT_ND_THREAD_CAP': '0',
    }
    for parallel in parallels:
        for shape in shapes:
            arr = make_array(rng, shape, dtype)
            spec_fn, anis = resolve_specialized(arr.ndim)

            if args.disable_tuning:
                spec_ad = spec_exact = 0.0  # placeholder will be overwritten below
                spec_exact, nd_exact, diff_exact, profile_exact = measure_variant(
                    arr,
                    parallel,
                    args.reps,
                    spec_fn,
                    anis,
                    min_samples,
                    min_time,
                    max_time,
                    exact_overrides,
                )
                spec_ad = spec_exact
                adaptive_summary = {
                    'nd_ms': nd_exact * 1000.0,
                    'ratio': nd_exact / spec_exact if spec_exact else float('inf'),
                    'parallel_used': profile_exact.get('parallel_used') if profile_exact else None,
                    'diff': diff_exact,
                }
                exact_summary = adaptive_summary
            else:
                spec_ad, nd_ad, diff_ad, profile_ad = measure_variant(
                    arr,
                    parallel,
                    args.reps,
                    spec_fn,
                    anis,
                    min_samples,
                    min_time,
                    max_time,
                    adaptive_overrides,
                )
                spec_exact, nd_exact, diff_exact, profile_exact = measure_variant(
                    arr,
                    parallel,
                    args.reps,
                    spec_fn,
                    anis,
                    min_samples,
                    min_time,
                    max_time,
                    exact_overrides,
                )
                adaptive_summary = {
                    'nd_ms': nd_ad * 1000.0,
                    'ratio': nd_ad / spec_ad if spec_ad else float('inf'),
                    'parallel_used': profile_ad.get('parallel_used') if profile_ad else None,
                    'diff': diff_ad,
                }
                exact_summary = {
                    'nd_ms': nd_exact * 1000.0,
                    'ratio': nd_exact / spec_exact if spec_exact else float('inf'),
                    'parallel_used': profile_exact.get('parallel_used') if profile_exact else None,
                    'diff': diff_exact,
                }
            profile_exact = profile_exact or {}
            sections = profile_exact.get('sections', {})
            row = {
                'shape': 'x'.join(str(s) for s in shape),
                'dims': len(shape),
                'parallel_request': parallel,
                'spec_ms_adaptive': spec_ad * 1000.0,
                'spec_ms_exact': spec_exact * 1000.0,
                'nd_adaptive_ms': adaptive_summary['nd_ms'],
                'nd_adaptive_ratio': adaptive_summary['ratio'],
                'nd_adaptive_parallel_used': adaptive_summary['parallel_used'],
                'max_abs_diff_adaptive': adaptive_summary['diff'],
                'nd_exact_ms': exact_summary['nd_ms'],
                'nd_exact_ratio': exact_summary['ratio'],
                'nd_exact_parallel_used': exact_summary['parallel_used'],
                'max_abs_diff_exact': exact_summary['diff'],
                'total_ms': float(sections.get('total', 0.0)) * 1000.0,
                'prep_ms': float(sections.get('prep', 0.0)) * 1000.0,
                'multi_pass_ms': float(sections.get('multi_pass', 0.0)) * 1000.0,
                'parabolic_pass_ms': float(sections.get('parabolic_pass', 0.0)) * 1000.0,
                'multi_fix_ms': float(sections.get('multi_fix', 0.0)) * 1000.0,
                'post_fix_ms': float(sections.get('post_fix', 0.0)) * 1000.0,
                'axes_detail': extract_axes(profile_exact),
            }
            rows.append(row)
            if args.disable_tuning:
                print(
                    f"shape={row['shape']:<12} p={parallel:<3d} spec={row['spec_ms_exact']:.3f}ms "
                    f"exact={row['nd_exact_ms']:.3f}ms exact/spec={row['nd_exact_ratio']:.3f} "
                    f"used={row['nd_exact_parallel_used']} diff={row['max_abs_diff_exact']:.3e}"
                )
            else:
                print(
                    f"shape={row['shape']:<12} p={parallel:<3d} spec_ad={row['spec_ms_adaptive']:.3f}ms "
                    f"adapt={row['nd_adaptive_ms']:.3f}ms adapt/spec={row['nd_adaptive_ratio']:.3f} "
                    f"used={row['nd_adaptive_parallel_used']} spec_ex={row['spec_ms_exact']:.3f}ms "
                    f"exact={row['nd_exact_ms']:.3f}ms "
                    f"exact/spec={row['nd_exact_ratio']:.3f} used={row['nd_exact_parallel_used']} "
                    f"diff_adapt={row['max_abs_diff_adaptive']:.3e} diff_exact={row['max_abs_diff_exact']:.3e}"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open('w', newline='') as fp:
        if fieldnames:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if not args.no_header:
                writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path} (overwritten)")


if __name__ == '__main__':
    main()
