#!/usr/bin/env python3
"""Benchmark legacy vs ND EDT with explicit thread counts.

Generates a CSV with columns: shape, dims, parallel, legacy_ms, nd_ms, ratio.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import math
import multiprocessing
import os
import time

import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

_override = os.environ.get('EDT_MODULE_PATH')
if _override:
    sys.path.insert(0, _override)

import edt  # noqa: E402
from debug_utils import (
    make_tiled_label_grid_nd,
    make_fibonacci_spiral_labels,
    make_random_circles_labels,
)  # noqa: E402


LEGACY_AUTO_CAP = int(os.environ.get('EDT_BENCH_LEGACY_AUTO_CAP', '64'))


def _require_legacy():
    legacy = getattr(edt, 'legacy', None)
    if legacy is None:
        raise RuntimeError('Legacy module not available; build edt.legacy first.')
    return legacy


def _default_shapes() -> Sequence[Tuple[int, ...]]:
    return [
        (128, 128),
        (512, 512),
        (96, 96, 96),
        (192, 192, 192),
    ]


def _parse_shape2d(text: str) -> Tuple[int, int]:
    parts = tuple(int(x) for x in text.split('x') if x)
    if len(parts) != 2:
        raise ValueError(f"Expected 2D shape like '10x10', got: {text}")
    return parts  # type: ignore[return-value]


def _make_structured_grid_base(
    grid_shape: Tuple[int, ...],
) -> np.ndarray:
    return np.arange(int(np.prod(grid_shape)), dtype=int).reshape(grid_shape)


def _tile_grid(base: np.ndarray, tile_shape: Tuple[int, ...]) -> np.ndarray:
    out = base
    for axis, tile in enumerate(tile_shape):
        out = np.repeat(out, tile, axis=axis)
    return out


def _make_structured_grid(
    shape: Tuple[int, ...],
    base_shape: Tuple[int, ...],
    tile: int | None,
    dtype: np.dtype,
) -> np.ndarray:
    if len(shape) != len(base_shape):
        raise ValueError(f"Shape {shape} dims do not match base {base_shape}")
    if tile is not None:
        tile_shape = (tile,) * len(shape)
    else:
        tile_shape = base_shape
    grid_shape = []
    for s, t in zip(shape, tile_shape):
        if s % t != 0:
            raise ValueError(f"Shape {shape} not divisible by tile {tile_shape}")
        grid_shape.append(s // t)
    base = _make_structured_grid_base(tuple(grid_shape))
    grid = _tile_grid(base, tile_shape)
    if grid.shape != shape:
        raise ValueError(f"Structured grid shape {grid.shape} does not match requested {shape}")
    return grid.astype(dtype, copy=False)


def _apply_label_mod_mask(base: np.ndarray, mod: int) -> np.ndarray:
    if mod <= 1:
        return base
    coords = np.indices(base.shape, dtype=np.uint64)
    h = np.zeros(base.shape, dtype=np.uint64)
    for ax in range(base.ndim):
        h ^= coords[ax] * np.uint64(0x9e3779b97f4a7c15 + ax * 0x85ebca6b)
        h ^= (h >> np.uint64(30))
        h *= np.uint64(0xbf58476d1ce4e5b9)
        h ^= (h >> np.uint64(27))
        h *= np.uint64(0x94d049bb133111eb)
        h ^= (h >> np.uint64(31))
    mask = (h % np.uint64(mod)) == 0
    out = base.copy()
    out[~mask] = 0
    return out


def _make_template_array(
    template: str,
    shape: Tuple[int, ...],
    seed: int,
    dtype: np.dtype,
    grid_base: Tuple[int, ...],
    grid_tile: int | None,
) -> np.ndarray:
    if template == 'structured':
        return _make_structured_grid(shape, grid_base, grid_tile, dtype)
    if template.startswith('structured_mod'):
        mod = int(template.replace('structured_mod', ''))
        if grid_tile is not None:
            tile_shape = (grid_tile,) * len(shape)
        else:
            tile_shape = grid_base
        grid_shape = []
        for s, t in zip(shape, tile_shape):
            if s % t != 0:
                raise ValueError(f"Shape {shape} not divisible by tile {tile_shape}")
            grid_shape.append(s // t)
        base = _make_structured_grid_base(tuple(grid_shape))
        base_masked = _apply_label_mod_mask(base, mod)
        arr = _tile_grid(base_masked, tile_shape)
        if arr.shape != shape:
            raise ValueError(f"Structured grid shape {arr.shape} does not match requested {shape}")
        return arr.astype(dtype, copy=False)
    if template == 'fib_spiral':
        if len(shape) != 2:
            raise ValueError('fib_spiral template only supported for 2D shapes.')
        return make_fibonacci_spiral_labels(shape).astype(dtype, copy=False)
    if template in ('circles_small', 'circles_large'):
        if len(shape) != 2:
            raise ValueError('circle templates only supported for 2D shapes.')
        min_dim = min(shape)
        if template == 'circles_small':
            rmin = max(2, int(min_dim * 0.01))
            rmax = max(rmin + 1, int(min_dim * 0.03))
        else:
            rmin = max(4, int(min_dim * 0.05))
            rmax = max(rmin + 1, int(min_dim * 0.12))
        arr = make_random_circles_labels(shape, rmin, rmax, seed=seed, coverage=0.35)
        return arr.astype(dtype, copy=False)
    raise ValueError(f'Unknown template: {template}')


def _time_and_run(
    fn,
    arr: np.ndarray,
    reps: int,
    *,
    profile_fetcher=None,
) -> tuple[float, np.ndarray | None, Exception | None, dict | None]:
    times: list[float] = []
    last_out: np.ndarray | None = None
    last_profile: dict | None = None
    try:
        fn(arr)  # warmup once
        for _ in range(max(1, reps)):
            start = time.perf_counter()
            out = fn(arr)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            last_out = out
            if profile_fetcher is not None:
                try:
                    last_profile = profile_fetcher()
                except Exception:
                    last_profile = None
    except Exception as exc:  # capture runtime errors (e.g. thread creation failure)
        return float('nan'), None, exc, None
    if not times:
        return float('nan'), last_out, None, last_profile
    return float(np.median(times)), last_out, None, last_profile


def _format_numeric(value: float | int) -> str:
    val = float(value)
    if math.isnan(val):
        return 'nan'
    if math.isinf(val):
        return 'inf' if val > 0 else '-inf'
    return f"{val:.6f}"


def _resolve_legacy_fn(legacy, dims: int):
    if dims == 2:
        return legacy.edt2dsq, (1.0, 1.0)
    if dims == 3:
        return legacy.edt3dsq, (1.0, 1.0, 1.0)
    raise ValueError('Only 2D and 3D shapes supported.')


def run_benchmark(
    shapes: Iterable[Tuple[int, ...]],
    parallels: Sequence[int],
    reps: int,
    dtype: str,
    seeds: Sequence[int],
    output: Path,
    profile: bool,
    template: str,
    grid_base: Tuple[int, int],
    grid_tile: int | None,
) -> None:
    legacy = _require_legacy()
    dtype_np = np.dtype(dtype)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    legacy_baselines: dict[Tuple[Tuple[int, ...], int], float] = {}
    nd_baselines: dict[Tuple[Tuple[int, ...], int], float] = {}

    # Only enable ND profiling if the caller explicitly requested it.
    if profile:
        os.environ['EDT_ND_PROFILE'] = '1'
    else:
        os.environ.pop('EDT_ND_PROFILE', None)

    for shape in shapes:
        dims = len(shape)
        if dims not in (2, 3):
            raise ValueError(f'Shape {shape} has unsupported dims={dims}.')
        legacy_fn, anis = _resolve_legacy_fn(legacy, dims)

        for seed in seeds:
            arr = _make_template_array(template, shape, seed, dtype_np, grid_base, grid_tile)
            baseline_key = (shape, seed)

            for parallel in parallels:
                legacy_call_threads = parallel
                if parallel <= 0:
                    try:
                        auto_threads = multiprocessing.cpu_count()
                    except Exception:
                        auto_threads = 1
                    if LEGACY_AUTO_CAP > 0:
                        auto_threads = min(auto_threads, LEGACY_AUTO_CAP)
                    legacy_call_threads = max(1, auto_threads)

                legacy_time, legacy_out, legacy_err, _ = _time_and_run(
                    lambda a, parallel=legacy_call_threads: legacy_fn(
                        a,
                        anisotropy=anis,
                        black_border=False,
                        parallel=parallel,
                    ),
                    arr,
                    reps,
                )
                nd_time, nd_out, nd_err, nd_profile = _time_and_run(
                    lambda a, parallel=parallel: edt.edtsq(
                        a,
                        anisotropy=anis,
                        black_border=False,
                        parallel=parallel,
                    ),
                    arr,
                    reps,
                    profile_fetcher=lambda: edt._nd_profile_last,
                )

                if legacy_out is not None and nd_out is not None:
                    max_abs_diff = float(np.max(np.abs(legacy_out - nd_out)))
                else:
                    max_abs_diff = float('nan')

                if (
                    parallel == 1
                    and baseline_key not in legacy_baselines
                    and math.isfinite(legacy_time)
                ):
                    legacy_baselines[baseline_key] = legacy_time
                if (
                    parallel == 1
                    and baseline_key not in nd_baselines
                    and math.isfinite(nd_time)
                ):
                    nd_baselines[baseline_key] = nd_time

                base_legacy = legacy_baselines.get(baseline_key)
                base_nd = nd_baselines.get(baseline_key)

                ratio = (
                    nd_time / legacy_time
                    if math.isfinite(nd_time)
                    and math.isfinite(legacy_time)
                    and legacy_time
                    else float('nan')
                )
                legacy_p1_ratio = (
                    legacy_time / base_legacy
                    if base_legacy
                    and math.isfinite(legacy_time)
                    and math.isfinite(base_legacy)
                    else float('nan')
                )
                nd_p1_ratio = (
                    nd_time / base_nd
                    if base_nd and math.isfinite(nd_time) and math.isfinite(base_nd)
                    else float('nan')
                )

                nd_parallel_used = None
                if isinstance(nd_profile, dict):
                    nd_parallel_used = nd_profile.get('parallel_used')

                grouped[('x'.join(map(str, shape)), parallel)].append(
                    {
                        'shape': 'x'.join(map(str, shape)),
                        'dims': dims,
                        'seed': seed,
                        'parallel': parallel,
                        'legacy_ms': legacy_time * 1e3,
                        'nd_ms': nd_time * 1e3,
                        'ratio': ratio,
                        'legacy_p1_ratio': legacy_p1_ratio,
                        'nd_p1_ratio': nd_p1_ratio,
                        'max_abs_diff': max_abs_diff,
                        'legacy_threads_used': legacy_call_threads,
                        'nd_parallel_used': nd_parallel_used,
                        'legacy_error': '' if legacy_err is None else str(legacy_err),
                        'nd_error': '' if nd_err is None else str(nd_err),
                    }
                )

    aggregated_rows: list[dict] = []
    for (shape_str, parallel), entries in grouped.items():
        dims = entries[0]['dims']
        legacy_ms_vals = [e['legacy_ms'] for e in entries if math.isfinite(e['legacy_ms'])]
        nd_ms_vals = [e['nd_ms'] for e in entries if math.isfinite(e['nd_ms'])]
        ratio_vals = [e['ratio'] for e in entries if math.isfinite(e['ratio'])]
        legacy_p1_vals = [e['legacy_p1_ratio'] for e in entries if math.isfinite(e['legacy_p1_ratio'])]
        nd_p1_vals = [e['nd_p1_ratio'] for e in entries if math.isfinite(e['nd_p1_ratio'])]
        max_abs_diff = max((e['max_abs_diff'] for e in entries), default=float('nan'))
        legacy_threads_used = entries[0]['legacy_threads_used']
        nd_used_vals = [e['nd_parallel_used'] for e in entries if e['nd_parallel_used'] is not None]
        legacy_errors = sorted({e['legacy_error'] for e in entries if e['legacy_error']})
        nd_errors = sorted({e['nd_error'] for e in entries if e['nd_error']})

        aggregated_rows.append(
            {
                'shape': shape_str,
                'dims': dims,
                'parallel': parallel,
                'samples': len(entries),
                'legacy_ms': float(np.median(legacy_ms_vals)) if legacy_ms_vals else float('nan'),
                'nd_ms': float(np.median(nd_ms_vals)) if nd_ms_vals else float('nan'),
                'ratio': float(np.median(ratio_vals)) if ratio_vals else float('nan'),
                'legacy_p1_ratio': float(np.median(legacy_p1_vals)) if legacy_p1_vals else float('nan'),
                'nd_p1_ratio': float(np.median(nd_p1_vals)) if nd_p1_vals else float('nan'),
                'max_abs_diff': max_abs_diff,
                'legacy_threads_used': legacy_threads_used,
                'nd_parallel_used': float(np.median(nd_used_vals)) if nd_used_vals else None,
                'legacy_error': '; '.join(legacy_errors),
                'nd_error': '; '.join(nd_errors),
            }
        )

    def sort_key(row: dict) -> tuple[str, int, int]:
        shape = row['shape']
        parallel = row['parallel']
        order_rank = 1 if parallel == -1 else 0
        return (shape, order_rank, parallel if parallel != -1 else 0)

    aggregated_rows.sort(key=sort_key)

    for row in aggregated_rows:
        nd_used_disp = row['nd_parallel_used']
        if nd_used_disp is not None:
            nd_used_disp = int(round(nd_used_disp))
        print(
            f"shape={row['shape']:<12} p={row['parallel']:<3d} "
            f"legacy={row['legacy_ms']:.3f}ms nd={row['nd_ms']:.3f}ms "
            f"ratio={row['ratio']:.3f} legacy/p1={row['legacy_p1_ratio']:.3f} "
            f"nd/p1={row['nd_p1_ratio']:.3f} diff={row['max_abs_diff']:.3e} "
            f"legacy_used={row['legacy_threads_used']} nd_used={nd_used_disp} "
            f"samples={row['samples']}"
            + (f" legacy_err={row['legacy_error']}" if row['legacy_error'] else '')
            + (f" nd_err={row['nd_error']}" if row['nd_error'] else '')
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                'shape',
                'dims',
                'parallel',
                'samples',
                'legacy_ms',
                'nd_ms',
                'ratio',
                'legacy_p1_ratio',
                'nd_p1_ratio',
                'max_abs_diff',
                'legacy_threads_used',
                'nd_parallel_used',
                'legacy_error',
                'nd_error',
            ],
        )
        writer.writeheader()
        formatted_rows = []
        for row in aggregated_rows:
            formatted = row.copy()
            for key in ['legacy_ms', 'nd_ms', 'ratio', 'legacy_p1_ratio', 'nd_p1_ratio', 'max_abs_diff']:
                formatted[key] = _format_numeric(row[key])
            formatted['legacy_threads_used'] = str(row['legacy_threads_used'])
            nd_used = row.get('nd_parallel_used')
            formatted['nd_parallel_used'] = '' if nd_used is None else str(int(round(nd_used)))
            formatted['samples'] = str(row['samples'])
            formatted_rows.append(formatted)
        writer.writerows(formatted_rows)
    print(f"\nWrote {len(aggregated_rows)} rows to {output}")


def parse_shapes(spec: str) -> Sequence[Tuple[int, ...]]:
    if not spec:
        return _default_shapes()
    shapes = []
    for token in spec.split(','):
        parts = tuple(int(x) for x in token.split('x') if x)
        if parts:
            shapes.append(parts)
    return shapes or _default_shapes()


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark legacy vs ND with explicit threads.')
    parser.add_argument('--parallels', default='1,4,8,16,-1', help='Comma-separated thread counts to test.')
    parser.add_argument('--shapes', default='', help='Comma-separated shapes like "128x128,96x96x96".')
    parser.add_argument('--reps', type=int, default=5, help='Repetitions per measurement.')
    parser.add_argument('--dtype', default='uint8', help='Array dtype.')
    parser.add_argument('--seeds', default='0', help='Comma-separated RNG seeds for inputs (e.g. "0,1,2,3,4").')
    parser.add_argument('--output', default=str(ROOT / 'benchmarks' / 'legacy_vs_nd_explicit.csv'))
    parser.add_argument('--profile', action='store_true', help='Enable EDT_ND_PROFILE during benchmarking.')
    parser.add_argument('--structured-grid', action='store_true',
                        help='Use a structured tiled label grid for 2D shapes (deprecated; use --template).')
    parser.add_argument('--template', default='structured',
                        choices=[
                            'structured',
                            'structured_mod2',
                            'structured_mod4',
                            'structured_mod8',
                            'fib_spiral',
                            'circles_small',
                            'circles_large',
                        ],
                        help='Input template to benchmark.')
    parser.add_argument('--grid-base', default='10x10',
                        help="Base grid size for structured labels, e.g. 10x10 or 10x10x10.")
    parser.add_argument('--grid-tile', default='',
                        help='Tile size for structured grid. If omitted, inferred from shape.')
    parser.add_argument('--nd-batch', default='',
                        help='ND multi-seg batch size (1-32).')
    args = parser.parse_args()

    parallels = [int(x) for x in args.parallels.split(',') if x.strip()]
    if not parallels:
        raise ValueError('At least one parallel value required.')
    shapes = parse_shapes(args.shapes)
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    if not seeds:
        raise ValueError('At least one seed required.')
    if 'x' not in args.grid_base:
        raise ValueError("grid-base must be like '10x10' or '10x10x10'")
    grid_base = tuple(int(x) for x in args.grid_base.split('x') if x)
    grid_tile = int(args.grid_tile) if args.grid_tile else None
    template = args.template
    if args.structured_grid:
        template = 'structured'
    if args.nd_batch:
        edt.nd_multi_batch(int(args.nd_batch))
    run_benchmark(
        shapes,
        parallels,
        args.reps,
        args.dtype,
        seeds,
        Path(args.output),
        args.profile,
        template,
        grid_base,
        grid_tile,
    )


if __name__ == '__main__':
    main()
