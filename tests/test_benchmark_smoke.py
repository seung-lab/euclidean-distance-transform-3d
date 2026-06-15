import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edt  # noqa: E402
import scripts.bench_nd_profile as mod  # noqa: E402


SMOKE_CSV = ROOT / "benchmarks" / "nd_smoke.csv"


def _require_legacy():
  import pytest
  legacy = getattr(edt, 'legacy', None)
  if legacy is None:
    pytest.skip(
        "edt.legacy must be built for benchmark tests. "
        "Run `pip install -e .` to compile the legacy extension."
    )
  return legacy


def _generate_smoke_csv(path: Path = SMOKE_CSV):
  """Create a tiny benchmark CSV so CI artifacts can capture the result."""
  legacy = _require_legacy()
  shapes = [(32, 32), (16, 16, 16)]
  rows = []
  for shape in shapes:
    rng = np.random.default_rng(0)
    arr = mod.make_array(rng, shape, np.uint8)
    if len(shape) == 1:
      spec_fn = lambda a, anisotropy, black_border, parallel: legacy.edt1dsq(
          a, anisotropy=anisotropy[0], black_border=black_border)
      anis = (1.0,)
    elif len(shape) == 2:
      spec_fn = lambda a, anisotropy, black_border, parallel: legacy.edt2dsq(
          a, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
      anis = (1.0, 1.0)
    else:
      spec_fn = lambda a, anisotropy, black_border, parallel: legacy.edt3dsq(
          a, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
      anis = (1.0, 1.0, 1.0)
    spec, nd, diff, _ = mod.measure_variant(
        arr,
        parallel=1,
        reps=1,
        spec_fn=spec_fn,
        anis=anis,
        min_samples=1,
        min_time=0.001,
        max_time=0.02,
        overrides={
            'EDT_ADAPTIVE_THREADS': None,
            'EDT_ND_AUTOTUNE': None,
            'EDT_ND_THREAD_CAP': None,
        },
    )
    rows.append({
        'shape': 'x'.join(map(str, shape)),
        'dims': len(shape),
        'parallel_request': 1,
        'spec_ms': spec * 1e3,
        'nd_ms': nd * 1e3,
        'ratio': nd / spec if spec else float('inf'),
        'max_abs_diff': diff,
    })

  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  return rows


def test_benchmark_script_import_and_measure():
  legacy = _require_legacy()
  rng = np.random.default_rng(0)
  arr = mod.make_array(rng, (16, 16), np.uint8)
  spec_fn = lambda a, anisotropy, black_border, parallel: legacy.edt2dsq(
      a, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
  spec, nd, diff, _ = mod.measure_variant(
      arr,
      parallel=1,
      reps=1,
      spec_fn=spec_fn,
      anis=(1.0, 1.0),
      min_samples=1,
      min_time=0.001,
      max_time=0.01,
      overrides={
          'EDT_ADAPTIVE_THREADS': None,
          'EDT_ND_AUTOTUNE': None,
          'EDT_ND_THREAD_CAP': None,
      },
  )
  assert np.isfinite(spec) and spec > 0
  assert np.isfinite(nd) and nd > 0
  assert diff >= 0


def test_benchmark_smoke_csv_written_and_sane():
  if SMOKE_CSV.exists():
    SMOKE_CSV.unlink()
  rows = _generate_smoke_csv(SMOKE_CSV)
  try:
    assert SMOKE_CSV.exists(), "Smoke benchmark did not produce a CSV."
    assert rows, "Smoke benchmark CSV contains no rows."
    for row in rows:
      ratio = float(row['ratio'])
      assert np.isfinite(ratio) and ratio > 0.0
      assert ratio < 1.5, f"Expected ND path faster than spec in smoke data; got {ratio}"
  finally:
    if SMOKE_CSV.exists():
      SMOKE_CSV.unlink()
