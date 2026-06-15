# Benchmark Commands

Use these commands to rebuild the extensions and capture ND vs. legacy timings on each platform.

## macOS (local machine)

```bash
# rebuild in-place (optional, keeps .so in src/)
python setup.py build_ext --inplace

# run benchmark sweep and write CSV in milliseconds
EDT_BENCH_MIN_TIME=0.05 EDT_BENCH_MAX_TIME=1.0 \
python scripts/bench_nd_profile.py \
  --parallels 1,2,4,8,16 --dims 2,3 --reps 5 \
  --output benchmarks/nd_profile_mac_20250929.csv
```

To inspect the ND profile for a specific shape:

```bash
python - <<'PY'
import numpy as np, edt, os
os.environ['EDT_ND_PROFILE'] = '1'
arr = np.zeros((384, 384, 384), dtype=np.uint8)
arr[192, 192, 192] = 1
edt.edtsq_nd(arr, parallel=4)
print(edt._nd_profile_last)
PY
```

## Threadripper (remote)

```bash
ssh kcutler@threadripper.local \
  'cd DataDrive/edt && PYTHONPATH=. ~/.pyenv/versions/3.12.11/bin/python \
     scripts/bench_nd_profile.py \
     --parallels 1,2,4,8,16 --dims 2,3 --reps 5 \
     --output benchmarks/nd_profile_threadripper_20250929.csv'
```

Make sure the editable install is up to date first:

```bash
ssh kcutler@threadripper.local \
  'cd DataDrive/edt && PYTHONPATH=. ~/.pyenv/versions/3.12.11/bin/pip install -e .'
```

## Quick ND profile subset → `/tmp/nd_full.csv`

The full CLI sweep (`python scripts/bench_nd_profile.py --output /tmp/nd_full.csv`) currently aborts on this machine when it reuses the RNG state across many shapes. Until we land a proper fix, use this trimmed subset script to capture representative ratios:

```bash
python - <<'PY'
import csv, os
from pathlib import Path
import numpy as np
import scripts.bench_nd_profile as mod

os.environ['EDT_ND_PROFILE'] = '1'
for key in ['EDT_ADAPTIVE_THREADS', 'EDT_ND_AUTOTUNE', 'EDT_ND_THREAD_CAP']:
    os.environ.pop(key, None)

parallels = [1, 4, 8]
shapes = [(96, 96), (128, 128), (192, 192), (48, 48, 48), (64, 64, 64)]
rows = []

for shape in shapes:
    rng = np.random.default_rng(0)
    for parallel in parallels:
        arr = mod.make_array(rng, shape, np.uint8)
        spec_fn, anis = mod.resolve_specialized(len(shape))
        spec_ad, nd_ad, diff_ad, profile_ad = mod.measure_variant(
            arr, parallel, reps=1, spec_fn=spec_fn, anis=anis,
            min_samples=1, min_time=0.002, max_time=0.05,
            overrides={'EDT_ADAPTIVE_THREADS': None,
                       'EDT_ND_AUTOTUNE': None,
                       'EDT_ND_THREAD_CAP': None})
        spec_ex, nd_ex, diff_ex, profile_ex = mod.measure_variant(
            arr, parallel, reps=1, spec_fn=spec_fn, anis=anis,
            min_samples=1, min_time=0.002, max_time=0.05,
            overrides={'EDT_ADAPTIVE_THREADS': '0',
                       'EDT_ND_AUTOTUNE': '0',
                       'EDT_ND_THREAD_CAP': '0'})

        profile_ex = profile_ex or {}
        sections = profile_ex.get('sections', {})
        rows.append({
            'shape': 'x'.join(map(str, shape)),
            'dims': len(shape),
            'parallel_request': parallel,
            'spec_ms_adaptive': spec_ad * 1e3,
            'spec_ms_exact': spec_ex * 1e3,
            'nd_adaptive_ms': nd_ad * 1e3,
            'nd_adaptive_ratio': nd_ad / spec_ad if spec_ad else float('inf'),
            'nd_adaptive_parallel_used': (profile_ad or {}).get('parallel_used'),
            'max_abs_diff_adaptive': diff_ad,
            'nd_exact_ms': nd_ex * 1e3,
            'nd_exact_ratio': nd_ex / spec_ex if spec_ex else float('inf'),
            'nd_exact_parallel_used': profile_ex.get('parallel_used'),
            'max_abs_diff_exact': diff_ex,
            'total_ms': float(sections.get('total', 0.0)) * 1e3,
            'prep_ms': float(sections.get('prep', 0.0)) * 1e3,
            'multi_pass_ms': float(sections.get('multi_pass', 0.0)) * 1e3,
            'parabolic_pass_ms': float(sections.get('parabolic_pass', 0.0)) * 1e3,
            'multi_fix_ms': float(sections.get('multi_fix', 0.0)) * 1e3,
            'post_fix_ms': float(sections.get('post_fix', 0.0)) * 1e3,
            'axes_detail': mod.extract_axes(profile_ex),
        })

out_path = Path('/tmp/nd_full.csv')
with out_path.open('w', newline='') as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} rows to {out_path}')
PY
```
