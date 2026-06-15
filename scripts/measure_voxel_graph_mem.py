#!/usr/bin/env python3
"""Measure voxel_graph memory usage for src vs legacy (baseline-corrected)."""
import subprocess
import sys

def measure_rss(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr[:200]}")
        return None
    for line in result.stdout.strip().split("\n"):
        if "peak_mb" in line:
            return float(line.split("=")[1])
    return None

N_2d = 4000*4000
N_3d = 300*300*300

# Measure baseline (imports only, no work)
baseline_code = """
import numpy as np
import resource
import sys
import edt
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak = peak / (1024*1024)
else:
    peak = peak / 1024
print(f"peak_mb={peak:.2f}")
"""
baseline = measure_rss(baseline_code)

print("voxel_graph memory measurement (baseline-corrected)")
print("=" * 50)
print(f"Baseline (Python+numpy+edt imports): {baseline:.1f} MB\n")

# 2D src
code = """
import numpy as np
import resource
import sys
import edt
graph = np.ones((4000, 4000), dtype=np.uint8) * 0b0101
graph[0, :] = 0
result = edt.edtsq(voxel_graph=graph)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak = peak / (1024*1024)
else:
    peak = peak / 1024
print(f"peak_mb={peak:.2f}")
"""
src_2d_abs = measure_rss(code)
src_2d = src_2d_abs - baseline if src_2d_abs else None
if src_2d:
    print(f"2D src: {src_2d:.1f} MB delta, {src_2d*1024*1024/N_2d:.1f} bytes/voxel")

# 2D legacy
code = """
import numpy as np
import resource
import sys
import edt
labels = np.ones((4000, 4000), dtype=np.uint16)
labels[0, :] = 0
graph = np.ones((4000, 4000), dtype=np.uint8) * 0b0101
graph[0, :] = 0
result = edt.legacy.edtsq(labels, voxel_graph=graph)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak = peak / (1024*1024)
else:
    peak = peak / 1024
print(f"peak_mb={peak:.2f}")
"""
leg_2d_abs = measure_rss(code)
leg_2d = leg_2d_abs - baseline if leg_2d_abs else None
if leg_2d:
    print(f"2D legacy: {leg_2d:.1f} MB delta, {leg_2d*1024*1024/N_2d:.1f} bytes/voxel")
if src_2d and leg_2d:
    print(f"2D ratio: {leg_2d/src_2d:.2f}x (theoretical 3.8x)")

# 3D src
code = """
import numpy as np
import resource
import sys
import edt
graph = np.ones((300, 300, 300), dtype=np.uint8) * 0b010101
graph[0, :, :] = 0
result = edt.edtsq(voxel_graph=graph)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak = peak / (1024*1024)
else:
    peak = peak / 1024
print(f"peak_mb={peak:.2f}")
"""
src_3d_abs = measure_rss(code)
src_3d = src_3d_abs - baseline if src_3d_abs else None
if src_3d:
    print(f"\n3D src: {src_3d:.1f} MB delta, {src_3d*1024*1024/N_3d:.1f} bytes/voxel")

# 3D legacy
code = """
import numpy as np
import resource
import sys
import edt
labels = np.ones((300, 300, 300), dtype=np.uint16)
labels[0, :, :] = 0
graph = np.ones((300, 300, 300), dtype=np.uint8) * 0b010101
graph[0, :, :] = 0
result = edt.legacy.edtsq(labels, voxel_graph=graph)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    peak = peak / (1024*1024)
else:
    peak = peak / 1024
print(f"peak_mb={peak:.2f}")
"""
leg_3d_abs = measure_rss(code)
leg_3d = leg_3d_abs - baseline if leg_3d_abs else None
if leg_3d:
    print(f"3D legacy: {leg_3d:.1f} MB delta, {leg_3d*1024*1024/N_3d:.1f} bytes/voxel")
if src_3d and leg_3d:
    print(f"3D ratio: {leg_3d/src_3d:.2f}x (theoretical 7.2x)")

print("\nTheoretical (uint16 labels):")
print("  2D: 6N vs 23N -> 3.8x")
print("  3D: 6N vs 43N -> 7.2x")
