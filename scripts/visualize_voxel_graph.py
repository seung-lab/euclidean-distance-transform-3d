#!/usr/bin/env python3
"""
Visualize voxel_graph comparison: ground truth vs single rect with barriers.
Shows total connectivity (sum of edge bits) instead of raw values.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

import edt as main_edt
import edt_legacy

N = 6  # Each square is NxN (interior)
PAD = 1  # Background border width

# Ground truth: two adjacent NxN squares with different labels, surrounded by background
ground_truth = np.zeros((N + 2*PAD, 2*N + 2*PAD), dtype=np.uint32)
ground_truth[PAD:PAD+N, PAD:PAD+N] = 1  # Left square = label 1
ground_truth[PAD:PAD+N, PAD+N:PAD+2*N] = 2  # Right square = label 2

# Single rectangle: Nx2N interior, all label=1, surrounded by background
single_rect = np.zeros((N + 2*PAD, 2*N + 2*PAD), dtype=np.uint32)
single_rect[PAD:PAD+N, PAD:PAD+2*N] = 1

def build_proper_voxel_graph(labels):
    """Build voxel_graph with edges only between same-label foreground voxels.

    Uses bidirectional format: for 2D, bit layout is:
    - bit 0: +X (right)
    - bit 1: -X (left)
    - bit 2: +Y (down)
    - bit 3: -Y (up)
    """
    graph = np.zeros_like(labels, dtype=np.uint8)
    h, w = labels.shape

    bit_right = 1   # +X
    bit_left = 2    # -X
    bit_down = 4    # +Y
    bit_up = 8      # -Y

    for y in range(h):
        for x in range(w):
            if labels[y, x] == 0:
                continue
            bits = 0
            # Check all 4 neighbors, add edge if same label
            if x + 1 < w and labels[y, x+1] == labels[y, x]:
                bits |= bit_right
            if x > 0 and labels[y, x-1] == labels[y, x]:
                bits |= bit_left
            if y + 1 < h and labels[y+1, x] == labels[y, x]:
                bits |= bit_down
            if y > 0 and labels[y-1, x] == labels[y, x]:
                bits |= bit_up
            graph[y, x] = bits
    return graph

def count_edges(voxel_graph):
    """Count total edges (connectivity) per voxel."""
    connectivity = np.zeros_like(voxel_graph, dtype=np.uint8)
    for bit in range(8):
        connectivity += ((voxel_graph >> bit) & 1).astype(np.uint8)
    return connectivity

# Build voxel graph from ground truth connectivity
voxel_graph = build_proper_voxel_graph(ground_truth)
connectivity = count_edges(voxel_graph)

# Compute EDTs
gt_edt = main_edt.edt(ground_truth)
single_rect_edt = main_edt.edt(single_rect)

# Main EDT with voxel_graph only (no labels needed!)
main_graph_edt = main_edt.edt(voxel_graph=voxel_graph)

# Legacy EDT with voxel_graph (requires labels)
legacy_graph_edt = edt_legacy.edt(single_rect, voxel_graph=voxel_graph)

# Common scale for all images (connectivity max is 4, EDT max is ~3)
vmax = 4

# Dark mode styling (no outlines)
plt.style.use('dark_background')
TEXT_COLOR = '#AAAAAA'
plt.rcParams.update({
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.edgecolor': 'none',
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
})

# Create visualization with space for colorbar on right
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.subplots_adjust(right=0.92)

# Row 1: Labels and connectivity
ax = axes[0, 0]
ax.imshow(ground_truth, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.set_title(f'Ground Truth Labels\n(two {N}x{N} squares with bg border)')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[0, 1]
ax.imshow(single_rect, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.set_title(f'Single Rectangle\n({N}x{2*N} with bg border)')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[0, 2]
ax.imshow(connectivity, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.axvline(x=PAD+N-0.5, color='red', linestyle='--', linewidth=2)
ax.set_title('Voxel Graph Connectivity\n(edges per voxel)')
ax.set_xticks([])
ax.set_yticks([])

# Add text box explaining connectivity (semi-transparent background)
ax = axes[0, 3]
blank = np.zeros_like(ground_truth, dtype=np.float32)
ax.imshow(blank, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax, alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.5, 0.5, 'Edge count per voxel\n\n4 = inside\n3 = edge\n2 = corner\n0 = background\n\nRed line = barrier',
        fontsize=8, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', color=TEXT_COLOR, transform=ax.transAxes)

# Row 2: EDT results (all same scale)
ax = axes[1, 0]
ax.imshow(gt_edt, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.set_title(f'Ground Truth EDT\nmax={gt_edt.max():.2f}')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 1]
ax.imshow(single_rect_edt, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.set_title(f'Single Rect (no graph)\nmax={single_rect_edt.max():.2f}')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 2]
ax.imshow(main_graph_edt, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.axvline(x=PAD+N-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title(f'Main EDT + voxel_graph\nmax={main_graph_edt.max():.2f}')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 3]
im = ax.imshow(legacy_graph_edt, cmap='magma', interpolation='nearest', vmin=0, vmax=vmax)
ax.axvline(x=PAD+N-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title(f'Legacy EDT + voxel_graph\nmax={legacy_graph_edt.max():.2f}')
ax.set_xticks([])
ax.set_yticks([])

# Single colorbar on right side of figure
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.savefig(ROOT / 'voxel_graph_comparison.png', dpi=150, bbox_inches='tight', transparent=True)
print(f"Saved to {ROOT / 'voxel_graph_comparison.png'}")

# Print numerical comparison at center row (middle of foreground)
center_y = PAD + N // 2
print(f"\nNumerical comparison at center row (y={center_y}):")
print(f"Ground truth: {gt_edt[center_y, :]}")
print(f"Main+graph:   {main_graph_edt[center_y, :]}")
print(f"Legacy+graph: {legacy_graph_edt[center_y, :]}")
print(f"\nMain matches ground truth: {np.allclose(main_graph_edt, gt_edt)}")
print(f"Labels needed for main EDT voxel_graph path: NO (voxel_graph alone is sufficient)")
