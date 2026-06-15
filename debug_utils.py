# Lazy annotations so the PEP 585 generics in the type hints below (e.g.
# tuple[int, ...]) import on Python 3.8, which CI still tests. Must be first.
from __future__ import annotations

import numpy as np


def make_label_matrix(ndim: int, size: int) -> np.ndarray:
    """
    General ND label matrix.

    Shape = (2*size,)*ndim
    Each axis is split into two halves of length size.
    The label is the binary code of the half-indices.

    Example:
      ndim=1 → [0...0,1...1]
      ndim=2 → quadrants labeled 0..3
      ndim=3 → octants labeled 0..7
      ndim=4 → 16 hyper-quadrants labeled 0..15
    """
    if ndim < 1:
        raise ValueError("ndim must be >=1")
    grids = np.ogrid[tuple(slice(0, 2 * size) for _ in range(ndim))]
    labels = np.zeros((2 * size,) * ndim, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // size).astype(int)  # 0 or 1
        labels += half << axis
    return labels


def then(N: int, M: int) -> np.ndarray:
    """Backwards-compatible alias for make_label_matrix."""
    return make_label_matrix(N, M)


def make_tiled_label_grid(base_shape: tuple[int, int], tile: int) -> np.ndarray:
    """
    Create a 2D grid where each pixel is a unique label, then upscale by tiling.

    Example:
      base_shape=(10, 10), tile=100 -> output shape (1000, 1000)
      labels are 0..99 expanded into 100x100 blocks.
    """
    return make_tiled_label_grid_nd(base_shape, tile)


def make_tiled_label_grid_nd(base_shape: tuple[int, ...], tile: int, gap: int = 0) -> np.ndarray:
    """
    Create an ND grid where each voxel is a unique label, then upscale by tiling.

    Parameters:
        base_shape: Number of tiles in each dimension
        tile: Size of each tile in pixels
        gap: Gap between tiles (background pixels). Default 0.

    When gap=0: Labels start from 0, so first tile is background.
    When gap>0: Labels start from 1, gaps are background (label 0).

    Example:
      base_shape=(10, 10, 10), tile=20 -> output shape (200, 200, 200)
      labels are 0..999 expanded into 20x20x20 blocks.
    """
    if len(base_shape) < 1:
        raise ValueError("base_shape must be at least 1D.")
    if tile < 1:
        raise ValueError("tile must be >= 1.")

    if gap == 0:
        # Original fast path: labels 0..N-1, first tile is background
        base = np.arange(int(np.prod(base_shape)), dtype=int).reshape(base_shape)
        for axis in range(len(base_shape)):
            base = np.repeat(base, tile, axis=axis)
        return base
    else:
        # With gaps: labels 1..N, gaps are background (0)
        if gap >= tile:
            raise ValueError("gap must be < tile.")
        output_shape = tuple(s * tile for s in base_shape)
        output = np.zeros(output_shape, dtype=int)
        fill_size = tile - gap
        label = 1
        for tile_idx in np.ndindex(*base_shape):
            slices = tuple(slice(idx * tile, idx * tile + fill_size) for idx in tile_idx)
            output[slices] = label
            label += 1
        return output


def make_fibonacci_spiral_labels(shape: tuple[int, int]) -> np.ndarray:
    """
    Create a 2D Fibonacci-style spiral of filled squares.

    Starts with a 1x1 label at the center, then grows outward with
    square sizes following the Fibonacci sequence.
    """
    if len(shape) != 2:
        raise ValueError("shape must be 2D for fibonacci spiral.")
    h, w = shape
    labels = np.zeros((h, w), dtype=int)

    # Fibonacci sizes
    sizes = [1, 1]
    while sizes[-1] < max(h, w):
        sizes.append(sizes[-1] + sizes[-2])

    # Start center
    cy, cx = h // 2, w // 2
    top = bottom = cy
    left = right = cx
    label_id = 1
    labels[cy, cx] = label_id

    # Directions: right, down, left, up
    directions = ["right", "down", "left", "up"]
    dir_idx = 0

    for size in sizes[1:]:
        direction = directions[dir_idx % 4]
        if direction == "right":
            new_top = top
            new_left = right + 1
        elif direction == "down":
            new_top = bottom + 1
            new_left = right - size + 1
        elif direction == "left":
            new_top = bottom - size + 1
            new_left = left - size
        else:  # up
            new_top = top - size
            new_left = left

        new_bottom = new_top + size - 1
        new_right = new_left + size - 1

        # Clip to image bounds (keep growing to fill the space)
        clip_top = max(new_top, 0)
        clip_left = max(new_left, 0)
        clip_bottom = min(new_bottom, h - 1)
        clip_right = min(new_right, w - 1)

        if clip_top <= clip_bottom and clip_left <= clip_right:
            label_id += 1
            labels[clip_top:clip_bottom + 1, clip_left:clip_right + 1] = label_id

            # Expand bounding box (clipped)
            top = min(top, clip_top)
            left = min(left, clip_left)
            bottom = max(bottom, clip_bottom)
            right = max(right, clip_right)
            dir_idx += 1

        # Stop when we've filled the full image
        if top == 0 and left == 0 and bottom == h - 1 and right == w - 1:
            break

    return labels


def make_random_hyperspheres_labels(
    shape: tuple[int, ...],
    rmin: int,
    rmax: int,
    seed: int = 0,
    coverage: float = 0.3,
) -> np.ndarray:
    """
    Create an ND label array with random filled hyperspheres, each with a unique label.

    Works for any dimensionality:
    - 2D: circles
    - 3D: spheres
    - ND: hyperspheres

    Hyperspheres may overlap (later ones overwrite earlier ones).
    Uses local bounding box for O(r^ndim) per sphere instead of O(volume).
    """
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("shape must be at least 1D.")
    if rmin < 1 or rmax < rmin:
        raise ValueError("invalid radius range")

    rng = np.random.default_rng(seed)
    total_volume = int(np.prod(shape))
    r_mean = (rmin + rmax) / 2.0

    # Hypersphere volume formula: V = pi^(n/2) / Gamma(n/2 + 1) * r^n
    # Simplified approximation for counting
    if ndim == 1:
        sphere_volume = 2 * r_mean
    elif ndim == 2:
        sphere_volume = np.pi * r_mean ** 2
    elif ndim == 3:
        sphere_volume = (4.0 / 3.0) * np.pi * r_mean ** 3
    else:
        # General formula using gamma function
        from math import gamma
        sphere_volume = (np.pi ** (ndim / 2)) / gamma(ndim / 2 + 1) * r_mean ** ndim

    count = max(1, int(coverage * total_volume / sphere_volume))
    labels = np.zeros(shape, dtype=np.int32)

    for label_id in range(1, count + 1):
        r = rng.integers(rmin, rmax + 1)

        # Random center, staying r away from edges when possible
        center = []
        for dim_size in shape:
            if dim_size > 2 * r:
                c = rng.integers(r, dim_size - r)
            else:
                c = rng.integers(0, dim_size)
            center.append(c)

        # Build local bounding box slices
        slices = []
        for ax, (c, dim_size) in enumerate(zip(center, shape)):
            lo = max(0, c - r)
            hi = min(dim_size, c + r + 1)
            slices.append(slice(lo, hi))

        # Build distance mask using ogrid for efficiency
        ogrid_slices = [np.arange(s.start, s.stop) for s in slices]
        grids = np.ogrid[tuple(slice(0, len(og)) for og in ogrid_slices)]

        # Compute squared distance from center
        dist_sq = sum((g + slices[ax].start - center[ax]) ** 2 for ax, g in enumerate(grids))
        mask = dist_sq <= r * r

        # Apply label
        labels[tuple(slices)][mask] = label_id

    return labels


def make_random_circles_labels(
    shape: tuple[int, int],
    rmin: int,
    rmax: int,
    seed: int = 0,
    coverage: float = 0.3,
) -> np.ndarray:
    """Backwards-compatible alias for 2D hyperspheres (circles)."""
    if len(shape) != 2:
        raise ValueError("shape must be 2D for random circles.")
    return make_random_hyperspheres_labels(shape, rmin, rmax, seed, coverage)


def make_random_spheres_labels(
    shape: tuple[int, int, int],
    rmin: int,
    rmax: int,
    seed: int = 0,
    coverage: float = 0.3,
) -> np.ndarray:
    """Backwards-compatible alias for 3D hyperspheres (spheres)."""
    if len(shape) != 3:
        raise ValueError("shape must be 3D for random spheres.")
    return make_random_hyperspheres_labels(shape, rmin, rmax, seed, coverage)


def make_random_boxes_labels(
    shape: tuple[int, ...],
    size_min: int = None,
    size_max: int = None,
    seed: int = 0,
    num_boxes: int = 50,
) -> np.ndarray:
    """
    Create an ND label array with random axis-aligned boxes stacked/overlapping.

    Works for any dimensionality:
    - 2D: rectangles (squares)
    - 3D: cubes
    - ND: hypercubes
    """
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("shape must be at least 1D.")

    min_dim = min(shape)
    if size_min is None:
        size_min = max(1, min_dim // 20)
    if size_max is None:
        size_max = max(size_min + 1, min_dim // 4)

    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.int32)

    for label_id in range(1, num_boxes + 1):
        size = rng.integers(size_min, size_max + 1)
        slices = []
        for dim_size in shape:
            lo = rng.integers(0, max(1, dim_size - size))
            hi = min(dim_size, lo + size)
            slices.append(slice(lo, hi))
        labels[tuple(slices)] = label_id

    return labels


def make_cube_stack_labels(
    shape: tuple[int, int, int],
    seed: int = 0,
    num_cubes: int = 50,
) -> np.ndarray:
    """Backwards-compatible alias for 3D random boxes."""
    if len(shape) != 3:
        raise ValueError("shape must be 3D for cube stack.")
    return make_random_boxes_labels(shape, seed=seed, num_boxes=num_cubes)


def make_voxel_graph_split_labels(
    labels: np.ndarray,
    axis: int = 1,
    split_at: float = 0.5,
    block_boundaries: bool = True,
) -> np.ndarray:
    """
    Build a voxel_graph that encodes all label boundaries (optional) and
    additionally splits every label along a chosen axis at a fractional
    position within each labeled region.

    For 2D: axis=1 splits left/right (x); axis=0 splits top/bottom (y).
    """
    if labels.ndim < 2:
        raise ValueError("labels must be at least 2D")
    if axis < 0 or axis >= labels.ndim:
        raise ValueError("axis out of range")
    if not (0.0 < split_at < 1.0):
        raise ValueError("split_at must be in (0, 1)")

    bits = 2 * labels.ndim
    dtype = np.uint8 if bits <= 8 else np.uint16
    bitmask = (1 << bits) - 1
    graph = np.full(labels.shape, bitmask, dtype=dtype)

    # Bit layout: bit = 1 << (2*(ndim-1-axis) + sign), sign 0 => +, 1 => -
    def bit_pos(ax: int) -> int:
        return 1 << (2 * (labels.ndim - 1 - ax) + 0)

    def bit_neg(ax: int) -> int:
        return 1 << (2 * (labels.ndim - 1 - ax) + 1)

    if block_boundaries:
        # Block edges between different labels in both directions.
        for ax in range(labels.ndim):
            slicer_hi = [slice(None)] * labels.ndim
            slicer_lo = [slice(None)] * labels.ndim
            slicer_hi[ax] = slice(1, None)
            slicer_lo[ax] = slice(0, -1)
            diff = labels[tuple(slicer_hi)] != labels[tuple(slicer_lo)]
            # block +ax on lower voxel, -ax on upper voxel
            graph[tuple(slicer_lo)][diff] &= (~bit_pos(ax)) & (bitmask)
            graph[tuple(slicer_hi)][diff] &= (~bit_neg(ax)) & (bitmask)

            # Also handle edges: block connections from foreground to outside
            # Left edge: block -ax direction where label != 0
            slicer_edge = [slice(None)] * labels.ndim
            slicer_edge[ax] = 0
            edge_fg = labels[tuple(slicer_edge)] != 0
            graph[tuple(slicer_edge)][edge_fg] &= (~bit_neg(ax)) & (bitmask)

            # Right edge: block +ax direction where label != 0
            slicer_edge[ax] = labels.shape[ax] - 1
            edge_fg = labels[tuple(slicer_edge)] != 0
            graph[tuple(slicer_edge)][edge_fg] &= (~bit_pos(ax)) & (bitmask)

    # Split each label along the chosen axis at split_at of its extent.
    for lab in np.unique(labels):
        if lab == 0:
            continue
        coords = np.argwhere(labels == lab)
        if coords.size == 0:
            continue
        lo = coords[:, axis].min()
        hi = coords[:, axis].max() + 1
        split = lo + int(round((hi - lo) * split_at))
        if split <= lo or split >= hi:
            continue

        # Block edges crossing the split plane in both directions.
        slicer_left = [slice(None)] * labels.ndim
        slicer_right = [slice(None)] * labels.ndim
        slicer_left[axis] = split - 1
        slicer_right[axis] = split
        mask_left = labels[tuple(slicer_left)] == lab
        mask_right = labels[tuple(slicer_right)] == lab
        mask = mask_left & mask_right
        graph[tuple(slicer_left)][mask] &= (~bit_pos(axis)) & bitmask
        graph[tuple(slicer_right)][mask] &= (~bit_neg(axis)) & bitmask

    return graph

def test_edt_consistency():
    """Test that edt functions give consistent results across dimensions"""
    
    print("="*60)
    print("EDT CONSISTENCY TEST")
    print("="*60)
    
    import edt
    legacy = getattr(edt, "legacy", None)
    has_legacy = bool(legacy) and getattr(legacy, "available", lambda: False)()
    
    for ndim in [1, 2, 3]:
        print(f"\n--- Testing {ndim}D ---")
        
        # Test with M=3 (smaller for readability)
        M = 3
        masks = make_label_matrix(ndim, M)
        
        print(f"Input shape: {masks.shape}")
        print(f"Input size: {masks.size}")
        print(f"Unique labels: {np.unique(masks)}")
        
        if has_legacy:
            if ndim == 1:
                dt_orig = legacy.edt1d(masks)
            elif ndim == 2:
                dt_orig = legacy.edt2d(masks)
            else:
                dt_orig = legacy.edt3d(masks)

            print(f"Original edt{ndim}d: range {dt_orig.min():.3f} to {dt_orig.max():.3f}")
            print(f"Expected max: {M} (side length)")
            print(f"Max matches expected: {abs(dt_orig.max() - M) < 1e-6}")
        else:
            dt_orig = None
            print("Legacy edt.legacy module unavailable; skipping specialized comparison.")

        dt_nd = edt.edt_nd(masks)
        print(f"ND edt_nd: range {dt_nd.min():.3f} to {dt_nd.max():.3f}")
        print(f"Max matches expected: {abs(dt_nd.max() - M) < 1e-6}")

        if dt_orig is not None:
            diff = np.abs(dt_orig - dt_nd)
            max_diff = diff.max()
            print(f"Max difference: {max_diff:.6f}")

            if max_diff < 1e-6:
                print("Results match within tolerance.")
            else:
                print("Results differ beyond tolerance.")
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"Max difference at position {max_diff_idx}:")
                print(f"  Original: {dt_orig[max_diff_idx]:.3f}")
                print(f"  ND: {dt_nd[max_diff_idx]:.3f}")
                print(f"  Input value: {masks[max_diff_idx]}")

if __name__ == "__main__":
    test_edt_consistency()
