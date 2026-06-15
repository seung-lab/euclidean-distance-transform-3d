import os
import numpy as np

import edt
from debug_utils import make_label_matrix


def _maybe_plot(title, arrs, labels):
    if not os.environ.get("EDT_TEST_PLOTS"):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, len(arrs), figsize=(4 * len(arrs), 4))
    if len(arrs) == 1:
        axes = [axes]
    for ax, arr, name in zip(axes, arrs, labels):
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    out = f"/tmp/edt_voxel_graph_{title.replace(' ', '_')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _maybe_plot_grid(title, grid, labels, blocked_arrows=None):
    if not os.environ.get("EDT_TEST_PLOTS"):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        if cols == 1:
            axes = [axes]
        else:
            # keep 1D array of axes for consistent indexing
            axes = axes
    # Fixed label colormap for consistent comparisons.
    label_cmap = plt.cm.get_cmap("viridis", 4)
    dist_cmap = plt.cm.get_cmap("magma")
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            name = labels[r][c]
            arr = grid[r][c]
            if name == "blocked_arrows" and blocked_arrows is not None:
                ax.imshow(np.zeros_like(arr), interpolation="nearest", cmap="gray")
                if isinstance(blocked_arrows, (list, tuple)) and len(blocked_arrows) == rows:
                    X, Y, U, V = blocked_arrows[r]
                else:
                    X, Y, U, V = blocked_arrows
                ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, color="red")
            elif "labels" in name or "expanded" in name or name.startswith("sk_"):
                ax.imshow(arr, interpolation="nearest", cmap=label_cmap, vmin=0, vmax=3)
            elif "dist_" in name:
                ax.imshow(arr, interpolation="nearest", cmap=dist_cmap)
            else:
                ax.imshow(arr, interpolation="nearest")
            ax.set_title(name)
            ax.axis("off")
    fig.suptitle(title)
    out = f"/tmp/edt_voxel_graph_{title.replace(' ', '_')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _axis_bits(ndim):
    return tuple(1 << (2 * (ndim - 1 - axis) + 1) for axis in range(ndim))


def _random_graph(shape, bits, seed=0):
    rng = np.random.default_rng(seed)
    graph = np.zeros(shape, dtype=np.uint8)
    for bit in bits:
        graph |= (rng.random(shape) > 0.5).astype(np.uint8) * bit
    return graph


def test_voxel_graph_2d_full_connectivity_matches_standard():
    """With full connectivity, voxel_graph should match standard EDT."""
    arr = np.zeros((5, 6), dtype=np.uint32)
    arr[0, 0] = 1
    arr[2, 3] = 2
    arr[4, 5] = 3
    # Full connectivity graph
    bits = _axis_bits(arr.ndim)
    graph = np.zeros(arr.shape, dtype=np.uint8)
    for bit in bits:
        graph |= bit
    graph[arr == 0] = 0  # Only foreground has edges
    anis = (1.25, 0.75)

    with_graph = edt.edtsq(arr, anisotropy=anis, black_border=True, voxel_graph=graph)
    standard = edt.edtsq(arr, anisotropy=anis, black_border=True)

    np.testing.assert_allclose(with_graph, standard, rtol=1e-5, atol=1e-5)


def test_voxel_graph_3d_full_connectivity_matches_standard():
    """With full connectivity, voxel_graph should match standard EDT."""
    arr = np.zeros((4, 5, 3), dtype=np.uint32)
    arr[0, 0, 0] = 1
    arr[1, 3, 2] = 2
    arr[3, 4, 1] = 3
    # Full connectivity graph
    bits = _axis_bits(arr.ndim)
    graph = np.zeros(arr.shape, dtype=np.uint8)
    for bit in bits:
        graph |= bit
    graph[arr == 0] = 0
    anis = (1.0, 1.5, 0.5)

    with_graph = edt.edtsq(arr, anisotropy=anis, black_border=False, voxel_graph=graph)
    standard = edt.edtsq(arr, anisotropy=anis, black_border=False)

    np.testing.assert_allclose(with_graph, standard, rtol=1e-5, atol=1e-5)


def test_voxel_graph_4d_runs_and_shapes():
    arr = np.zeros((2, 3, 4, 2), dtype=np.uint32)
    arr[0, 0, 0, 0] = 1
    arr[1, 2, 3, 1] = 2
    # Need >= 8 bits (2 * (dims-1) + 1 = 7), uint8 is fine for 4D.
    graph = _random_graph(arr.shape, bits=_axis_bits(arr.ndim), seed=3)
    anis = (1.0, 1.0, 1.0, 1.0)

    nd = edt.edtsq(arr, anisotropy=anis, black_border=True, voxel_graph=graph)

    assert nd.shape == arr.shape
    assert nd.dtype == np.float32


def test_voxel_graph_quadrants_parity_and_effects():
    M = 8
    labels = make_label_matrix(2, M).astype(np.uint32)
    # Quadrant 0 is background.
    labels[:M, :M] = 0

    axis_bits = _axis_bits(labels.ndim)

    def make_graph(sym=True):
        g = np.zeros(labels.shape, dtype=np.uint8)
        g[:] = np.uint8(axis_bits[0] | axis_bits[1])
        # Block only the boundary faces between quadrants.
        # +axis0 (down) blocked on the horizontal boundary row (M-1).
        g[M - 1 : M, :] &= np.uint8(0xFF ^ axis_bits[0])
        # +axis1 (right) blocked on the vertical boundary column (M-1).
        g[:, M - 1 : M] &= np.uint8(0xFF ^ axis_bits[1])
        if sym:
            # Mirror blocking to make it bidirectional on the opposite side.
            g[M : M + 1, :] &= np.uint8(0xFF ^ axis_bits[0])
            g[:, M : M + 1] &= np.uint8(0xFF ^ axis_bits[1])
        return g

    def blocked_faces(g):
        return ((g & axis_bits[0]) == 0) | ((g & axis_bits[1]) == 0)

    def blocked_arrows(g, mirror):
        h, w = labels.shape
        yy, xx = np.mgrid[0:h, 0:w]
        u = np.zeros_like(labels, dtype=float)
        v = np.zeros_like(labels, dtype=float)
        # axis0 (+down) blocked -> arrow down
        mask_down = (g & axis_bits[0]) == 0
        v[mask_down] = 0.6
        if mirror:
            v[np.roll(mask_down, 1, axis=0)] = -0.6
        # axis1 (+right) blocked -> arrow right
        mask_right = (g & axis_bits[1]) == 0
        u[mask_right] = 0.6
        if mirror:
            u[np.roll(mask_right, 1, axis=1)] = -0.6
        return (xx, yy, u, v)

    graph_asym = make_graph(sym=False)
    graph_sym = make_graph(sym=True)

    def compute_row(g, black_border):
        nd = edt.edtsq(labels, voxel_graph=g, black_border=black_border)
        nd_plain = edt.edtsq(labels, black_border=black_border)
        # Verify the voxel_graph produced valid output
        assert nd.shape == labels.shape
        assert nd.dtype == np.float32
        assert np.all(nd >= 0)
        # Verify barriers have an effect (blocked regions differ from plain)
        # Only check foreground regions where blocking should matter
        fg_mask = labels != 0
        # At least some positions should differ due to blocking
        has_effect = not np.allclose(nd[fg_mask], nd_plain[fg_mask])
        return nd, nd_plain, has_effect

    nd_asym_bb, nd_plain_bb, eff1 = compute_row(graph_asym, True)
    nd_asym_open, nd_plain_open, eff2 = compute_row(graph_asym, False)
    nd_sym_bb, nd_plain_sym_bb, eff3 = compute_row(graph_sym, True)
    nd_sym_open, nd_plain_sym_open, eff4 = compute_row(graph_sym, False)

    # At least some configs should show barrier effects
    assert any([eff1, eff2, eff3, eff4]), "Barriers should affect at least some configurations"

    _maybe_plot_grid(
        "quadrants",
        [
            [labels, blocked_faces(graph_asym).astype(np.uint8), blocked_faces(graph_asym).astype(np.uint8),
             nd_asym_bb, nd_plain_bb],
            [labels, blocked_faces(graph_asym).astype(np.uint8), blocked_faces(graph_asym).astype(np.uint8),
             nd_asym_open, nd_plain_open],
            [labels, blocked_faces(graph_sym).astype(np.uint8), blocked_faces(graph_sym).astype(np.uint8),
             nd_sym_bb, nd_plain_sym_bb],
            [labels, blocked_faces(graph_sym).astype(np.uint8), blocked_faces(graph_sym).astype(np.uint8),
             nd_sym_open, nd_plain_sym_open],
        ],
        [
            ["labels", "blocked_faces", "blocked_arrows", "dist_graph_bb", "dist_plain_bb"],
            ["labels", "blocked_faces", "blocked_arrows", "dist_graph_open", "dist_plain_open"],
            ["labels", "blocked_faces", "blocked_arrows", "dist_graph_bb", "dist_plain_bb"],
            ["labels", "blocked_faces", "blocked_arrows", "dist_graph_open", "dist_plain_open"],
        ],
        blocked_arrows=[
            blocked_arrows(graph_asym, mirror=False),
            blocked_arrows(graph_asym, mirror=False),
            blocked_arrows(graph_sym, mirror=True),
            blocked_arrows(graph_sym, mirror=True),
        ],
    )


def test_expand_labels_vs_skimage_plot():
    M = 8
    labels = make_label_matrix(2, M).astype(np.uint32)
    labels[:M, :M] = 0

    expanded = edt.expand_labels(labels)

    sk = None
    try:
        from skimage.segmentation import expand_labels as sk_expand
        sk = sk_expand(labels, distance=np.inf)
    except Exception:
        pass

    if sk is not None:
        grid = [[labels, expanded, sk]]
        names = [["labels", "expanded", "sk_expand"]]
    else:
        grid = [[labels, expanded]]
        names = [["labels", "expanded"]]

    _maybe_plot_grid("expand_labels_vs_skimage", grid, names)


def test_voxel_graph_examples_png():
    if not os.environ.get("EDT_TEST_PLOTS"):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def axis_bits(ndim):
        return tuple(1 << (2 * (ndim - 1 - axis) + 1) for axis in range(ndim))

    def make_two_squares(shape, size=6, gap=4):
        h, w = shape
        arr = np.zeros(shape, dtype=np.uint32)
        y0 = h // 2 - size - gap // 2
        x0 = w // 2 - size - gap // 2
        arr[y0 : y0 + size, x0 : x0 + size] = 1
        y1 = h // 2
        x1 = w // 2
        arr[y1 : y1 + size, x1 : x1 + size] = 2
        return arr

    def block_outgoing(graph, mask, bits):
        # Block +axis edges for voxels in mask.
        for axis, bit in enumerate(bits):
            graph[mask] &= np.uint8(0xFF ^ bit)

    def block_incoming(graph, mask, bits):
        # Block +axis edges on neighbors just outside mask (reverse direction).
        for axis, bit in enumerate(bits):
            outside = np.zeros_like(mask, dtype=bool)
            outside[tuple(slice(None) for _ in range(mask.ndim))] = mask
            outside = np.roll(outside, -1, axis=axis)
            graph[outside] &= np.uint8(0xFF ^ bit)

    def expand_for_cases(labels, target_mask):
        bits = axis_bits(labels.ndim)
        base = np.zeros(labels.shape, dtype=np.uint8)
        base[:] = np.uint8(bits[0] | bits[1])

        graphs = {}
        graphs["none"] = base.copy()
        g_out = base.copy()
        block_outgoing(g_out, target_mask, bits)
        graphs["outgoing"] = g_out
        g_in = base.copy()
        block_incoming(g_in, target_mask, bits)
        graphs["incoming"] = g_in
        g_both = base.copy()
        block_outgoing(g_both, target_mask, bits)
        block_incoming(g_both, target_mask, bits)
        graphs["both"] = g_both

        dists = {}
        for name, g in graphs.items():
            dist_sq = edt.edtsq(labels, voxel_graph=g)
            dists[name] = np.sqrt(dist_sq, dtype=np.float32)
        return dists, graphs

    def block_midline(graph, axis, idx, bits):
        bit = bits[axis]
        slc = [slice(None)] * graph.ndim
        slc[axis] = slice(idx, idx + 1)
        graph[tuple(slc)] &= np.uint8(0xFF ^ bit)
        # mirror to make it bidirectional
        slc[axis] = slice(idx + 1, idx + 2)
        graph[tuple(slc)] &= np.uint8(0xFF ^ bit)

    label_cmap = plt.cm.get_cmap("viridis", 4)
    dist_cmap = plt.cm.get_cmap("magma")

    # Example 1: two squares, expand labels under different blocking.
    labels = make_two_squares((48, 48), size=8, gap=4)
    target = labels == 1
    dists, graphs = expand_for_cases(labels, target)

    # Example 2: single rectangle with midline blocked both directions.
    rect = np.zeros((48, 48), dtype=np.uint32)
    rect[16:32, 12:36] = 1
    bits = axis_bits(rect.ndim)
    graph_rect = np.zeros(rect.shape, dtype=np.uint8)
    graph_rect[:] = np.uint8(bits[0] | bits[1])
    block_midline(graph_rect, axis=0, idx=24, bits=bits)
    dist_rect = edt.edt(rect, voxel_graph=graph_rect)
    dist_rect_legacy = edt.legacy.edt(rect, voxel_graph=graph_rect)

    # Plot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ["none", "outgoing", "incoming", "both"]
    for c, name in enumerate(titles):
        ax = axes[0, c]
        ax.imshow(dists[name], cmap=dist_cmap, interpolation="nearest")
        ax.set_title(f"dist_{name}")
        ax.axis("off")

    axes[2, 0].imshow(labels, cmap=label_cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[2, 0].set_title("labels")
    axes[2, 0].axis("off")
    for c, name in enumerate(titles[1:], start=1):
        axes[2, c].imshow(graphs[name], interpolation="nearest")
        axes[2, c].set_title(f"graph_{name}")
        axes[2, c].axis("off")

    axes[3, 0].imshow(rect, cmap=label_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[3, 0].set_title("rect_labels")
    axes[3, 0].axis("off")
    axes[3, 1].imshow(graph_rect, interpolation="nearest")
    axes[3, 1].set_title("rect_graph")
    axes[3, 1].axis("off")
    axes[3, 2].imshow(dist_rect, cmap=dist_cmap, interpolation="nearest")
    axes[3, 2].set_title("rect_dist_nd")
    axes[3, 2].axis("off")
    axes[3, 3].imshow(dist_rect_legacy, cmap=dist_cmap, interpolation="nearest")
    axes[3, 3].set_title("rect_dist_legacy")
    axes[3, 3].axis("off")

    fig.suptitle("voxel_graph_examples")
    out = "/tmp/edt_voxel_graph_examples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
