import numpy as np

import edt


def _bruteforce_nearest(shape, seeds, anisotropy):
    """Return linear index of nearest seed per voxel (C-order).

    Tie-breaks by choosing the larger seed index to match expand_labels 1D.
    """
    anis = np.asarray(anisotropy, dtype=np.float64)
    coords = np.indices(shape, dtype=np.float64).reshape(len(shape), -1).T
    seed_coords = np.array([s[0] for s in seeds], dtype=np.float64)
    seed_lin = np.array(
        [np.ravel_multi_index(tuple(s[0]), shape, order="C") for s in seeds],
        dtype=np.int64,
    )
    # Compute squared distances with anisotropy scaling.
    diffs = coords[:, None, :] - seed_coords[None, :, :]
    diffs *= anis[None, None, :]
    d2 = np.sum(diffs * diffs, axis=2)
    # Tie-break toward the larger seed index.
    nearest = np.empty((d2.shape[0],), dtype=np.int64)
    for i in range(d2.shape[0]):
        row = d2[i]
        m = np.min(row)
        candidates = np.flatnonzero(row == m)
        if candidates.size == 1:
            nearest[i] = candidates[0]
        else:
            # Choose the largest linear index among tied seeds.
            best = candidates[np.argmax(seed_lin[candidates])]
            nearest[i] = best
    return seed_lin[nearest].reshape(shape)


def test_feature_transform_matches_bruteforce_2d():
    shape = (7, 9)
    arr = np.zeros(shape, dtype=np.uint32)
    seeds = [((1, 1), 10), ((5, 6), 20)]
    for coord, label in seeds:
        arr[coord] = label
    anis = (1.0, 2.0)

    feats = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    expected = _bruteforce_nearest(shape, seeds, anis)
    np.testing.assert_array_equal(feats, expected)
    np.testing.assert_array_equal(arr.ravel()[feats], arr.ravel()[expected])


def test_feature_transform_return_distances_matches_edtsq_nd():
    shape = (6, 6, 4)
    arr = np.zeros(shape, dtype=np.uint32)
    arr[1, 1, 1] = 5
    arr[4, 2, 3] = 7
    anis = (1.0, 1.5, 2.0)

    feats, dist = edt.feature_transform(
        arr, anisotropy=anis, parallel=1, return_distances=True
    )
    ref = edt.edtsq((arr != 0).astype(np.uint8), anisotropy=anis, parallel=1)
    np.testing.assert_allclose(dist, ref, rtol=1e-6, atol=1e-6)
    assert feats.shape == arr.shape


def test_expand_labels_return_features_consistent_with_feature_transform():
    shape = (5, 8)
    arr = np.zeros(shape, dtype=np.uint32)
    arr[0, 0] = 1
    arr[4, 7] = 2
    anis = (1.0, 1.0)

    labels, feats = edt.expand_labels(
        arr, anisotropy=anis, parallel=1, return_features=True
    )
    ft = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    np.testing.assert_array_equal(feats, ft)
    np.testing.assert_array_equal(labels, arr.ravel()[feats].reshape(shape))



def test_feature_transform_anisotropy_length_mismatch_raises():
    arr = np.zeros((4, 4), dtype=np.uint32)
    with np.testing.assert_raises(ValueError):
        edt.feature_transform(arr, anisotropy=(1.0, 2.0, 3.0))


def test_feature_transform_1d_matches_bruteforce():
    arr = np.zeros((8,), dtype=np.uint32)
    arr[2] = 3
    arr[6] = 5
    anis = (1.0,)

    feats = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    seeds = [((2,), 3), ((6,), 5)]
    expected = _bruteforce_nearest(arr.shape, seeds, anis)
    np.testing.assert_array_equal(feats, expected)


# ---------------------------------------------------------------------------
# black_border tests for expand_labels
# ---------------------------------------------------------------------------

def test_expand_labels_black_border_1d():
    """1D: voxels closer to border than any seed get label 0."""
    labels = np.array([0, 0, 1, 0, 0, 0, 2, 0, 0], dtype=np.uint32)
    result = edt.expand_labels(labels, black_border=True)
    # Position 0: border_dist=1, seed_dist=2 (seed at 2) → border wins → 0
    # Position 8: border_dist=1, seed_dist=2 (seed at 6) → border wins → 0
    assert result[0] == 0, f"Expected 0 at border, got {result[0]}"
    assert result[8] == 0, f"Expected 0 at border, got {result[8]}"
    # Interior: same as no-border
    result_no = edt.expand_labels(labels, black_border=False)
    np.testing.assert_array_equal(result[1:-1], result_no[1:-1])


def test_expand_labels_black_border_no_effect_when_seeds_at_border():
    """1D: seed at position 0 — border and seed coincide, border should not zero it out."""
    labels = np.array([1, 0, 0, 0, 2], dtype=np.uint32)
    result = edt.expand_labels(labels, black_border=True)
    # Position 0: seed IS at border, seed_dist=0 < border_dist=1 → seed wins → 1
    assert result[0] == 1, f"Expected 1 (seed at border), got {result[0]}"
    # Position 4: seed IS at border, seed_dist=0 < border_dist=1 → seed wins → 2
    assert result[4] == 2, f"Expected 2 (seed at border), got {result[4]}"


def test_expand_labels_black_border_2d_corners():
    """2D: border voxels far from any seed get label 0 with black_border."""
    labels = np.zeros((9, 9), dtype=np.uint32)
    labels[4, 4] = 1  # single central seed
    result_bb = edt.expand_labels(labels, black_border=True)
    result_no = edt.expand_labels(labels, black_border=False)
    # Corners are farther from center (distance ~5.6) than from border (distance 1)
    assert result_bb[0, 0] == 0, "Corner should be 0 with black_border"
    assert result_bb[0, 8] == 0, "Corner should be 0 with black_border"
    # Without black_border, everything is 1
    assert np.all(result_no == 1), "Without black_border, all voxels should expand"


