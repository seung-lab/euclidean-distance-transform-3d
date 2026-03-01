"""Tests for Fortran-order array support in edt."""

import numpy as np
import pytest
import edt


@pytest.fixture
def labels_3d():
    rng = np.random.default_rng(42)
    return (rng.random((20, 30, 40)) > 0.1).astype(np.uint8)


def test_fortran_matches_c_order_2d():
    labels = np.ones((32, 48), dtype=np.uint8)
    labels[10:20, 10:20] = 0
    labels_f = np.asfortranarray(labels)
    assert labels_f.flags.f_contiguous

    result_c = edt.edt(labels)
    result_f = edt.edt(labels_f)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_matches_c_order_3d(labels_3d):
    labels_f = np.asfortranarray(labels_3d)
    assert labels_f.flags.f_contiguous

    result_c = edt.edt(labels_3d)
    result_f = edt.edt(labels_f)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_output_is_fortran_contiguous(labels_3d):
    labels_f = np.asfortranarray(labels_3d)
    result = edt.edt(labels_f)
    assert result.flags.f_contiguous, "Output should be F-contiguous for F-contiguous input"
    assert result.shape == labels_3d.shape


def test_fortran_no_copy_for_correct_dtype():
    """F-contiguous uint8 input should not be copied (output shares no buffer with input,
    but the *input* itself should not be needlessly copied to C-order)."""
    labels = np.asfortranarray(np.ones((20, 20), dtype=np.uint8))
    labels[5:15, 5:15] = 0
    arr, is_f = edt._prepare_array(labels, np.uint8)
    assert is_f, "_prepare_array should detect F-contiguous input"
    assert arr.flags.f_contiguous, "_prepare_array should preserve F-order"


def test_fortran_with_anisotropy(labels_3d):
    anis = (2.0, 1.0, 0.5)
    labels_f = np.asfortranarray(labels_3d)

    result_c = edt.edt(labels_3d, anisotropy=anis)
    result_f = edt.edt(labels_f, anisotropy=anis)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_1d_unaffected_by_fortran_path():
    """1D arrays are both C and F contiguous — should take C path."""
    labels = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
    assert labels.flags.c_contiguous and labels.flags.f_contiguous
    arr, is_f = edt._prepare_array(labels, np.uint8)
    assert not is_f, "1D array should take C path even though f_contiguous is True"


def test_non_contiguous_falls_back_to_c():
    labels = np.ones((20, 20), dtype=np.uint8)
    sliced = labels[::2, ::2]  # Non-contiguous slice
    assert not sliced.flags.c_contiguous and not sliced.flags.f_contiguous
    arr, is_f = edt._prepare_array(sliced, np.uint8)
    assert not is_f
    assert arr.flags.c_contiguous


# ---------------------------------------------------------------------------
# voxel_graph + Fortran-order
# ---------------------------------------------------------------------------

def test_fortran_voxel_graph_matches_c_order_2d():
    """F-contiguous voxel_graph gives same result as C-contiguous (forced to C inside edtsq)."""
    labels = np.ones((24, 36), dtype=np.uint8)
    labels[8:16, 8:16] = 0
    vg = np.ones_like(labels) * 0x3F  # all connectivity open

    result_c = edt.edt(labels, voxel_graph=vg)
    result_f = edt.edt(labels, voxel_graph=np.asfortranarray(vg))
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_voxel_graph_matches_c_order_3d(labels_3d):
    """F-contiguous voxel_graph with 3D labels."""
    vg = np.ones(labels_3d.shape, dtype=np.uint8) * 0x3F

    result_c = edt.edt(labels_3d, voxel_graph=vg)
    result_f = edt.edt(labels_3d, voxel_graph=np.asfortranarray(vg))
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_labels_with_voxel_graph_2d():
    """F-contiguous labels with C-order voxel_graph."""
    labels = np.ones((24, 36), dtype=np.uint8)
    labels[8:16, 8:16] = 0
    vg = np.ones_like(labels) * 0x3F

    result_c = edt.edt(labels, voxel_graph=vg)
    result_f = edt.edt(np.asfortranarray(labels), voxel_graph=vg)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


# ---------------------------------------------------------------------------
# expand_labels + Fortran-order
# ---------------------------------------------------------------------------

def test_expand_labels_fortran_matches_c_2d():
    """F-contiguous input to expand_labels gives same result as C-contiguous."""
    rng = np.random.default_rng(7)
    labels = np.zeros((30, 40), dtype=np.uint32)
    labels[5, 5] = 1
    labels[20, 30] = 2
    labels[10, 25] = 3

    result_c = edt.expand_labels(labels)
    result_f = edt.expand_labels(np.asfortranarray(labels))
    np.testing.assert_array_equal(result_c, result_f)


def test_expand_labels_fortran_matches_c_3d():
    """F-contiguous 3D input to expand_labels gives same result as C-contiguous."""
    labels = np.zeros((15, 20, 25), dtype=np.uint32)
    labels[2, 3, 4] = 1
    labels[10, 15, 20] = 2
    labels[7, 10, 12] = 3

    result_c = edt.expand_labels(labels)
    result_f = edt.expand_labels(np.asfortranarray(labels))
    np.testing.assert_array_equal(result_c, result_f)


def test_expand_labels_fortran_output_is_fortran():
    """F-contiguous input should produce F-contiguous output."""
    labels = np.zeros((20, 30), dtype=np.uint32)
    labels[5, 5] = 1
    labels[15, 25] = 2

    result = edt.expand_labels(np.asfortranarray(labels))
    assert result.flags.f_contiguous, "F-contiguous input should yield F-contiguous output"
    assert result.shape == labels.shape


def test_expand_labels_fortran_with_anisotropy():
    """F-contiguous input with anisotropy gives same result as C-contiguous."""
    labels = np.zeros((20, 30), dtype=np.uint32)
    labels[5, 5] = 1
    labels[15, 25] = 2

    anis = (2.0, 0.5)
    result_c = edt.expand_labels(labels, anisotropy=anis)
    result_f = edt.expand_labels(np.asfortranarray(labels), anisotropy=anis)
    np.testing.assert_array_equal(result_c, result_f)


def test_expand_labels_fortran_return_features():
    """F-contiguous input with return_features=True gives same result."""
    labels = np.zeros((20, 30), dtype=np.uint32)
    labels[5, 5] = 1
    labels[15, 25] = 2

    lbl_c, feat_c = edt.expand_labels(labels, return_features=True)
    lbl_f, feat_f = edt.expand_labels(np.asfortranarray(labels), return_features=True)
    np.testing.assert_array_equal(lbl_c, lbl_f)
    np.testing.assert_array_equal(feat_c, feat_f)
    assert lbl_f.flags.f_contiguous
    assert feat_f.flags.f_contiguous
