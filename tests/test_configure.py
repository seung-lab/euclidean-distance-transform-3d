"""Tests for edt.configure() programmatic override of environment variables."""

import os
import pytest
import numpy as np
import edt


@pytest.fixture(autouse=True)
def reset_config():
    """Clear _ND_CONFIG before and after each test."""
    edt._ND_CONFIG.clear()
    yield
    edt._ND_CONFIG.clear()


def test_configure_sets_adaptive_threads():
    edt.configure(adaptive_threads=False)
    assert edt._ND_CONFIG['EDT_ADAPTIVE_THREADS'] == 0

    edt.configure(adaptive_threads=True)
    assert edt._ND_CONFIG['EDT_ADAPTIVE_THREADS'] == 1


def test_configure_sets_min_voxels():
    edt.configure(min_voxels_per_thread=1000)
    assert edt._ND_CONFIG['EDT_ND_MIN_VOXELS_PER_THREAD'] == 1000


def test_configure_sets_min_lines():
    edt.configure(min_lines_per_thread=4)
    assert edt._ND_CONFIG['EDT_ND_MIN_LINES_PER_THREAD'] == 4


def test_configure_overrides_env_var(monkeypatch):
    """configure() must take priority over the environment variable."""
    monkeypatch.setenv('EDT_ADAPTIVE_THREADS', '0')
    edt.configure(adaptive_threads=True)
    # _env_int should return the in-process value, not the env var
    assert edt._env_int('EDT_ADAPTIVE_THREADS', 1) == 1


def test_env_var_used_when_no_configure(monkeypatch):
    """Without configure(), _env_int falls back to the env var."""
    monkeypatch.setenv('EDT_ND_MIN_VOXELS_PER_THREAD', '99')
    assert edt._env_int('EDT_ND_MIN_VOXELS_PER_THREAD', 50000) == 99


def test_configure_none_args_are_no_ops():
    """Passing None should not touch _ND_CONFIG."""
    edt.configure(adaptive_threads=None, min_voxels_per_thread=None)
    assert 'EDT_ADAPTIVE_THREADS' not in edt._ND_CONFIG
    assert 'EDT_ND_MIN_VOXELS_PER_THREAD' not in edt._ND_CONFIG


def test_configure_affects_thread_limiting():
    """With very permissive thresholds, more threads should be used on a small ND array."""
    labels = np.ones((4, 4, 4, 4), dtype=np.uint8)

    # Default heuristics may cap threads; with min=1 they should not
    edt.configure(min_voxels_per_thread=1, min_lines_per_thread=1)
    # Just confirm it runs without error and returns correct shape
    result = edt.edtsq(labels, parallel=4)
    assert result.shape == labels.shape
