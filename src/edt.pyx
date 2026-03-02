# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Multi-label Euclidean Distance Transform based on the algorithms of
Saito et al (1994), Meijster et al (2002), and Felzenszwalb & Huttenlocher (2012).

Uses connectivity graphs internally (uint8 for 1-4D, uint16 for 5-8D, uint32 for 9-16D, uint64 for 17-32D).
Memory-efficient for larger input dtypes (up to 38% savings for uint32 input
vs label-segment approaches).
Supports custom voxel_graph input for user-defined boundaries.

Key methods:
  edt, edtsq - main EDT functions
  edt_graph, edtsq_graph - EDT from pre-built connectivity graph
  build_graph - build connectivity graph from labels

Additional utilities:
  feature_transform, expand_labels, sdf, each

Programmatic configuration:
  edt.configure(...) - set threading parameters in-process (see configure docstring)

Environment Variables (runtime):
  EDT_ADAPTIVE_THREADS         - 0/1, enable adaptive thread limiting by array size (default: 1)
  EDT_ND_MIN_VOXELS_PER_THREAD - min voxels per thread (default: 4000)
  EDT_ND_MIN_LINES_PER_THREAD  - min scanlines per thread (default: 32)
  EDT_ND_PROFILE               - if set, record shape/thread info in edt._nd_profile_last (default: off)

Environment Variables (build-time):
  EDT_MARCH_NATIVE          - 0/1, compile with -march=native (default: 1)

License: GNU 3.0

Original EDT: William Silversmith (Seung Lab, Princeton),  August 2018 - February 2026
ND connectivity graph EDT: Kevin Cutler, February 2026
"""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as native_bool
cimport numpy as np
np.import_array()

import numpy as np
import multiprocessing
import os

# Profile storage for last edtsq/edtsq_graph call
_nd_profile_last = None

# Thread limiting: cap threads so each gets at least this much work.
# Both criteria are computed; whichever allows FEWER threads wins (both must hold).
_ND_MIN_VOXELS_PER_THREAD_DEFAULT = 4000
_ND_MIN_LINES_PER_THREAD_DEFAULT = 32

# In-process overrides set via configure(), take priority over env vars
_ND_CONFIG = {}


def _check_dims(nd):
    if nd == 0:
        raise ValueError("EDT requires at least 1 dimension (got a 0-dimensional scalar).")
    if nd > 32:
        raise ValueError(f"EDT supports at most 32 dimensions, got {nd}.")


def _graph_dtype(ndim):
    """Return the minimal uint dtype for a connectivity graph of ndim dimensions.

    Bit 0 is the foreground marker. Each axis edge occupies bit 2*(ndim-1-axis)+1,
    so max bit = 2*(ndim-1)+1:
      dims 1-4   -> uint8  (max bit 7)
      dims 5-8   -> uint16 (max bit 15)
      dims 9-16  -> uint32 (max bit 31)
      dims 17-32 -> uint64 (max bit 63)
    """
    _check_dims(ndim)
    if ndim <= 4:  return np.uint8
    if ndim <= 8:  return np.uint16
    if ndim <= 16: return np.uint32
    return np.uint64


def _prepare_array(arr, dtype):
    """Return (contiguous_array, is_fortran).

    Preserves F-contiguous layout to avoid an unnecessary copy.
    Checks C-contiguous first so arrays that satisfy both (e.g. 1D or
    size-1 dimensions) take the cheaper C path.
    """
    if arr.flags.c_contiguous:
        return np.ascontiguousarray(arr, dtype=dtype), False
    if arr.flags.f_contiguous:
        return np.asfortranarray(arr, dtype=dtype), True
    # Non-contiguous: force C-order copy
    return np.ascontiguousarray(arr, dtype=dtype), False


def _resolve_label_dtype(arr):
    """Map a label array's dtype to the uint dtype used internally.

    bool -> uint8; signed/float -> same-width uint; already-uint -> unchanged.
    Returned dtype is always one of uint8/uint16/uint32/uint64.
    """
    dtype = arr.dtype
    if dtype == np.bool_:
        return np.uint8
    if dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        return dtype
    unsigned_map = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    return unsigned_map.get(dtype.itemsize, np.uint32)


def _normalize_anisotropy(anisotropy, nd):
    """Return anisotropy as a float tuple of length nd.

    None -> isotropic (1.0,)*nd; scalar -> replicated; sequence -> validated.
    """
    if anisotropy is None:
        return (1.0,) * nd
    if hasattr(anisotropy, '__len__'):
        anis = tuple(float(a) for a in anisotropy)
    else:
        anis = (float(anisotropy),) * nd
    if len(anis) != nd:
        raise ValueError(f"anisotropy must have {nd} elements, got {len(anis)}")
    return anis


def _resolve_parallel(parallel):
    """Cap parallel thread count to cpu_count; 0 or negative means use all CPUs."""
    if parallel <= 0:
        return multiprocessing.cpu_count()
    return max(1, min(parallel, multiprocessing.cpu_count()))


cdef extern from "edt.hpp" namespace "nd":
    # Tuning
    cdef void _nd_set_tuning "nd::set_tuning"(size_t chunks_per_thread) nogil

    # EDT from voxel graph
    cdef void edtsq_from_graph[GRAPH_T](
        const GRAPH_T* graph,
        float* output,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    # Build connectivity graph from labels
    cdef void build_connectivity_graph[T, GRAPH_T](
        const T* labels,
        GRAPH_T* graph,
        const size_t* shape,
        size_t dims,
        int parallel
    ) nogil

    # Fused: build graph internally then run EDT (more efficient)
    cdef void edtsq_from_labels_fused[T](
        const T* labels,
        float* output,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    # Fused expand_labels (orchestration in C++)
    cdef void expand_labels_fused[T](
        const T* data,
        uint32_t* labels_out,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    cdef void expand_labels_features_fused[T, INDEX](
        const T* data,
        uint32_t* labels_out,
        INDEX* features_out,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil


def set_tuning(chunks_per_thread=1):
    """Set tuning parameters for ND EDT."""
    _nd_set_tuning(chunks_per_thread)


def _voxel_graph_to_nd(voxel_graph, labels=None):
    """
    Convert bidirectional voxel_graph to ND graph format.

    The voxel_graph format uses 2*ndim bits per voxel:
    - positive direction at bit (2*(ndim-1-axis))
    - negative direction at bit (2*(ndim-1-axis)+1)

    The ND format uses forward edges only + foreground marker:
    - Forward edge for axis a at bit (2*(ndim-1-a)+1)
    - Bit 0 (0b00000001) marks foreground

    Positive direction bits are shifted left by 1 to make room for
    the foreground marker at bit 0, then the marker is added.

    If labels is None, foreground is inferred from voxel_graph != 0
    (any voxel with connectivity is foreground).
    """
    ndim = voxel_graph.ndim
    _check_dims(ndim)
    if labels is not None and voxel_graph.shape != labels.shape:
        raise ValueError("voxel_graph shape must match labels")

    # Validate input dtype has enough bits for this dimensionality.
    # voxel_graph format uses 2*ndim bits (positive + negative per axis).
    min_bits = 2 * ndim
    actual_bits = voxel_graph.dtype.itemsize * 8
    if actual_bits < min_bits:
        raise ValueError(
            f"voxel_graph dtype {voxel_graph.dtype} has {actual_bits} bits, "
            f"but {ndim}D requires at least {min_bits} bits"
        )

    # Build mask for positive direction bits only (even bits 0, 2, ..., 2*(ndim-1))
    pos_mask = sum(1 << (2 * i) for i in range(ndim))

    # Extract positive direction bits and shift left by 1 to make room for FG at bit 0
    # Use minimal dtype based on ndim (not input dtype) to avoid large intermediates
    mask_dtype = _graph_dtype(ndim)
    graph = (voxel_graph.astype(mask_dtype, copy=False) & mask_dtype(pos_mask)) << 1

    # Add foreground marker at bit 0 - infer from voxel_graph if no labels provided
    if labels is not None:
        graph[labels != 0] |= 0b00000001
    else:
        graph[voxel_graph != 0] |= 0b00000001

    return graph


def edtsq(labels=None, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute squared Euclidean distance transform via graph-first architecture.

    Builds a connectivity graph internally (uint8 for 1-4D, uint16 for 5-8D,
    uint32 for 9-16D, uint64 for 17-32D) then computes EDT. Graph is built and
    freed in C++ — no Python-visible intermediate allocation.

    Parameters
    ----------
    labels : ndarray or None
        Input label array. Non-zero values are foreground.
        Can be None if voxel_graph is provided (foreground inferred from connectivity).
    anisotropy : tuple or None
        Physical voxel size for each dimension. Default is isotropic (1, 1, ...).
    black_border : bool
        Treat image boundary as an object boundary.
    parallel : int
        Number of threads. 0 means auto-detect.
    voxel_graph : ndarray, optional
        Per-voxel bitfield describing allowed connections. Positive direction
        bits are extracted and used for EDT computation. If labels is None,
        foreground is inferred from voxel_graph != 0.
    order : ignored
        For backwards compatibility.

    Returns
    -------
    ndarray
        Squared Euclidean distance transform (float32).
    """
    # Handle voxel_graph input by converting to ND graph format
    if voxel_graph is not None:
        voxel_graph = np.ascontiguousarray(voxel_graph)
        if labels is not None:
            labels = np.asarray(labels)
        graph = _voxel_graph_to_nd(voxel_graph, labels)
        return edtsq_graph(graph, anisotropy, black_border, parallel)

    if labels is None:
        raise ValueError("labels is required when voxel_graph is not provided")

    # Preserve input dtype where possible to avoid copies.
    # For signed/float types, use .view() to reinterpret as same-width
    # unsigned — zero-copy, and equality semantics are identical.
    labels = np.asarray(labels)
    _check_dims(labels.ndim)
    dtype = _resolve_label_dtype(labels)
    labels, is_fortran = _prepare_array(labels, labels.dtype)
    if labels.dtype != dtype:
        labels = labels.view(dtype)
    cdef int nd = labels.ndim
    cdef tuple shape = labels.shape

    anisotropy = _normalize_anisotropy(anisotropy, nd)

    parallel_requested = parallel
    parallel = _resolve_parallel(parallel)
    parallel = _adaptive_thread_limit_nd(parallel, shape)

    # For F-contiguous arrays, reverse shape and anisotropy so C++ sees a
    # C-order array of reversed shape — same memory, no copy.
    cpp_shape = shape[::-1] if is_fortran else shape
    cpp_anis  = anisotropy[::-1] if is_fortran else anisotropy

    if os.environ.get('EDT_ND_PROFILE'):
        global _nd_profile_last
        _nd_profile_last = {
            'shape': shape,
            'dims': nd,
            'parallel_requested': parallel_requested,
            'parallel_used': parallel,
        }

    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    cdef float* canis = <float*> malloc(nd * sizeof(float))
    if cshape == NULL or canis == NULL:
        if cshape != NULL:
            free(cshape)
        if canis != NULL:
            free(canis)
        raise MemoryError('Allocation failure')

    cdef int i
    for i in range(nd):
        cshape[i] = <size_t>cpp_shape[i]
        canis[i] = <float>cpp_anis[i]

    cdef np.ndarray output = np.empty(cpp_shape, dtype=np.float32)
    cdef float* outp = <float*> np.PyArray_DATA(output)
    cdef native_bool bb = black_border
    cdef int par = parallel

    # Dispatch based on dtype to avoid unnecessary copies
    cdef int dtype_code = 0  # 0=uint8, 1=uint16, 2=uint32, 3=uint64
    if dtype == np.uint16:
        dtype_code = 1
    elif dtype == np.uint32:
        dtype_code = 2
    elif dtype == np.uint64:
        dtype_code = 3

    cdef uint8_t* labelsp8
    cdef uint16_t* labelsp16
    cdef uint32_t* labelsp32
    cdef uint64_t* labelsp64

    try:
        if dtype_code == 0:
            labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint8_t](labelsp8, outp, cshape, canis, nd, bb, par)
        elif dtype_code == 1:
            labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint16_t](labelsp16, outp, cshape, canis, nd, bb, par)
        elif dtype_code == 2:
            labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint32_t](labelsp32, outp, cshape, canis, nd, bb, par)
        else:  # uint64
            labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint64_t](labelsp64, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    if is_fortran:
        return output.T
    return output


def edt(labels=None, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute Euclidean distance transform.

    Returns the square root of edtsq.
    """
    dt = edtsq(labels, anisotropy, black_border, parallel, voxel_graph, order)
    return np.sqrt(dt, out=dt)


def edtsq_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute squared EDT from a voxel connectivity graph.

    Parameters
    ----------
    graph : ndarray (uint8 for 1D-4D, uint16 for 5D-8D, uint32 for 9D-16D, uint64 for 17D-32D)
        Voxel connectivity graph. Each element encodes edge bits for each axis.
        For 2D: axis 0 -> bit 3, axis 1 -> bit 1
        For 3D: axis 0 -> bit 5, axis 1 -> bit 3, axis 2 -> bit 1
    anisotropy : tuple or None
        Physical voxel size for each dimension.
    black_border : bool
        Treat image boundary as an object boundary.
    parallel : int
        Number of threads.

    Returns
    -------
    ndarray
        Squared Euclidean distance transform (float32).
    """
    cdef int nd = graph.ndim
    cdef tuple shape = graph.shape
    _check_dims(nd)

    graph_dtype = _graph_dtype(nd)
    # Connectivity graphs encode direction-specific edge bits per axis.
    # General formula: axis a -> bit (2*(ndim-1-a)+1); bit 0 = foreground.
    # For 2D: axis 0 -> bit 3, axis 1 -> bit 1.
    # The axis-reversal trick used for label arrays cannot be applied here: reversing the shape
    # would cause C++ to read axis-0 bits for the axis-1 sweep and vice versa, corrupting
    # direction-specific connectivity. Always copy to C-order to ensure correct bit interpretation.
    graph = np.ascontiguousarray(graph, dtype=graph_dtype)

    anisotropy = _normalize_anisotropy(anisotropy, nd)

    parallel_requested = parallel
    parallel = _resolve_parallel(parallel)
    parallel = _adaptive_thread_limit_nd(parallel, shape)

    if os.environ.get('EDT_ND_PROFILE'):
        global _nd_profile_last
        _nd_profile_last = {
            'shape': shape,
            'dims': nd,
            'parallel_requested': parallel_requested,
            'parallel_used': parallel,
        }

    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    cdef float* canis = <float*> malloc(nd * sizeof(float))
    if cshape == NULL or canis == NULL:
        if cshape != NULL:
            free(cshape)
        if canis != NULL:
            free(canis)
        raise MemoryError('Allocation failure')

    cdef int i
    for i in range(nd):
        cshape[i] = <size_t>shape[i]
        canis[i] = <float>anisotropy[i]

    cdef np.ndarray output = np.empty(shape, dtype=np.float32)
    cdef float* outp = <float*> np.PyArray_DATA(output)

    cdef native_bool bb = black_border
    cdef int par = parallel

    # Get graph pointer before nogil (dispatch based on dtype)
    cdef void* graphp = np.PyArray_DATA(graph)

    try:
        if nd <= 4:
            with nogil:
                edtsq_from_graph[uint8_t](<uint8_t*>graphp, outp, cshape, canis, nd, bb, par)
        elif nd <= 8:
            with nogil:
                edtsq_from_graph[uint16_t](<uint16_t*>graphp, outp, cshape, canis, nd, bb, par)
        elif nd <= 16:
            with nogil:
                edtsq_from_graph[uint32_t](<uint32_t*>graphp, outp, cshape, canis, nd, bb, par)
        else:
            with nogil:
                edtsq_from_graph[uint64_t](<uint64_t*>graphp, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    return output


def edt_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute EDT from a voxel connectivity graph.

    Returns the square root of edtsq_graph.
    """
    dt = edtsq_graph(graph, anisotropy, black_border, parallel)
    return np.sqrt(dt, out=dt)


def build_graph(labels, parallel=0):
    """
    Build a connectivity graph from labels.

    Parameters
    ----------
    labels : ndarray
        Input label array.
    parallel : int
        Number of threads.

    Returns
    -------
    ndarray
        Connectivity graph (uint8 for 1D-4D, uint16 for 5D-8D, uint32 for 9D-16D, uint64 for 17D-32D)
        where each element encodes per-axis edge bits.
    """
    # Preserve input dtype where possible to avoid copies.
    # For signed/float types, use .view() to reinterpret as same-width
    # unsigned — zero-copy, and equality semantics are identical.
    labels = np.asarray(labels)
    _check_dims(labels.ndim)
    dtype = _resolve_label_dtype(labels)
    labels = np.ascontiguousarray(labels)
    if labels.dtype != dtype:
        labels = labels.view(dtype)
    cdef int nd = labels.ndim
    cdef tuple shape = labels.shape

    parallel = _resolve_parallel(parallel)

    graph_dtype = _graph_dtype(nd)

    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    if cshape == NULL:
        raise MemoryError('Allocation failure')

    cdef int i
    for i in range(nd):
        cshape[i] = <size_t>shape[i]

    cdef np.ndarray graph = np.zeros(shape, dtype=graph_dtype)
    cdef int par = parallel

    # Dispatch based on label dtype
    cdef int dtype_code = 0  # 0=uint8, 1=uint16, 2=uint32, 3=uint64
    if dtype == np.uint16:
        dtype_code = 1
    elif dtype == np.uint32:
        dtype_code = 2
    elif dtype == np.uint64:
        dtype_code = 3

    cdef uint8_t* labelsp8
    cdef uint16_t* labelsp16
    cdef uint32_t* labelsp32
    cdef uint64_t* labelsp64
    cdef uint8_t* graphp8
    cdef uint16_t* graphp16
    cdef uint32_t* graphp32
    cdef uint64_t* graphp64

    try:
        if nd <= 4:
            graphp8 = <uint8_t*> np.PyArray_DATA(graph)
            if dtype_code == 0:
                labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint8_t, uint8_t](labelsp8, graphp8, cshape, nd, par)
            elif dtype_code == 1:
                labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint16_t, uint8_t](labelsp16, graphp8, cshape, nd, par)
            elif dtype_code == 2:
                labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint32_t, uint8_t](labelsp32, graphp8, cshape, nd, par)
            else:
                labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint64_t, uint8_t](labelsp64, graphp8, cshape, nd, par)
        elif nd <= 8:
            graphp16 = <uint16_t*> np.PyArray_DATA(graph)
            if dtype_code == 0:
                labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint8_t, uint16_t](labelsp8, graphp16, cshape, nd, par)
            elif dtype_code == 1:
                labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint16_t, uint16_t](labelsp16, graphp16, cshape, nd, par)
            elif dtype_code == 2:
                labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint32_t, uint16_t](labelsp32, graphp16, cshape, nd, par)
            else:
                labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint64_t, uint16_t](labelsp64, graphp16, cshape, nd, par)
        elif nd <= 16:
            graphp32 = <uint32_t*> np.PyArray_DATA(graph)
            if dtype_code == 0:
                labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint8_t, uint32_t](labelsp8, graphp32, cshape, nd, par)
            elif dtype_code == 1:
                labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint16_t, uint32_t](labelsp16, graphp32, cshape, nd, par)
            elif dtype_code == 2:
                labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint32_t, uint32_t](labelsp32, graphp32, cshape, nd, par)
            else:
                labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint64_t, uint32_t](labelsp64, graphp32, cshape, nd, par)
        else:
            graphp64 = <uint64_t*> np.PyArray_DATA(graph)
            if dtype_code == 0:
                labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint8_t, uint64_t](labelsp8, graphp64, cshape, nd, par)
            elif dtype_code == 1:
                labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint16_t, uint64_t](labelsp16, graphp64, cshape, nd, par)
            elif dtype_code == 2:
                labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint32_t, uint64_t](labelsp32, graphp64, cshape, nd, par)
            else:
                labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
                with nogil:
                    build_connectivity_graph[uint64_t, uint64_t](labelsp64, graphp64, cshape, nd, par)
    finally:
        free(cshape)

    return graph


# Signed Distance Function - positive inside foreground, negative in background
def sdf(data, anisotropy=None, black_border=False, int parallel=0):
    """
    Compute the Signed Distance Function (SDF).

    Foreground pixels get positive distance (to nearest background).
    Background pixels get negative distance (to nearest foreground).

    Parameters
    ----------
    data : ndarray
        Input array (binary or labels, 0 = background).
    anisotropy : float or sequence of float, optional
        Per-axis voxel size (default 1.0 for all axes).
    black_border : bool, optional
        Treat image edges as background.
    parallel : int, optional
        Number of threads.

    Returns
    -------
    ndarray
        SDF as float32 array.
    """
    dt = edt(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    dt -= edt(data == 0, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    return dt


def sdfsq(data, anisotropy=None, black_border=False, int parallel=0):
    """Squared SDF - same as sdf but with squared distances."""
    dt = edtsq(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    dt -= edtsq(data == 0, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    return dt

# LEGACY COMPAT (remove when edt_legacy is retired):
try:
    from edt_legacy import each, draw, erase
    import edt_legacy as legacy
except ImportError:
    legacy = None

# expand_labels and feature_transform - ported from barrier implementation
def expand_labels(data, anisotropy=None, black_border=False, int parallel=1, return_features=False):
    """Expand nonzero labels to zeros by nearest-neighbor in Euclidean metric (ND).

    Parameters
    ----------
    data : ndarray
        Input array; nonzero elements are seeds whose values are the labels.
    anisotropy : float or sequence of float, optional
        Per-axis voxel size (default 1.0 for all axes).
    black_border : bool, optional
        Treat image edges as background (default False).
    parallel : int, optional
        Number of threads; if <= 0, uses cpu_count().
    return_features : bool, optional
        If True, also return the feature (nearest-seed linear index) array.

    Returns
    -------
    labels : ndarray, dtype=uint32
        Expanded labels, same shape as input.
    features : ndarray, optional
        If return_features=True, the nearest-seed linear indices.
    """
    cdef int nd
    cdef size_t total
    cdef size_t* cshape
    cdef float* canis
    cdef const uint32_t* data_p
    cdef uint32_t* lout_p
    cdef uint32_t* feat_u32_p
    cdef size_t* feat_sz_p
    cdef bint use_u32_feat
    cdef bint is_fortran
    cdef native_bool bb
    cdef np.ndarray[np.uint32_t, ndim=1] labels_out
    cdef np.ndarray[np.uint32_t, ndim=1] feat_u32
    cdef np.ndarray feat_sz
    cdef int i

    arr = np.asarray(data)
    _check_dims(arr.ndim)
    if arr.dtype == np.int32:
        # Same width — reinterpret without copy; label values are non-negative so
        # bit patterns are identical to the uint32 representation.
        arr, is_fortran = _prepare_array(arr, np.int32)
        arr = arr.view(np.uint32)
    else:
        arr, is_fortran = _prepare_array(arr, np.uint32)
    nd = arr.ndim

    anis = _normalize_anisotropy(anisotropy, nd)

    # For F-contiguous arrays, reverse shape and anisotropy so C++ sees a
    # C-order array of reversed shape — same memory, no copy.
    cpp_shape = arr.shape[::-1] if is_fortran else arr.shape
    cpp_anis  = anis[::-1] if is_fortran else anis

    parallel = _resolve_parallel(parallel)

    bb = black_border

    cshape = <size_t*> malloc(nd * sizeof(size_t))
    canis = <float*> malloc(nd * sizeof(float))
    if cshape == NULL or canis == NULL:
        if cshape != NULL: free(cshape)
        if canis != NULL: free(canis)
        raise MemoryError('Allocation failure')

    total = 1
    for i in range(nd):
        cshape[i] = <size_t>cpp_shape[i]
        canis[i] = <float>cpp_anis[i]
        total *= cshape[i]

    labels_out = np.empty((total,), dtype=np.uint32)
    lout_p = <uint32_t*> np.PyArray_DATA(labels_out)
    data_p = <const uint32_t*> np.PyArray_DATA(arr)

    try:
        if return_features:
            use_u32_feat = (total < (<size_t>1 << 32))
            if use_u32_feat:
                feat_u32 = np.empty((total,), dtype=np.uint32)
                feat_u32_p = <uint32_t*> np.PyArray_DATA(feat_u32)
                with nogil:
                    expand_labels_features_fused[uint32_t, uint32_t](
                        data_p, lout_p, feat_u32_p,
                        cshape, canis, <size_t>nd, bb, parallel)
            else:
                feat_sz = np.empty((total,), dtype=np.uintp)
                feat_sz_p = <size_t*> np.PyArray_DATA(feat_sz)
                with nogil:
                    expand_labels_features_fused[uint32_t, size_t](
                        data_p, lout_p, feat_sz_p,
                        cshape, canis, <size_t>nd, bb, parallel)
        else:
            with nogil:
                expand_labels_fused[uint32_t](
                    data_p, lout_p, cshape, canis, <size_t>nd, bb, parallel)
    finally:
        free(cshape)
        free(canis)

    if return_features:
        if is_fortran:
            # C++ returned flat buffer offsets in cpp_shape (reversed) space.
            # Convert to C-order linear indices in the original arr.shape so the
            # caller gets consistent indices regardless of input memory order.
            if use_u32_feat:
                feat_raw = feat_u32
            else:
                feat_raw = feat_sz
            coords = np.unravel_index(feat_raw.reshape(cpp_shape), cpp_shape)
            feat_dtype = np.uint32 if use_u32_feat else np.uintp
            feat_conv = np.ravel_multi_index(coords[::-1], tuple(arr.shape)).astype(feat_dtype)
            return labels_out.reshape(cpp_shape).T, feat_conv.T
        if use_u32_feat:
            return labels_out.reshape(arr.shape), feat_u32.reshape(arr.shape)
        return labels_out.reshape(arr.shape), feat_sz.reshape(arr.shape)
    return labels_out.reshape(cpp_shape).T if is_fortran else labels_out.reshape(arr.shape)


def configure(
    adaptive_threads=None,
    min_voxels_per_thread=None,
    min_lines_per_thread=None,
):
    """
    Set EDT parameters programmatically, overriding environment variables
    for the current process.

    Parameters
    ----------
    adaptive_threads : bool or None
        Enable adaptive thread limiting based on array size.
        Overrides EDT_ADAPTIVE_THREADS.
    min_voxels_per_thread : int or None
        Minimum voxels per thread (applied for all dims >= 2).
        Overrides EDT_ND_MIN_VOXELS_PER_THREAD.
    min_lines_per_thread : int or None
        Minimum scanlines per thread (applied for all dims >= 2).
        Overrides EDT_ND_MIN_LINES_PER_THREAD.
    """
    if adaptive_threads is not None:
        _ND_CONFIG['EDT_ADAPTIVE_THREADS'] = int(bool(adaptive_threads))
    if min_voxels_per_thread is not None:
        _ND_CONFIG['EDT_ND_MIN_VOXELS_PER_THREAD'] = int(min_voxels_per_thread)
    if min_lines_per_thread is not None:
        _ND_CONFIG['EDT_ND_MIN_LINES_PER_THREAD'] = int(min_lines_per_thread)


def _env_int(name, default):
    if name in _ND_CONFIG:
        return _ND_CONFIG[name]
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _adaptive_thread_limit_nd(parallel, shape):
    """Cap thread count so each thread has enough work to justify its overhead.

    Two criteria, both must hold (whichever allows fewer threads wins):
      - voxels per thread >= EDT_ND_MIN_VOXELS_PER_THREAD (default 4000)
      - scan lines per thread >= EDT_ND_MIN_LINES_PER_THREAD (default 32)

    Applies uniformly for all dims >= 2.
    Disable entirely with EDT_ADAPTIVE_THREADS=0 or edt.configure(adaptive_threads=False).
    """
    parallel = max(1, parallel)
    if not bool(_env_int('EDT_ADAPTIVE_THREADS', 1)):
        return parallel
    if len(shape) <= 1:
        return parallel

    total = 1
    for extent in shape:
        total *= extent
    if total == 0:
        return 1

    longest = max(shape)
    lines = max(1, total // longest)

    min_voxels = max(1, _env_int('EDT_ND_MIN_VOXELS_PER_THREAD', _ND_MIN_VOXELS_PER_THREAD_DEFAULT))
    min_lines  = max(1, _env_int('EDT_ND_MIN_LINES_PER_THREAD',  _ND_MIN_LINES_PER_THREAD_DEFAULT))

    cap = min(max(1, total // min_voxels), max(1, lines // min_lines))
    return max(1, min(parallel, cap))


def feature_transform(data, anisotropy=None, black_border=False, int parallel=1, return_distances=False):
    """ND feature transform (nearest seed) with optional Euclidean distances.

    Parameters
    ----------
    data : ndarray
        Seed image (nonzero are seeds).
    anisotropy : float or sequence of float, optional
        Per-axis voxel size (default 1.0 for all axes).
    black_border : bool, optional
        If True, treat the border as background (default False).
    parallel : int, optional
        Number of threads; if <= 0, uses cpu_count().
    return_distances : bool, optional
        If True, also return the EDT of the seed mask.

    Returns
    -------
    feat : ndarray
        Linear index of nearest seed for each voxel (uint32 or uint64).
    dist : ndarray of float32, optional
        Euclidean distance to nearest seed, if return_distances=True.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        if return_distances:
            return np.zeros_like(arr, dtype=np.uint32), np.zeros_like(arr, dtype=np.float32)
        return np.zeros_like(arr, dtype=np.uint32)

    nd = arr.ndim
    _check_dims(nd)
    anis = _normalize_anisotropy(anisotropy, nd)
    parallel = _resolve_parallel(parallel)

    labels, feats = expand_labels(arr, anisotropy=anis, black_border=black_border, parallel=parallel, return_features=True)

    if return_distances:
        dist = edt(arr != 0, anis, black_border, parallel)
        return feats, dist
    return feats
