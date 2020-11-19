/* Multi-Label Anisotropic Euclidean Distance Transform 3D
 *
 * This file contains a specialization for handling voxel
 * graphs that restrict movement in particular directions.
 * This is a simple version of handling fractional voxels.
 *
 * The definiton of this graph lives in
 * https://github.com/seung-lab/connected-components-3d/blob/master/cc3d_graphs.hpp
 * because that is the function that will produce it.
 *
 * There are many ways to define fractional voxels. Here we use
 * a definition consistent with the ordinary edt in that we 
 * concentrate the volume of the voxel into a point. In the
 * ordinary EDT, each voxel's mass is located in the center
 * of a voxel. When the voxel graph prevents motion along 
 * a particular direction, we consider the edge between those
 * two voxels to contain a point of background. This means
 * that along the x axis, the unobstructed distance from a 
 * background voxel to a foreground voxel is 1, but from 
 * an obstructed edge to a voxel is 1/2.
 *
 * The edt represents the foreground labels as 6-connected,
 * and so we only concern outselves with the XYZ axes and not
 * any of the diagonals in the graph representation.
 * 
 * edt_voxel_graph - compute the euclidean distance transform 
 *     on a single or multi-labeled image all at once that exist
 *     in the context of a voxel graph that proscribes certain
 *     directions for moving between
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton Neuroscience Insitute
 * Date: October-November 2020
 */

#ifndef EDT_VOXEL_GRAPH_H
#define EDT_VOXEL_GRAPH_H

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include "threadpool.h"
#include "edt.hpp"

// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {

/* 1D Euclidean Distance Transform for Multiple Segids
 *
 * Map a row of segids to a euclidean distance transform.
 * Zero is considered a universal boundary as are differing
 * segids. Segments touching the boundary are mapped to 1.
 * Segments touching an obstructed edge are mapped to 0.5.
 *
 * T* segids: 1d array of (un)signed integers
 * GRAPH_TYPE* graph: a 1D array containing bitfields
 *    that indicate which directions can be traversed
 *    from a given voxel.
 * GRAPH_TYPE fwd_mask: mask representing the bit that
 *    indicates the forward direction of travel.
* GRAPH_TYPE bwd_mask: mask representing the bit that
 *    indicates the backward direction of travel.
 * *d: write destination, equal sized array as *segids
 * n: size of segids, d
 * stride: typically 1, but can be used on a 
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 *
 * Writes output to *d
 */
template <typename T, typename GRAPH_TYPE>
void squared_edt_1d_multi_seg_voxel_graph(
    T* segids, 
    GRAPH_TYPE* graph, GRAPH_TYPE fwd_mask, GRAPH_TYPE bwd_mask,
    float *d, const int n, 
    const long int stride, const float anistropy,
    const bool black_border=false
  ) {

  long int i;

  T working_segid = segids[0];
  GRAPH_TYPE full_mask = fwd_mask | bwd_mask;

  if (black_border) {
    d[0] = static_cast<float>(working_segid != 0) * anistropy; // 0 or 1
    if ((graph[0] & fwd_mask) == false) {
      d[0] /= 2;
    }
  }
  else {
    d[0] = working_segid == 0 ? 0 : INFINITY;
  }

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid && (graph[i - stride] & full_mask) == full_mask) {
      d[i] = d[i - stride] + anistropy;
    }
    else if (segids[i] == working_segid) {
      d[i] = anistropy / 2;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = static_cast<float>(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  long int min_bound = 0;
  if (black_border) {
    d[(n - 1) * stride] = static_cast<float>(segids[(n - 1) * stride] != 0) * anistropy;
    if ((graph[(n - 1) * stride] & bwd_mask) == false) {
      d[n - stride] /= 2;
    }
    min_bound = stride;
  }

  for (i = (n - 2) * stride; i >= min_bound; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}

/* Same as squared_edt_1d_parabolic except that it handles
 * a simultaneous transform of multiple labels (like squared_edt_1d_multi_seg)
 * and accounts for the allowed directions specified by the voxel
 * connectivity graph. 
 *
 *  Parameters:
 *    *segids: an integer labeled image where 0 is background
 *    *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    GRAPH_TYPE* graph: a 1D array containing bitfields
 *       that indicate which directions can be traversed
 *       from a given voxel.
 *    GRAPH_TYPE fwd_mask: mask representing the bit that
 *       indicates the forward direction of travel.
 *    GRAPH_TYPE bwd_mask: mask representing the bit that
 *       indicates the backward direction of travel.
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4.0 = 4nm, 40.0 = 40nm)
 * 
 * Returns: writes squared distance transform of f to d
 */
template <typename T, typename GRAPH_TYPE>
void squared_edt_1d_parabolic_multi_seg_voxel_graph(
    T* segids, float* f, float* d, 
    GRAPH_TYPE* graph, GRAPH_TYPE fwd_mask, GRAPH_TYPE bwd_mask,
    const int n, const long int stride, const float anisotropy,
    const bool black_border=false) {

  T working_segid = segids[0];
  T segid;
  GRAPH_TYPE bitfield;
  GRAPH_TYPE full_mask = fwd_mask | bwd_mask;
  long int last = 0;

  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid == 0) {
      continue;
    }

    bitfield = graph[i * stride];
    if (segid == working_segid && !(bitfield & full_mask)) {
      f[i * stride] = anisotropy / 2.0;
    }

    if (segid != working_segid || !(bitfield & bwd_mask)) {
      if (working_segid != 0) {
        _squared_edt_1d_parabolic(
          f + last * stride, 
          d + last * stride, 
          i - last, stride, anisotropy,
          (black_border || last > 0), (i < n - 1) 
        );
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    _squared_edt_1d_parabolic(
      f + last * stride, 
      d + last * stride, 
      n - last, stride, anisotropy,
      (black_border || last > 0), black_border
    );
  }
}

template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt2dsq_voxel_graph(
    T* labels, GRAPH_TYPE* graph,
    const size_t sx, const size_t sy,
    const float wx, const float wy, 
    const bool black_border=false,  float* workspace=NULL
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sxy;

  if (workspace == NULL) {
    workspace = new float[sx * sy]();
  }

  const GRAPH_TYPE fwd_xmask = 0b00000001;
  const GRAPH_TYPE bwd_xmask = 0b00000010;

  for (size_t y = 0; y < sy; y++) {
    squared_edt_1d_multi_seg_voxel_graph<T, GRAPH_TYPE>(
      (labels + sx * y), 
      (graph + sx * y), fwd_xmask, bwd_xmask,
      (workspace + sx * y), 
      sx, 1, wx, black_border
    );
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  const GRAPH_TYPE fwd_ymask = 0b00000100;
  const GRAPH_TYPE bwd_ymask = 0b00001000;

  for (size_t x = 0; x < sx; x++) {
    squared_edt_1d_parabolic_multi_seg_voxel_graph<T, GRAPH_TYPE>(
      (labels + x),
      (workspace + x), 
      (workspace + x), 
      (graph + x), fwd_ymask, bwd_ymask,
      sy, sx, wy, black_border
    );
  }

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt3dsq_voxel_graph(
    T* labels, GRAPH_TYPE* graph,
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false,  float* workspace=NULL
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sz * sxy;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }

  const GRAPH_TYPE fwd_xmask = 0b00000001;
  const GRAPH_TYPE bwd_xmask = 0b00000010;

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      squared_edt_1d_multi_seg_voxel_graph<T, GRAPH_TYPE>(
        (labels + sx * y + sxy * z), 
        (graph + sx * y + sxy * z), fwd_xmask, bwd_xmask,
        (workspace + sx * y + sxy * z), 
        sx, 1, wx, black_border
      );
    }
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  const GRAPH_TYPE fwd_ymask = 0b00000100;
  const GRAPH_TYPE bwd_ymask = 0b00001000;

  for (size_t z = 0; z < sz; z++) {
    for (size_t x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg_voxel_graph<T,GRAPH_TYPE>(
        (labels + x + sxy * z),
        (workspace + x + sxy * z), 
        (workspace + x + sxy * z), 
        (graph + x + sxy * z), fwd_ymask, bwd_ymask,
        sy, sx, wy, black_border
      );
    }
  }

  const GRAPH_TYPE fwd_zmask = 0b00010000;
  const GRAPH_TYPE bwd_zmask = 0b00100000;

  for (size_t y = 0; y < sy; y++) {
    for (size_t x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg_voxel_graph<T,GRAPH_TYPE>(
        (labels + x + sx * y), 
        (workspace + x + sx * y), 
        (workspace + x + sx * y), 
        (graph + x + sx * y), fwd_zmask, bwd_zmask,
        sz, sxy, wz, black_border
      );
    }
  }

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt3d_voxel_graph(
  T* labels, GRAPH_TYPE* graph,
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, float* workspace=NULL) {

  float* transform = _edt3dsq_voxel_graph<T,GRAPH_TYPE>(
    labels, graph,
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, workspace
  );

  for (size_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

} // namespace pyedt



#endif

