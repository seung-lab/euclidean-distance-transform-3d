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
 *
 * Returns whether the other passes should include an offset starting from 
 * this row. 
 */
template <typename T, typename GRAPH_TYPE>
bool squared_edt_1d_multi_seg_voxel_graph(
    T* segids, 
    GRAPH_TYPE* graph, 
    const GRAPH_TYPE fwd_mask, 
    const GRAPH_TYPE bwd_mask,
    const GRAPH_TYPE other_mask,
    float *d, const int n, 
    const long int stride, const float anistropy,
    const bool black_border=false
  ) {

  long int i;

  T working_segid = segids[0];
  GRAPH_TYPE full_mask = fwd_mask | bwd_mask;

  bool offset = false;

  if (black_border) {
    d[0] = static_cast<float>(working_segid != 0) * anistropy; // 0 or 1
    if ((graph[0] & fwd_mask) != fwd_mask) {
      d[0] /= 2;
    }
  }
  else if (working_segid == 0) {
    d[0] = 0;
  }
  else if ((graph[0] & fwd_mask) != fwd_mask) {
    d[0] = anistropy / 2;
  }
  else if ((graph[0] & other_mask) != other_mask) {
    d[0] = 0;
    offset = true;
  }
  else {
    d[0] = INFINITY;
  }

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid && (graph[i] & other_mask) != other_mask) {
      d[i] = 0.0;
      offset = true;
    }
    else if (segids[i] == working_segid && (graph[i] & full_mask) == full_mask) {
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
    if ((graph[(n - 1) * stride] & other_mask) != other_mask) {
      d[n - stride] = 0.0;
      offset = true;
    }
    else if ((graph[(n - 1) * stride] & bwd_mask) != bwd_mask) {
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

  return offset;
}

template <typename GRAPH_TYPE = uint8_t>
void squared_edt_1d_parabolic_voxel_graph(
    float* f, 
    const GRAPH_TYPE* graph, 
    const GRAPH_TYPE fwd_mask, 
    const GRAPH_TYPE bwd_mask,
    const int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;
  const GRAPH_TYPE full_mask = fwd_mask | bwd_mask;

  int* v = new int[2*n + 1](); // 2*n+1 = voxels and edges
  float* ff = new float[2*n + 1]();

  for (long int i = 0; i < n; i++) {
    ff[2*i] = ((graph[i * stride] & bwd_mask) == bwd_mask) * (w2 * sq(std::max(i, n - i) + 0.5) + f[i * stride] + 1); // 0 or too big to ever be useful
    if (ff[2*i] == 0 || ((graph[i * stride] & fwd_mask) != fwd_mask)) {
      ff[2*i + 1] = w2 * sq(n) + 1; // too big to ever be useful  
    }
    else {
      ff[2*i + 1] = f[i * stride];  
    }
    printf("%.2f, %.2f, ", ff[2*i], ff[2*i+1]);
  }
  ff[2*n] = ((graph[(n-1) * stride] & fwd_mask) == fwd_mask) * (w2 * sq(n + 0.5) * f[(n-1) * stride] + 1);

  int k = 0;
  float* ranges = new float[2*n + 1 + 1]();

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
  for (long int i = 1; i < 2*n + 1; i++) {
    factor1 = static_cast<float>(i - v[k]) * w2 / 2.0;
    factor2 = static_cast<float>(i + v[k]) / 2.0 - 1;
    s = (factor1 * factor2 + ff[i] - ff[v[k]]) / (2.0 * factor1);

    while (s <= ranges[k]) {
      k--;
      factor1 = static_cast<float>(i - v[k]) * w2 / 2.0;
      factor2 = static_cast<float>(i + v[k]) / 2.0 - 1;
      s = (factor1 * factor2 + ff[i] - ff[v[k]]) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    f[i * stride] = w2 * sq(i - (static_cast<float>(v[k] - 1) / 2.0)) + ff[v[k]];
    if (black_border_left && black_border_right) {
      envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
      f[i * stride] = std::fminf(envelope, f[i * stride]);
    }
    else if (black_border_left) {
      f[i * stride] = std::fminf(w2 * sq(i + 1), f[i * stride]);
    }
    else if (black_border_right) {
      f[i * stride] = std::fminf(w2 * sq(n - i), f[i * stride]);      
    }
  }

  delete [] v;
  delete [] ff;
  delete [] ranges;
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
    T* segids, float* f, 
    GRAPH_TYPE* graph, 
    const GRAPH_TYPE fwd_mask, 
    const GRAPH_TYPE bwd_mask, 
    const int n, const long int stride, const float anisotropy,
    const bool black_border=false) {

  T working_segid = segids[0];
  T segid;
  long int last = 0;

  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid == 0) {
      continue;
    }

    if (segid != working_segid || !(graph[i * stride] & bwd_mask)) {
      if (working_segid != 0) {
        squared_edt_1d_parabolic_voxel_graph<GRAPH_TYPE>(
          (f + last * stride), 
          (graph + last * stride), fwd_mask, bwd_mask,
          i - last, stride, anisotropy,
          (black_border || last > 0), (i < n - 1) 
        );
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    squared_edt_1d_parabolic_voxel_graph<GRAPH_TYPE>(
      (f + last * stride), 
      (graph + last * stride), fwd_mask, bwd_mask,
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
    workspace = new float[voxels]();
  }

  const GRAPH_TYPE fwd_xmask = 0b00000001;
  const GRAPH_TYPE bwd_xmask = 0b00000010;
  const GRAPH_TYPE other_mask = 0b00001100;

  for (size_t y = 0; y < sy; y++) {
    squared_edt_1d_multi_seg_voxel_graph<T, GRAPH_TYPE>(
      (labels + sx * y), 
      (graph + sx * y), fwd_xmask, bwd_xmask, other_mask,
      (workspace + sx * y), 
      /*n=*/sx, /*stride=*/1, /*anisotropy=*/wx, 
      black_border
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
      (graph + x), fwd_ymask, bwd_ymask,
      /*n=*/sy, /*stride=*/sx, /*anisotropy=*/wy, 
      black_border
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

  const GRAPH_TYPE fwd_xmask = 0b00010101;
  const GRAPH_TYPE bwd_xmask = 0b00101010;
  const GRAPH_TYPE other_mask = 0b00111100;

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      squared_edt_1d_multi_seg_voxel_graph<T, GRAPH_TYPE>(
        (labels + sx * y + sxy * z), 
        (graph + sx * y + sxy * z), fwd_xmask, bwd_xmask, other_mask,
        (workspace + sx * y + sxy * z), 
        /*n=*/sx, /*stride=*/1, /*anisotropy=*/wx, 
        black_border
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
        (graph + x + sxy * z), fwd_ymask, bwd_ymask,
        /*n=*/sy, /*stride=*/sx, /*anisotropy=*/wy, 
        black_border
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
        (graph + x + sx * y), fwd_zmask, bwd_zmask,
        /*n=*/sz, /*stride=*/sxy, /*anisotropy=*/wz, 
        black_border
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

