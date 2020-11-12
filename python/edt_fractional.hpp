/* Multi-Label Anisotropic Euclidean Distance Transform 3D
 *
 * edt, edtsq - compute the euclidean distance transform 
 *     on a single or multi-labeled image all at once.
 *     boolean images are faster.
 *
 * binary_edt, binary_edtsq: Compute the EDT on a binary image
 *     for all input data types. Multiple labels are not handled
 *     but it's faster.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton Neuroscience Insitute
 * Date: October 2020
 */

#ifndef EDT_FRACTIONAL_VOXELS_H
#define EDT_FRACTIONAL_VOXELS_H

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
 *
 * T* segids: 1d array of (un)signed integers
 * *d: write destination, equal sized array as *segids
 * n: size of segids, d
 * stride: typically 1, but can be used on a 
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 *
 * Writes output to *d
 */
template <typename T, typename GRAPH_TYPE>
void squared_edt_1d_multi_seg_fractional_voxels(
    T* segids, 
    GRAPH_TYPE* graph, GRAPH_TYPE mask,
    float *d, const int n, 
    const long int stride, const float anistropy,
    const bool black_border=false
  ) {

  long int i;

  T working_segid = segids[0];

  if (black_border) {
    d[0] = static_cast<float>(working_segid != 0) * anistropy; // 0 or 1
  }
  else {
    d[0] = working_segid == 0 ? 0 : INFINITY;
  }

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid && (graph[i - stride] & mask)) {
      d[i] = d[i - stride] + anistropy;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = static_cast<float>(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  long int min_bound = 0;
  if (black_border) {
    d[n - stride] = static_cast<float>(segids[n - stride] != 0) * anistropy;
    min_bound = stride;
  }

  for (i = (n - 2) * stride; i >= min_bound; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}

 /* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 * Essentially, the distance function can be 
 * modeled as the lower envelope of parabolas
 * that spring mainly from edges of the shape
 * you want to transform. The array is scanned
 * to find the parabolas, then a second scan
 * writes the correct values.
 *
 * O(N) time complexity.
 *
 * I (wms) make a few modifications for our use case
 * of executing a euclidean distance transform on
 * a 3D anisotropic image that contains many segments
 * (many binary images). This way we do it correctly
 * without running EDT > 100x in a 512^3 chunk.
 *
 * The first modification is to apply an envelope 
 * over the entire volume by defining two additional
 * vertices just off the ends at x=-1 and x=n. This
 * avoids needing to create a black border around the
 * volume (and saves 6s^2 additional memory).
 *
 * The second, which at first appeared to be important for
 * optimization, but after reusing memory appeared less important,
 * is to avoid the division operation in computing the intersection
 * point. I describe this manipulation in the code below.
 *
 * I make a third modification in squared_edt_1d_parabolic_multi_seg
 * to enable multiple segments.
 *
 * Parameters:
 *   *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4nm, 4nm, 40nm)
 * 
 * Returns: writes distance transform of f to d
 */
void squared_edt_1d_parabolic_fractional_voxels(
    float* f, 
    float *d, 
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

  int k = 0;
  int* v = new int[n]();
  float* ff = new float[n]();
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }
  
  float* ranges = new float[n + 1]();

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
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 =  i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 =  i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
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

    d[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    if (black_border_left && black_border_right) {
      envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
      d[i * stride] = std::fminf(envelope, d[i * stride]);
    }
    else if (black_border_left) {
      d[i * stride] = std::fminf(w2 * sq(i + 1), d[i * stride]);
    }
    else if (black_border_right) {
      d[i * stride] = std::fminf(w2 * sq(n - i), d[i * stride]);      
    }
  }

  delete [] v;
  delete [] ff;
  delete [] ranges;
}

// about 5% faster
void squared_edt_1d_parabolic_fractional_voxels(
    float* f, 
    float *d, 
    const int n, 
    const long int stride, 
    const float anisotropy
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;

  int k = 0;
  int* v = new int[n]();
  float* ff = new float[n]();
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }

  float* ranges = new float[n + 1]();

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
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 = i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 = i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
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

    d[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
    d[i * stride] = std::fminf(envelope, d[i * stride]);
  }

  delete [] v;
  delete [] ff;
  delete [] ranges;
}

void _squared_edt_1d_parabolic_fractional_voxels(
    float* f, 
    float *d, 
    const int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (black_border_left && black_border_right) {
    squared_edt_1d_parabolic(f, d, n, stride, anisotropy);
  }
  else {
    squared_edt_1d_parabolic(f, d, n, stride, anisotropy, black_border_left, black_border_right); 
  }
}

/* Same as squared_edt_1d_parabolic except that it handles
 * a simultaneous transform of multiple labels (like squared_edt_1d_multi_seg).
 * 
 *  Parameters:
 *    *segids: an integer labeled image where 0 is background
 *    *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4.0 = 4nm, 40.0 = 40nm)
 * 
 * Returns: writes squared distance transform of f to d
 */
template <typename T>
void squared_edt_1d_parabolic_multi_seg_fractional_voxels(
    T* segids, float* f, float *d, 
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
    else if (segid != working_segid) {
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

/* Df(x,y,z) = min( wx^2 * (x-x')^2 + Df|x'(y,z) )
 *              x'                   
 * Df(y,z) = min( wy^2 * (y-y') + Df|x'y'(z) )
 *            y'
 * Df(z) = wz^2 * min( (z-z') + i(z) )
 *          z'
 * i(z) = 0   if voxel in set (f[p] == 1)
 *        inf if voxel out of set (f[p] == 0)
 *
 * In english: a 3D EDT can be accomplished by
 *    taking the x axis EDT, followed by y, followed by z.
 * 
 * The 2012 paper by Felzenszwalb and Huttenlocher describes using
 * an indicator function (above) to use their sampled function
 * concept on all three axes. This is unnecessary. The first
 * transform (x here) can be done very dumbly and cheaply using
 * the method of Rosenfeld and Pfaltz (1966) in 1D (where the L1
 * and L2 norms agree). This first pass is extremely fast and so
 * saves us about 30% in CPU time. 
 *
 * The second and third passes use the Felzenszalb and Huttenlocher's
 * method. The method uses a scan then write sequence, so we are able
 * to write to our input block, which increases cache coherency and
 * reduces memory usage.
 *
 * Parameters:
 *    *labels: an integer labeled image where 0 is background
 *    sx, sy, sz: size of the volume in voxels
 *    wx, wy, wz: physical dimensions of voxels (weights)
 *
 * Returns: writes squared distance transform of f to d
 */
template <typename T, typename GRAPH_TYPE = uint32_t>
float* _edt3dsq_fractional_voxels(
    T* labels, GRAPH_TYPE* graph,
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, 
    float* workspace=NULL
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sz * sxy;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      squared_edt_1d_multi_seg_fractional_voxels<T, GRAPH_TYPE>(
        (labels + sx * y + sxy * z), 
        (workspace + sx * y + sxy * z), 
        sx, 1, wx, black_border
      );
    }
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  for (size_t z = 0; z < sz; z++) {
    for (size_t x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg<T>(
        (labels + x + sxy * z),
        (workspace + x + sxy * z), 
        (workspace + x + sxy * z), 
        sy, sx, wy, black_border
      );
    }
  }

  for (size_t y = 0; y < sy; y++) {
    for (size_t x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg<T>(
        (labels + x + sx * y), 
        (workspace + x + sx * y), 
        (workspace + x + sx * y), 
        sz, sxy, wz, black_border
      );
    }
  }

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

// Same as _edt3dsq, but applies square root to get
// euclidean distance.
template <typename T, typename GRAPH_TYPE = uint32_t>
float* _edt3d_fractional_voxels(
  T* input, GRAPH_TYPE* graph,
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, float* workspace=NULL) {

  float* transform = _edt3dsq_fractional_voxels<T,GRAPH_TYPE>(
    input, graph,
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, workspace
  );

  for (size_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// Should be trivial to make an N-d version
// if someone asks for it. Might simplify the interface.

} // namespace pyedt



#endif

