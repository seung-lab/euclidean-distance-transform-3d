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
 * Date: July 2018
 */

#include <cmath>
#include <cstdint>
#include <stdio.h>

#include <string.h>
#include <algorithm>

#ifndef EDT_H
#define EDT_H


// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {

#define sq(x) ((x) * (x))

const int VERSION_MAJOR = 1;
const int VERSION_MINOR = 0;
const int VERSION_BUGFIX = 0;

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
template <typename T>
void squared_edt_1d_multi_seg(
    T* segids, float *d, const int n, 
    const int stride, const float anistropy
  ) {
  int i;

  T working_segid = segids[0];

  d[0] = (float)(working_segid != 0) * anistropy; // 0 or 1
  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid) {
      d[i] = d[i - stride] + anistropy;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = (float)(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  d[n - stride] = (float)(segids[n - stride] != 0) * anistropy;
  for (i = (n - 2) * stride; i >= stride; i -= stride) {
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
void squared_edt_1d_parabolic(float* f, float *d, const int n, const int stride, const float anisotropy) {
  int k = 0;
  int* v = new int[n]();
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
  for (int i = 1; i < n; i++) {
    factor1 = i - v[k];
    factor2 = i + v[k];
    s = (f[i * stride] - f[v[k] * stride] + factor1 * factor2) / (2.0 * factor1);

    while (s <= ranges[k]) {
      k--;
      factor1 = i - v[k];
      factor2 = i + v[k];
      s = (f[i * stride] - f[v[k] * stride] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (int i = 0; i < n; i++) {
    // compensate for not dividing ranges by 2.0 earlier w/ bit shift left
    // and use factor1 from earlier
    while (ranges[k + 1] < i) { 
      k++;
    }

    d[i * stride] = sq(anisotropy * (i - v[k])) + f[v[k] * stride];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    envelope = std::fminf(sq(anisotropy * (i + 1)), sq(anisotropy * (n - i)));
    d[i * stride] = std::fminf(envelope, d[i * stride]);
  }

  delete [] v;
  delete [] ranges;
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
void squared_edt_1d_parabolic_multi_seg(
    T* segids, float* f, float *d, 
    const int n, const int stride, const float anisotropy
  ) {

  T working_segid = segids[0];
  T segid;
  int last = 0;
  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid == 0) {
      continue;
    }
    else if (segid != working_segid) {
      squared_edt_1d_parabolic(f + last * stride, d + last * stride, i - last, stride, anisotropy);
      working_segid = segid;
      last = i;
    }
  }

  if (last < n) {
    squared_edt_1d_parabolic(f + last * stride, d + last * stride, n - last, stride, anisotropy);
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
template <typename T>
float* _edt3dsq(T* labels, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz) {

  const size_t sxy = sx * sy;

  float *workspace = new float[sx * sy * sz]();
  for (int z = 0; z < sz; z++) {
    for (int y = 0; y < sy; y++) { 
      // Might be possible to write this as a single pass, might be faster
      // however, it's already only using about 3-5% of total CPU time.
      // NOTE: Tried it, same speed overall.
      squared_edt_1d_multi_seg<T>(
        (labels + sx * y + sxy * z), 
        (workspace + sx * y + sxy * z), 
        sx, 1, wx); 
    }
  }

  for (int z = 0; z < sz; z++) {
    for (int x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg<T>(
        (labels + x + sxy * z),
        (workspace + x + sxy * z), 
        (workspace + x + sxy * z), 
        sy, sx, wy);
    }
  }

  for (int y = 0; y < sy; y++) {
    for (int x = 0; x < sx; x++) {
      squared_edt_1d_parabolic_multi_seg<T>(
        (labels + x + sx * y), 
        (workspace + x + sx * y), 
        (workspace + x + sx * y), 
        sz, sxy, wz);
    }
  }

  return workspace; 
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt3dsq(T* binaryimg, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz) {

  const size_t sxy = sx * sy;

  float *workspace = new float[sx * sy * sz]();
  for (int z = 0; z < sz; z++) {
    for (int y = 0; y < sy; y++) { 
      // Might be possible to write this as a single pass, might be faster
      // however, it's already only using about 3-5% of total CPU time.
      // NOTE: Tried it, same speed overall.
      squared_edt_1d_multi_seg<T>(
        (binaryimg + sx * y + sxy * z), 
        (workspace + sx * y + sxy * z), 
        sx, 1, wx); 
    }
  }

  for (int z = 0; z < sz; z++) {
    for (int x = 0; x < sx; x++) {
      squared_edt_1d_parabolic(
        (workspace + x + sxy * z), 
        (workspace + x + sxy * z), 
        sy, sx, wy);
    }
  }

  for (int y = 0; y < sy; y++) {
    for (int x = 0; x < sx; x++) {
      squared_edt_1d_parabolic(
        (workspace + x + sx * y), 
        (workspace + x + sx * y), 
        sz, sxy, wz);
    }
  }

  return workspace; 
}

// about 20% faster on binary images by skipping
// multisegment logic in parabolic
template <typename T>
float* _edt3dsq(bool* binaryimg, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz) {

  return _binary_edt3dsq(binaryimg, sx, sy, sz, wx, wy, wz);
}

// Same as _edt3dsq, but applies square root to get
// euclidean distance.
template <typename T>
float* _edt3d(T* input, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz) {

  float* transform = _edt3dsq<T>(input, sx, sy, sz, wx, wy, wz);

  for (int i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt3d(T* input, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz) {

  float* transform = _binary_edt3dsq<T>(input, sx, sy, sz, wx, wy, wz);

  for (int i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
float* _edt2dsq(T* input, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  float *xaxis = new float[sx * sy]();
  for (int y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>((input + sx * y), (xaxis + sx * y), sx, 1, wx); 
  }

  for (int x = 0; x < sx; x++) {
    squared_edt_1d_parabolic_multi_seg<T>(
      (input + x), 
      (xaxis + x), 
      (xaxis + x), 
      sy, sx, wy);
  }

  return xaxis;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt2dsq(T* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  float *xaxis = new float[sx * sy]();
  for (int y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>((binaryimg + sx * y), (xaxis + sx * y), sx, 1, wx); 
  }

  for (int x = 0; x < sx; x++) {
    squared_edt_1d_parabolic(
      (xaxis + x), 
      (xaxis + x), 
      sy, sx, wy);
  }

  return xaxis;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt2d(T* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  float *transform = _binary_edt2dsq(binaryimg, sx, sy, wx, wy);

  for (int i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
float* _edt2dsq(bool* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  return _binary_edt2dsq(binaryimg, sx, sy, wx, wy);
}

// returns euclidean distance instead of squared distance
template <typename T>
float* _edt2d(T* input, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  float* transform = _edt2dsq<T>(input, sx, sy, wx, wy);

  for (int i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}


// Should be trivial to make an N-d version
// if someone asks for it. Might simplify the interface.

} // namespace pyedt

namespace edt {

template <typename T>
float* edt(T* labels, int sx, float wx) {
  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx);

  for (int i = 0; i < sx; i++) {
    d[i] = std::sqrt(d[i]);
  }

  return d;
}

template <typename T>
float* edt(T* labels, int sx, int sy, float wx, float wy) {
  return pyedt::_edt2d(labels, sx, sy, wx, wy);
}


template <typename T>
float* edt(T* labels, int sx, int sy, int sz, float wx, float wy, float wz) {
  return pyedt::_edt3d(labels, sx, sy, sz, wx, wy, wz);
}

template <typename T>
float* binary_edt(T* labels, int sx, float wx) {
  return edt::edt(labels, sx, wx);
}

template <typename T>
float* binary_edt(T* labels, int sx, int sy, float wx, float wy) {
  return pyedt::_binary_edt2d(labels, sx, sy, wx, wy);
}

template <typename T>
float* binary_edt(T* labels, int sx, int sy, int sz, float wx, float wy, float wz) {
  return pyedt::_binary_edt3d(labels, sx, sy, sz, wx, wy, wz);
}

template <typename T>
float* edtsq(T* labels, int sx, float wx) {
  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx);
  return d;
}

template <typename T>
float* edtsq(T* labels, int sx, int sy, float wx, float wy) {
  return pyedt::_edt2dsq(labels, sx, sy, wx, wy);
}

template <typename T>
float* edtsq(T* labels, int sx, int sy, int sz, float wx, float wy, float wz) {
  return pyedt::_edt3dsq(labels, sx, sy, sz, wx, wy, wz);
}

template <typename T>
float* binary_edtsq(T* labels, int sx, float wx) {
  return edt::edtsq(labels, sx, wx);
}

template <typename T>
float* binary_edtsq(T* labels, int sx, int sy, float wx, float wy) {
  return pyedt::_binary_edt2dsq(labels, sx, sy, wx, wy);
}

template <typename T>
float* binary_edtsq(T* labels, int sx, int sy, int sz, float wx, float wy, float wz) {
  return pyedt::_binary_edt3dsq(labels, sx, sy, sz, wx, wy, wz);
}


} // namespace edt



#endif

