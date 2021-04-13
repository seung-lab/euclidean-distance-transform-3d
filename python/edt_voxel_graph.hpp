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
 * Date: October-November 2020, April 2021
 */

#ifndef EDT_VOXEL_GRAPH_H
#define EDT_VOXEL_GRAPH_H

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include "threadpool.h"
#include "edt.hpp"

// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {


template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt2dsq_voxel_graph(
    T* labels, GRAPH_TYPE* graph,
    const size_t sx, const size_t sy,
    const float wx, const float wy, 
    const bool black_border=false,  float* workspace=NULL
  ) {

  const size_t voxels = sx * sy;

  if (workspace == NULL) {
    workspace = new float[voxels]();
  }

  for (size_t i = 0; i < voxels; i++) {
    workspace[i] = INFINITY;
  }

  struct coord_type {
    float x;
    float y;

    coord_type(float ix, float iy) : x(ix), y(iy) {}
  };

  std::vector<coord_type> coords;

  size_t loc = 0;

  for (size_t y = 0; y < sy; y++) {
    for (size_t x = 0; x < sx; x++) {
      loc = x + sx * y;
      if (labels[loc] == 0) {
        coords.emplace_back(static_cast<float>(x), static_cast<float>(y));
      }
      if ((graph[loc] & 0b1) == false) {
        coords.emplace_back(static_cast<float>(x) + 0.5, static_cast<float>(y));
      }
      if ((graph[loc] & 0b10) == false) {
        coords.emplace_back(static_cast<float>(x), static_cast<float>(y) + 0.5);
      }
    }
  }

  if (black_border) {
    for (size_t x = 0; x < sx; x++) {
      coords.emplace_back(static_cast<float>(x), -.5);
      coords.emplace_back(static_cast<float>(x), sy - 0.5);
    }
    for (size_t y = 0; y < sy; y++) {
      coords.emplace_back(-0.5, static_cast<float>(y));
      coords.emplace_back(sx - 0.5, static_cast<float>(y));
    }
  }

  for (size_t y = 0; y < sy; y++) {
    for (size_t x = 0; x < sx; x++) {
      loc = x + sx * y;
      for (coord_type coord : coords) {
        float dx = coord.x - x;
        float dy = coord.y - y;
        float dist = dx * dx + dy * dy;
        if (dist < workspace[loc]) {
          workspace[loc] = dist;
        }
      }
    }
  }

  return workspace;
}

// template <typename T, typename GRAPH_TYPE = uint8_t>
// float* _edt2dsq_voxel_graph(
//     T* labels, GRAPH_TYPE* graph,
//     const size_t sx, const size_t sy,
//     const float wx, const float wy, 
//     const bool black_border=false,  float* workspace=NULL
//   ) {

//   const size_t voxels = sx * sy;
//   const size_t sx2 = 2 * sx;

//   uint8_t* double_labels = new uint8_t[voxels * 4]();

//   size_t loc = 0;
//   size_t loc2 = 0;

//   for (size_t y = 0; y < sy; y++) {
//     for (size_t x = 0; x < sx; x++) {
//       loc = x + sx * y;
//       loc2 = 2 * x + 4 * sx * y;

//       uint8_t foreground = labels[loc] > 0;

//       double_labels[loc2] = foreground;
//       double_labels[loc2 + 1] = foreground && (graph[loc] & 0b00000001);
//       double_labels[loc2 + sx2] = foreground && (graph[loc] & 0b00000010);
//       double_labels[loc2 + sx2 + 1] = foreground;
//     }
//     if (black_border) {
//       double_labels[loc2 + 1] = 0;
//       double_labels[loc2 + sx2 + 1] = 0;
//     }
//   }
//   if (black_border) {
//     for (size_t x = 0; x < sx2; x++) {
//       double_labels[4 * voxels - x - 1] = 0;
//     }
//   }

//   float* transform2 = _edt2dsq<uint8_t>(
//     double_labels, 
//     sx*2, sy*2,
//     wx / 2, wy / 2,
//     black_border, /*parallel=*/1
//   );

//   delete[] double_labels;

//   if (workspace == NULL) {
//     workspace = new float[voxels]();
//   }

//   for (size_t y = 0; y < sy; y++) {
//     for (size_t x = 0; x < sx; x++) {
//       loc = x + sx * y;
//       loc2 = 2 * x + 4 * sx * y;

//       workspace[loc] = transform2[loc2];
//     }
//   }
//   delete[] transform2;

//   return workspace;
// }

template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt3dsq_voxel_graph(
    T* labels, GRAPH_TYPE* graph,
    const size_t sx, const size_t sy, const size_t sz,
    const float wx, const float wy, const float wz, 
    const bool black_border=false,  float* workspace=NULL
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  if (workspace == NULL) {
    workspace = new float[voxels]();
  }

  for (size_t i = 0; i < voxels; i++) {
    workspace[i] = INFINITY;
  }

  struct coord_type {
    float x;
    float y;
    float z;

    coord_type(float ix, float iy, float iz) : x(ix), y(iy), z(iz) {}
  };

  std::vector<coord_type> coords;

  size_t loc = 0;

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++) {
        loc = x + sx * y + sxy * z;
        if (labels[loc] == 0) {
          coords.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
        if ((graph[loc] & 0b1) == false) {
          coords.emplace_back(static_cast<float>(x) + 0.5, static_cast<float>(y), static_cast<float>(z));
        }
        if ((graph[loc] & 0b10) == false) {
          coords.emplace_back(static_cast<float>(x), static_cast<float>(y) + 0.5, static_cast<float>(z));
        }
        if ((graph[loc] & 0b10) == false) {
          coords.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) + 0.5);
        }
      }
    }
  }

  if (black_border) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++) {
        coords.emplace_back(static_cast<float>(x), static_cast<float>(y), -0.5);
        if (sz > 1) {
          coords.emplace_back(static_cast<float>(x), static_cast<float>(y), sz - 0.5);
        }
      }
    }
    for (size_t z = 1; z < sz - 1; z++) {
      for (size_t x = 0; x < sx; x++) {
        coords.emplace_back(static_cast<float>(x), -0.5, static_cast<float>(z));
        coords.emplace_back(static_cast<float>(x), sy - 0.5, static_cast<float>(z));
      }
      for (size_t y = 0; y < sy; y++) {
        coords.emplace_back(-0.5, static_cast<float>(y), static_cast<float>(z));
        coords.emplace_back(sx - 0.5, static_cast<float>(y), static_cast<float>(z));
      }
    }
  }

  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++) {
        loc = x + sx * y + sxy * z;
        for (coord_type coord : coords) {
          float dx = coord.x - x;
          float dy = coord.y - y;
          float dz = coord.z - z;
          float dist = dx * dx + dy * dy + dz * dz;
          if (dist < workspace[loc]) {
            workspace[loc] = dist;
          }
        }
      }
    }
  }

  return workspace;
}

// template <typename T, typename GRAPH_TYPE = uint8_t>
// float* _edt3dsq_voxel_graph(
//     T* labels, GRAPH_TYPE* graph,
//     const size_t sx, const size_t sy, const size_t sz, 
//     const float wx, const float wy, const float wz,
//     const bool black_border=false,  float* workspace=NULL
//   ) {

//   const size_t sxy = sx * sy;
//   const size_t voxels = sx * sy * sz;
//   const size_t sx2 = 2 * sx;
//   const size_t sxy2 = 4 * sxy;

//   uint8_t* double_labels = new uint8_t[voxels * 8]();

//   size_t loc = 0;
//   size_t loc2 = 0;

//   size_t x, y, z;

//   for (z = 0; z < sz; z++) {
//     for (y = 0; y < sy; y++) {
//       for (x = 0; x < sx; x++) {
//         loc = x + sx * y + sxy * z;
//         loc2 = 2 * x + 4 * sx * y + 8 * sxy * z;

//         uint8_t foreground = labels[loc] > 0;

//         double_labels[loc2] = foreground;
//         double_labels[loc2 + 1] = foreground && (graph[loc] & 0b00000001);
//         double_labels[loc2 + sx2] = foreground && (graph[loc] & 0b00000010);
//         double_labels[loc2 + sxy2] = foreground && (graph[loc] & 0b00000100);
//         double_labels[loc2 + sx2 + 1] = foreground;
//         double_labels[loc2 + sxy2 + 1] = foreground;
//         double_labels[loc2 + sx2 + sxy2] = foreground;
//         double_labels[loc2 + sx2 + sxy2 + 1] = foreground;
//       }
//       if (black_border) {
//         double_labels[loc2 + 1] = 0;
//         double_labels[loc2 + sx2 + 1] = 0;
//         double_labels[loc2 + 1 + sxy2] = 0;
//         double_labels[loc2 + sx2 + 1 + sxy2] = 0;
//       }
//     }
//     if (black_border) {
//       y = sy - 1;
//       for (x = 0; x < sx; x++) {
//         loc2 = 2 * x + 4 * sx * y + 8 * sxy * z;

//         double_labels[loc2 + sx2] = 0;
//         double_labels[loc2 + sx2 + 1] = 0;
//         double_labels[loc2 + sx2 + sxy2] = 0;
//         double_labels[loc2 + sx2 + sxy2 + 1] = 0;
//       }
//     }
//   }
//   if (black_border) {
//     z = sz - 1;
//     for (y = 0; y < sy; y++) {
//       for (x = 0; x < sx; x++) {
//         loc2 = 2 * x + 4 * sx * y + 8 * sxy * z;

//         double_labels[loc2 + sxy2] = 0;
//         double_labels[loc2 + sxy2 + 1] = 0;
//         double_labels[loc2 + sx2 + sxy2] = 0;
//         double_labels[loc2 + sx2 + sxy2 + 1] = 0;
//       }    
//     }
//   }

//   float* transform2 = _edt3dsq<uint8_t>(
//     double_labels, sx*2, sy*2, sz*2,
//     wx / 2, wy / 2, wz / 2,
//     black_border, /*parallel=*/1
//   );

//   delete[] double_labels;

//   if (workspace == NULL) {
//     workspace = new float[voxels]();
//   }

//   for (z = 0; z < sz; z++) {
//     for (y = 0; y < sy; y++) {
//       for (x = 0; x < sx; x++) {
//         loc = x + sx * y + sxy * z;
//         loc2 = 2 * x + 4 * sx * y + 8 * sxy * z;

//         workspace[loc] = transform2[loc2];
//       }
//     }
//   }
//   delete[] transform2;

//   return workspace;
// }

template <typename T, typename GRAPH_TYPE = uint8_t>
float* _edt3d_voxel_graph(
  T* labels, GRAPH_TYPE* graph,
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, float* workspace=NULL) {

  float* transform = _edt3dsq_voxel_graph<T, GRAPH_TYPE>(
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

