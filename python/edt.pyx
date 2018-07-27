"""
Cython binding for the C++ multi-label Euclidean Distance
Transform library by William Silversmith based on the 
algorithms of Felzenzwalb et al. 2012 and Saito et al. 1994.

Given a 1d, 2d, or 3d volume of labels, compute the Euclidean
Distance Transform such that label boundaries are marked as
distance 1 and 0 is always 0.

You can then use 

Key methods: 
  edt1d,   edt2d,   edt3d,
  edt1dsq, edt2dsq, edt3dsq

License: GNU 3.0

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018
"""
from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
   int8_t,  int16_t,  int32_t,  int64_t
)

cimport numpy as numpy
from cpython cimport array 
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.0.0'

cdef extern from "edt.hpp" namespace "edt":
  cdef void squared_edt_1d_multi_seg[T](
    T *labels,
    float *dest,
    int n,
    int stride,
    float anisotropy,
  )

  cdef float* edt2dsq[T](
    T* labels,
    int sx, int sy, 
    int wx, int wy
  )

  cdef float* edt3dsq[T](
    T* labels,
    int sx, int sy, int sz,
    int wx, int wy, int wz
  )

def edt1d(data, anisotropy=1.0):
  return np.sqrt(edt1dsq(data, anisotropy))

def edt1dsq(data, anisotropy=1.0):
  cdef uint8_t[:] arr_memview8
  cdef uint16_t[:] arr_memview16
  cdef uint32_t[:] arr_memview32
  cdef uint64_t[:] arr_memview64

  cdef float* xform = <float*>calloc(data.size, sizeof(float))

  if data.dtype == np.uint8:
    arr_memview8 = data
    squared_edt_1d_multi_seg[uint8_t](
      <uint8_t*>&arr_memview8[0],
      xform,
      data.size,
      1,
      anisotropy
    )
  elif data.dtype == np.uint16:
    arr_memview16 = data
    squared_edt_1d_multi_seg[uint16_t](
      <uint16_t*>&arr_memview16[0],
      xform,
      data.size,
      1,
      anisotropy
    )
  elif data.dtype == np.uint32:
    arr_memview32 = data
    squared_edt_1d_multi_seg[uint32_t](
      <uint32_t*>&arr_memview32[0],
      xform,
      data.size,
      1,
      anisotropy
    )
  elif data.dtype == np.uint64:
    arr_memview64 = data
    squared_edt_1d_multi_seg[uint64_t](
      <uint64_t*>&arr_memview64[0],
      xform,
      data.size,
      1,
      anisotropy
    )
  
  cdef float[:] xform_view = <float[:data.size]>xform

  return np.array(xform_view, dtype=np.float32)
