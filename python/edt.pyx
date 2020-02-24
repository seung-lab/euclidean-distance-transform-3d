"""
Cython binding for the C++ multi-label Euclidean Distance
Transform library by William Silversmith based on the 
algorithms of Meijister et al (2002) Felzenzwalb et al. (2012) 
and Saito et al. (1994).

Given a 1d, 2d, or 3d volume of labels, compute the Euclidean
Distance Transform such that label boundaries are marked as
distance 1 and 0 is always 0.

Key methods: 
  edt, edtsq
  edt1d,   edt2d,   edt3d,
  edt1dsq, edt2dsq, edt3dsq

License: GNU 3.0

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July-November 2018
"""
from libc.stdlib cimport calloc, free
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
   int8_t,  int16_t,  int32_t,  int64_t
)
from libcpp cimport bool

import multiprocessing

from cpython cimport array 
cimport numpy as np
import numpy as np

__VERSION__ = '2.0.1'

cdef extern from "edt.hpp" namespace "pyedt":
  cdef void squared_edt_1d_multi_seg[T](
    T *labels,
    float *dest,
    int n,
    int stride,
    float anisotropy,
    bool black_border
  ) nogil

  cdef float* _edt2dsq[T](
    T* labels,
    size_t sx, size_t sy, 
    float wx, float wy,
    bool black_border, int parallel,
    float* output
  ) nogil

  cdef float* _edt3dsq[T](
    T* labels, 
    size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool black_border, int parallel,
    float* output
  ) nogil

def nvl(val, default_val):
  if val is None:
    return default_val
  return val

def edt(data, anisotropy=None, black_border=False, order='K', int parallel=1):
  """
  edt(data, anisotropy=None, black_border=False, order='K', parallel=1)

  Computes the anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    order: 'K','C' or 'F' interpret the input data as C (row major) 
      or Fortran (column major) order. 'K' means "Keep", and will
      detect whether the array is arleady C or F order and use that.
      If the array is discontiguous in 'K' mode, it will be copied into 
      C order, otherwise if 'C' or 'F' is specified, it will copy it
      into that layout.
    parallel: number of threads to use (only applies to 2D and 3D)

  Returns: EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape, dtype=np.float32)

  if not data.flags.c_contiguous and not data.flags.f_contiguous:
    order = 'C' if order != 'F' else 'F'
    data = np.copy(data, order=order)
  elif order == 'K' and data.flags.c_contiguous:
    order = 'C'
  elif order == 'K' and data.flags.f_contiguous:
    order = 'F'

  if order not in ('C', 'F'):
    raise ValueError("order must be 'K', 'C' or 'F'. Got: " + str(order))

  if parallel <= 0:
    parallel = multiprocessing.cpu_count()

  if dims == 1:
    anisotropy = nvl(anisotropy, 1.0)
    return edt1d(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = nvl(anisotropy, (1.0, 1.0))
    return edt2d(data, anisotropy, black_border, order, parallel=parallel)
  elif dims == 3:
    anisotropy = nvl(anisotropy, (1.0, 1.0, 1.0))
    return edt3d(data, anisotropy, black_border, order, parallel=parallel)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

def edtsq(data, anisotropy=None, bool black_border=False, order='C', int parallel=1):
  """
  edtsq(data, anisotropy=None, black_border=False, order='C', parallel=1)

  Computes the squared anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  Squaring allows for omitting an sqrt operation, so may be faster if your use case allows for it.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    order: 'C' or 'F' interpret the input data as C (row major) 
      or Fortran (column major) order.
    parallel: number of threads to use (only applies to 2D and 3D)

  Returns: Squared EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape).astype(np.float32)

  if not data.flags['C_CONTIGUOUS'] and not data.flags['F_CONTIGUOUS']:
    data = np.copy(data, order=order)

  if parallel <= 0:
    parallel = multiprocessing.cpu_count()

  if dims == 1:
    anisotropy = nvl(anisotropy, 1.0)
    return edt1dsq(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = nvl(anisotropy, (1.0, 1.0))
    return edt2dsq(data, anisotropy, black_border, order, parallel=parallel)
  elif dims == 3:
    anisotropy = nvl(anisotropy, (1.0, 1.0, 1.0))
    return edt3dsq(data, anisotropy, black_border, order, parallel=parallel)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

def edt1d(data, anisotropy=1.0, bool black_border=False):
  result = edt1dsq(data, anisotropy, black_border)
  return np.sqrt(result, result)

def edt1dsq(data, anisotropy=1.0, bool black_border=False):
  cdef uint8_t[:] arr_memview8
  cdef uint16_t[:] arr_memview16
  cdef uint32_t[:] arr_memview32
  cdef uint64_t[:] arr_memview64
  cdef float[:] arr_memviewfloat
  cdef double[:] arr_memviewdouble

  cdef size_t voxels = data.size
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[uint8_t](
      <uint8_t*>&arr_memview8[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    squared_edt_1d_multi_seg[uint16_t](
      <uint16_t*>&arr_memview16[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    squared_edt_1d_multi_seg[uint32_t](
      <uint32_t*>&arr_memview32[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    squared_edt_1d_multi_seg[uint64_t](
      <uint64_t*>&arr_memview64[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    squared_edt_1d_multi_seg[float](
      <float*>&arr_memviewfloat[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    squared_edt_1d_multi_seg[double](
      <double*>&arr_memviewdouble[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[bool](
      <bool*>&arr_memview8[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  
  return output

def edt2d(
    data, anisotropy=(1.0, 1.0), 
    bool black_border=False, order='C', 
    parallel=1
  ):
  result = edt2dsq(data, anisotropy, black_border, order, parallel)
  return np.sqrt(result, result)

def edt2dsq(
    data, anisotropy=(1.0, 1.0), 
    bool black_border=False, order='C',
    parallel=1
  ):
  cdef uint8_t[:,:] arr_memview8
  cdef uint16_t[:,:] arr_memview16
  cdef uint32_t[:,:] arr_memview32
  cdef uint64_t[:,:] arr_memview64
  cdef float[:,:] arr_memviewfloat
  cdef double[:,:] arr_memviewdouble
  cdef bool[:,:] arr_memviewbool

  cdef size_t sx = data.shape[1] # C: rows
  cdef size_t sy = data.shape[0] # C: cols
  cdef float ax = anisotropy[1]
  cdef float ay = anisotropy[0]

  if order == 'F':
    sx = data.shape[0] # F: cols
    sy = data.shape[1] # F: rows
    ax = anisotropy[0]
    ay = anisotropy[1]

  cdef size_t voxels = sx * sy
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt2dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt2dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt2dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt2dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt2dsq[float](
      <float*>&arr_memviewfloat[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt2dsq[double](
      <double*>&arr_memviewdouble[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    _edt2dsq[bool](
      <bool*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0]      
    )

  return output.reshape( data.shape, order=order)

def edt3d(
    data, anisotropy=(1.0, 1.0, 1.0), 
    bool black_border=False, order='C', 
    parallel=1
  ):
  result = edt3dsq(data, anisotropy, black_border, order, parallel)
  return np.sqrt(result, result)

def edt3dsq(
    data, anisotropy=(1.0, 1.0, 1.0), 
    bool black_border=False, order='C',
    int parallel=1
  ):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef float[:,:,:] arr_memviewdouble

  cdef size_t sx = data.shape[2]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[0]
  cdef float ax = anisotropy[2]
  cdef float ay = anisotropy[1]
  cdef float az = anisotropy[0]

  if order == 'F':
    sx, sy, sz = sz, sy, sx
    ax = anisotropy[0]
    ay = anisotropy[1]
    az = anisotropy[2]

  cdef size_t voxels = sx * sy * sz
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt3dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt3dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt3dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt3dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt3dsq[float](
      <float*>&arr_memviewfloat[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt3dsq[double](
      <double*>&arr_memviewdouble[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.view(np.uint8)
    _edt3dsq[bool](
      <bool*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0]
    )

  return output.reshape( data.shape, order=order)
