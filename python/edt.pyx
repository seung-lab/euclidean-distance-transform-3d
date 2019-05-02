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

cimport numpy as numpy
from cpython cimport array 
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.2.4'

cdef extern from "edt.hpp" namespace "pyedt":
  cdef void squared_edt_1d_multi_seg[T](
    T *labels,
    float *dest,
    int n,
    int stride,
    float anisotropy,
    bool black_border
  )

  cdef float* _edt2dsq[T](
    T* labels,
    int sx, int sy, 
    float wx, float wy,
    bool black_border
  )

  cdef float* _edt3dsq[T](
    T* labels,
    int sx, int sy, int sz,
    float wx, float wy, float wz,
    bool black_border
  )

def nvl(val, default_val):
  if val is None:
    return default_val
  return val

def edt(data, anisotropy=None, black_border=False, order='C'):
  """
  edt(data, anisotropy=None, black_border=False, order='C')

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
    order: 'C' or 'F' interpret the input data as C (row major) 
      or Fortran (column major) order.

  Returns: EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape).astype(np.float32)

  if not data.flags['C_CONTIGUOUS'] and not data.flags['F_CONTIGUOUS']:
    data = np.copy(data, order=order)

  if dims == 1:
    anisotropy = nvl(anisotropy, 1.0)
    return edt1d(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = nvl(anisotropy, (1.0, 1.0))
    return edt2d(data, anisotropy, black_border, order)
  elif dims == 3:
    anisotropy = nvl(anisotropy, (1.0, 1.0, 1.0))
    return edt3d(data, anisotropy, black_border, order)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

def edtsq(data, anisotropy=None, bool black_border=False, order='C'):
  """
  edtsq(data, anisotropy=None, black_border=False, order='C')

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

  Returns: Squared EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape).astype(np.float32)

  if not data.flags['C_CONTIGUOUS'] and not data.flags['F_CONTIGUOUS']:
    data = np.copy(data, order=order)

  if dims == 1:
    anisotropy = nvl(anisotropy, 1.0)
    return edt1dsq(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = nvl(anisotropy, (1.0, 1.0))
    return edt2dsq(data, anisotropy, black_border, order)
  elif dims == 3:
    anisotropy = nvl(anisotropy, (1.0, 1.0, 1.0))
    return edt3dsq(data, anisotropy, black_border, order)
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
  
  cdef float* xform = <float*>calloc(data.size, sizeof(float))

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[uint8_t](
      <uint8_t*>&arr_memview8[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    squared_edt_1d_multi_seg[uint16_t](
      <uint16_t*>&arr_memview16[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    squared_edt_1d_multi_seg[uint32_t](
      <uint32_t*>&arr_memview32[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    squared_edt_1d_multi_seg[uint64_t](
      <uint64_t*>&arr_memview64[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    squared_edt_1d_multi_seg[float](
      <float*>&arr_memviewfloat[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    squared_edt_1d_multi_seg[double](
      <double*>&arr_memviewdouble[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[bool](
      <bool*>&arr_memview8[0],
      xform,
      data.size,
      1,
      anisotropy,
      black_border
    )
  
  cdef float[:] xform_view = <float[:data.size]>xform
  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(xform_view[:])
  free(xform)
  return np.frombuffer(buf, dtype=np.float32)

def edt2d(data, anisotropy=(1.0, 1.0), bool black_border=False, order='C'):
  result = edt2dsq(data, anisotropy, black_border, order)
  return np.sqrt(result, result)

def edt2dsq(data, anisotropy=(1.0, 1.0), bool black_border=False, order='C'):
  cdef uint8_t[:,:] arr_memview8
  cdef uint16_t[:,:] arr_memview16
  cdef uint32_t[:,:] arr_memview32
  cdef uint64_t[:,:] arr_memview64
  cdef float[:,:] arr_memviewfloat
  cdef double[:,:] arr_memviewdouble
  cdef bool[:,:] arr_memviewbool

  cdef float* xform

  cdef int sx = data.shape[1] # C: rows
  cdef int sy = data.shape[0] # C: cols
  cdef float ax = anisotropy[1]
  cdef float ay = anisotropy[0]

  if order == 'F':
    sx = data.shape[0] # F: cols
    sy = data.shape[1] # F: rows
    ax = anisotropy[0]
    ay = anisotropy[1]

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    xform = _edt2dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border   
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    xform = _edt2dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    xform = _edt2dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    xform = _edt2dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    xform = _edt2dsq[float](
      <float*>&arr_memviewfloat[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    xform = _edt2dsq[double](
      <double*>&arr_memviewdouble[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    xform = _edt2dsq[bool](
      <bool*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border      
    )

  cdef float[:] xform_view = <float[:data.size]>xform
  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(xform_view[:])
  free(xform)
  return np.frombuffer(buf, dtype=np.float32).reshape( data.shape, order=order)

def edt3d(data, anisotropy=(1.0, 1.0, 1.0), bool black_border=False, order='C'):
  result = edt3dsq(data, anisotropy, black_border, order)
  return np.sqrt(result, result)

def edt3dsq(data, anisotropy=(1.0, 1.0, 1.0), bool black_border=False, order='C'):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef float[:,:,:] arr_memviewdouble

  cdef float* xform

  cdef int sx = data.shape[2]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[0]
  cdef float ax = anisotropy[2]
  cdef float ay = anisotropy[1]
  cdef float az = anisotropy[0]

  if order == 'F':
    sx, sy, sz = sz, sy, sx
    ax = anisotropy[0]
    ay = anisotropy[1]
    az = anisotropy[2]

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    xform = _edt3dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    xform = _edt3dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    xform = _edt3dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    xform = _edt3dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    xform = _edt3dsq[float](
      <float*>&arr_memviewfloat[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    xform = _edt3dsq[double](
      <double*>&arr_memviewdouble[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    xform = _edt3dsq[bool](
      <bool*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border
    )

  cdef float[:] xform_view = <float[:data.size]>xform
  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(xform_view[:])
  free(xform)
  return np.frombuffer(buf, dtype=np.float32).reshape( data.shape, order=order)

