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
Date: July 2018
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

__VERSION__ = '1.0.0'

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
    int wx, int wy,
    bool black_border
  )

  cdef float* _edt3dsq[T](
    T* labels,
    int sx, int sy, int sz,
    int wx, int wy, int wz,
    bool black_border
  )

def edt(data, anisotropy=None, black_border=False):
  """
  edt(data, anisotropy=None, black_border=False)

  Computes the anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supports uint8, uint16, uint32, and uint64 data types.

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.

  Returns: EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape).astype(np.float32)

  if dims == 1:
    anisotropy = anisotropy or 1.0
    return edt1d(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = anisotropy or (1.0, 1.0)
    return edt2d(data, anisotropy, black_border)
  elif dims == 3:
    anisotropy = anisotropy or (1.0, 1.0, 1.0)
    return edt3d(data, anisotropy, black_border)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

def edtsq(data, anisotropy=None, bool black_border=False):
  """
  edtsq(data, anisotropy=None, black_border=False)

  Computes the squared anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  Squaring allows for omitting an sqrt operation, so may be faster if your use case allows for it.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supports uint8, uint16, uint32, and uint64 data types.

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.

  Returns: Squared EDT of data
  """
  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape).astype(np.float32)

  if dims == 1:
    anisotropy = anisotropy or 1.0
    return edt1dsq(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = anisotropy or (1.0, 1.0)
    return edt2dsq(data, anisotropy, black_border)
  elif dims == 3:
    anisotropy = anisotropy or (1.0, 1.0, 1.0)
    return edt3dsq(data, anisotropy, black_border)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

def edt1d(data, anisotropy=1.0, bool black_border=False):
  return np.sqrt(edt1dsq(data, anisotropy, black_border))

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

def edt2d(data, anisotropy=(1.0, 1.0), bool black_border=False):
  return np.sqrt(edt2dsq(data, anisotropy, black_border))

def edt2dsq(data, anisotropy=(1.0, 1.0), bool black_border=False):
  cdef uint8_t[:,:] arr_memview8
  cdef uint16_t[:,:] arr_memview16
  cdef uint32_t[:,:] arr_memview32
  cdef uint64_t[:,:] arr_memview64
  cdef float[:,:] arr_memviewfloat
  cdef double[:,:] arr_memviewdouble
  cdef bool[:,:] arr_memviewbool

  cdef float* xform

  cdef int rows = data.shape[0]
  cdef int cols = data.shape[1]

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    xform = _edt2dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border   
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    xform = _edt2dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    xform = _edt2dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    xform = _edt2dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    xform = _edt2dsq[float](
      <float*>&arr_memviewfloat[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    xform = _edt2dsq[double](
      <double*>&arr_memviewdouble[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    xform = _edt2dsq[bool](
      <bool*>&arr_memview8[0,0],
      cols, rows,
      anisotropy[0], anisotropy[1],
      black_border      
    )

  cdef float[:] xform_view = <float[:data.size]>xform
  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(xform_view[:])
  free(xform)
  order = 'F' if data.flags['F_CONTIGUOUS'] else 'C'
  return np.frombuffer(buf, dtype=np.float32).reshape( data.shape, order=order)

def edt3d(data, anisotropy=(1.0, 1.0, 1.0), bool black_border=False):
  return np.sqrt(edt3dsq(data, anisotropy, black_border))

def edt3dsq(data, anisotropy=(1.0, 1.0, 1.0), bool black_border=False):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef float[:,:,:] arr_memviewdouble

  cdef float* xform

  cdef int depth = data.shape[0]
  cdef int rows = data.shape[1]
  cdef int cols = data.shape[2]

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    xform = _edt3dsq[uint8_t](
      <uint8_t*>&arr_memview8[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    xform = _edt3dsq[uint16_t](
      <uint16_t*>&arr_memview16[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    xform = _edt3dsq[uint32_t](
      <uint32_t*>&arr_memview32[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    xform = _edt3dsq[uint64_t](
      <uint64_t*>&arr_memview64[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    xform = _edt3dsq[float](
      <float*>&arr_memviewfloat[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    xform = _edt3dsq[double](
      <double*>&arr_memviewdouble[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )
  elif data.dtype == np.bool:
    arr_memview8 = data.astype(np.uint8)
    xform = _edt3dsq[bool](
      <bool*>&arr_memview8[0,0,0],
      cols, rows, depth,
      anisotropy[0], anisotropy[1], anisotropy[2],
      black_border
    )

  cdef float[:] xform_view = <float[:data.size]>xform
  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(xform_view[:])
  free(xform)

  order = 'F' if data.flags['F_CONTIGUOUS'] else 'C'
  return np.frombuffer(buf, dtype=np.float32).reshape( data.shape, order=order)

