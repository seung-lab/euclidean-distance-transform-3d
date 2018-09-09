import pytest

import edt
import numpy as np
from scipy import ndimage

INTEGER_TYPES = [
  np.uint8, np.uint16, np.uint32, np.uint64,
]

TYPES_NO_BOOL = INTEGER_TYPES + [ np.float32 ]

TYPES = TYPES_NO_BOOL + [ np.bool ]

def test_one_d_simple():
  for dtype in TYPES:
    print(dtype)
    labels = np.array([ 0 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)

    result = edt.edt(labels, black_border=False)
    assert np.all(result == labels)

    labels = np.array([ 1 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)

    result = edt.edt(labels, black_border=False)
    assert np.all(result == np.array([ np.inf ]))

    labels = np.array([ 0, 1 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)

    result = edt.edt(labels, black_border=False)
    assert np.all(result == labels)

    labels = np.array([ 1, 0 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)

    result = edt.edt(labels, black_border=False)
    assert np.all(result == labels)

    labels = np.array([ 0, 1, 0 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)  

    result = edt.edt(labels, black_border=False)
    assert np.all(result == labels)  

    labels = np.array([ 0, 1, 1, 0 ], dtype=dtype)
    result = edt.edt(labels, black_border=True)
    assert np.all(result == labels)  

    result = edt.edt(labels, black_border=False)
    assert np.all(result == labels)  

def test_one_d_black_border():
  def cmp(labels, ans, types=TYPES, anisotropy=1.0):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=True)
      assert np.all(result == ans)  

  cmp([], [])

  cmp([1], [1])

  cmp([5], [1])

  cmp(
    [ 0, 1, 1, 1, 0 ],
    [ 0, 1, 4, 1, 0 ]
  )

  cmp(
    [ 1, 1, 1, 1 ],
    [ 1, 4, 4, 1 ]
  )

  cmp(
    [ 1, 1, 1, 1 ],
    [ 4, 16, 16, 4 ],
    anisotropy=2.0
  )

  cmp(
    [ 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 3 ],
    [ 1, 4, 9, 4, 1, 0, 1, 4, 9, 4, 1, 1, 4, 4, 1, 1 ],
    types=TYPES_NO_BOOL,
  )

def test_one_d():
  def cmp(labels, ans, types=TYPES, anisotropy=1.0):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=False)
      assert np.all(result == ans)  

  inf = np.inf

  cmp([], [])

  cmp([1], [inf])

  cmp([5], [inf])

  cmp(
    [ 0, 1, 1, 1, 0 ],
    [ 0, 1, 4, 1, 0 ]
  )

  cmp(
    [ 0, 1, 1, 1,  1 ],
    [ 0, 1, 4, 9, 16 ]
  )

  cmp(
    [  1, 1, 1, 1, 0 ],
    [ 16, 9, 4, 1, 0 ]
  )

  cmp(
    [ 1, 1, 1, 1 ],
    [ inf, inf, inf, inf ]
  )

  cmp(
    [ 1, 1, 1, 1 ],
    [ inf, inf, inf, inf ],
    anisotropy=2.0
  )

  cmp(
    [  1,  1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 3 ],
    [ 25, 16, 9, 4, 1, 0, 1, 4, 9, 4, 1, 1, 4, 4, 1, 1 ],
    types=TYPES_NO_BOOL,
  )

def test_1d_scipy_comparison():
  for _ in range(20):
    randos = np.random.randint(0, 2, size=(100), dtype=np.uint32)
    labels = np.zeros( (randos.shape[0] + 2,), dtype=np.uint32)
    # Scipy requires zero borders
    labels[1:-1] = randos

    print("INPUT")
    print(labels)

    print("MLAEDT")
    mlaedt_result_bb = edt.edt(labels, black_border=True)
    mlaedt_result = edt.edt(labels, black_border=True)
    print(mlaedt_result)

    print("SCIPY")
    scipy_result = ndimage.distance_transform_edt(labels)
    print(scipy_result)

    assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )
    assert np.all( np.abs(scipy_result - mlaedt_result_bb) < 0.000001 )

def test_1d_scipy_comparison_no_border():
  for _ in range(20):
    randos = np.random.randint(0, 2, size=(100), dtype=np.uint32)
    labels = np.zeros( (randos.shape[0] + 2,), dtype=np.uint32)

    print("INPUT")
    print(labels)

    print("MLAEDT")
    mlaedt_result = edt.edt(labels, black_border=False)
    print(mlaedt_result)

    print("SCIPY")
    scipy_result = ndimage.distance_transform_edt(labels)
    print(scipy_result)

    assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )

def test_two_d_ident_no_border():  
  def cmp(labels, ans, types=TYPES, anisotropy=(1.0, 1.0)):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=False)
      assert np.all(result == ans)  

  I = np.inf

  cmp([[]], [[]])
  cmp([[0]], [[0]])
  cmp([[1]], [[I]])
  cmp([[1, 0], [0, 1]], [[1, 0], [0, 1]])

  cmp([[1, 1], [1, 1]], [[I, I], [I, I]])

  cmp(
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 
    [[I, I, I, I, I], [I, I, I, I, I]]
  )


def test_two_d_ident_black_border():  
  def cmp(labels, ans, types=TYPES, anisotropy=(1.0, 1.0)):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=True)
      assert np.all(result == ans)  

  cmp([[]], [[]])
  cmp([[0]], [[0]])
  cmp([[1]], [[1]])
  cmp([[1, 1], [1, 1]], [[1, 1], [1, 1]])
  cmp([[1, 0], [0, 1]], [[1, 0], [0, 1]])
  
  cmp(
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
  )

def test_two_d():  
  def cmp(labels, ans, types=TYPES, anisotropy=(1.0, 1.0)):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=True)
      print(result)
      assert np.all(result == ans)  

  cmp(
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ], 
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 1 ], 
      [ 1, 4, 9, 4, 1 ], 
      [ 1, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ]
  )

  cmp(
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ], 
    [
      [ 25, 100, 100, 100, 25 ], 
      [ 25, 100, 225, 100, 25 ], 
      [ 25, 100, 225, 100, 25 ], 
      [ 25, 100, 225, 100, 25 ], 
      [ 25, 100, 100, 100, 25 ], 
    ],
    anisotropy=(5.0, 10.0)
  )

  cmp(
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 0, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ], 
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 2, 1, 2, 1 ], 
      [ 1, 1, 0, 1, 1 ], 
      [ 1, 2, 1, 2, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ]
  )

  cmp(
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 2, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ], 
    [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 2, 1, 2, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 2, 1, 2, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ],
    types=TYPES_NO_BOOL
  )
  
  cmp(
    [
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 2, 2, 2, 2, 2, 2 ], 
      [ 2, 2, 2, 2, 2, 2 ], 
      [ 2, 2, 2, 2, 2, 2 ], 
    ], 
    [
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ], 
    ],
    types=TYPES_NO_BOOL
  )

  labels = np.ones( (6, 5), dtype=np.uint32)
  labels[3:,:] = 2 # rows 3-6 = 2

  cmp(labels, [
      [ 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1 ],
      [ 1, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1 ], 
    ],
    types=TYPES_NO_BOOL
  )

  labels = np.ones( (5, 6), dtype=np.uint32)
  labels[3:,:] = 2 # rows 4-5 = 2

  cmp(labels, 
    [
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1 ], 
    ],
    types=TYPES_NO_BOOL
  )

  labels = np.ones( (7, 7), dtype=np.uint32)
  labels[0,:] = 0 
  labels[1:3,:] = 1 
  labels[3:,:] = 2 
  labels[5,5] = 3

  cmp(labels, 
    [
      [ 0, 0, 0, 0, 0, 0, 0 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 2, 1, 1 ], 
      [ 1, 4, 4, 4, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
    ],
    types=TYPES_NO_BOOL
  )

def test_2d_scipy_comparison_black_border():
  for dtype in (np.uint8,):
    print(dtype)
    # randos = np.random.randint(0, 2, size=(3, 3), dtype=dtype)
    # labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2), dtype=dtype)
    # # Scipy requires zero borders
    # labels[1:-1,1:-1] = randos
    labels = np.array([
      [0,0,0,0],
      [0,1,1,1],
      [0,0,1,0]
    ])#, dtype=np.uint8)
    print(labels.dtype)

    print("INPUT")
    print(labels)

    print("MLAEDT")
    mlaedt_result = edt.edt(labels, black_border=False)
    # mlaedt_result_bb = edt.edt(labels, black_border=True)
    print(mlaedt_result)

    print("SCIPY")
    scipy_result = ndimage.distance_transform_edt(labels)
    print(scipy_result)

    assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )
    # assert np.all( np.abs(scipy_result - mlaedt_result_bb) < 0.000001 )

def test_2d_scipy_comparison():
  for _ in range(20):
    randos = np.random.randint(0, 2, size=(5, 5), dtype=np.uint32)
    labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2), dtype=np.uint32)

    print("INPUT")
    print(labels)

    print("MLAEDT")
    mlaedt_result = edt.edt(labels, black_border=False)
    print(mlaedt_result)

    print("SCIPY")
    scipy_result = ndimage.distance_transform_edt(labels)
    print(scipy_result)

    assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )

def test_three_d():  
  def cmp(labels, ans, types=TYPES, anisotropy=(1.0, 1.0, 1.0)):
    for dtype in types:
      print(dtype)
      labels = np.array(labels, dtype=dtype)
      ans = np.array(ans, dtype=np.float32)
      result = edt.edtsq(labels, anisotropy=anisotropy, black_border=True)
      assert np.all(result == ans)  

  cmp([[[]]], [[[]]])
  cmp([[[0]]], [[[0]]])
  cmp([[[1]]], [[[1]]])
  cmp([[[5]]], [[[1]]])

  cmp([
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
  ], 
  [
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 4, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
  ])


  cmp([
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
  ], 
  [
    [
      [16, 16, 16], 
      [16, 64, 16],
      [16, 16, 16]
    ],
    [
      [16, 16, 16], 
      [16, 64, 16],
      [16, 16, 16]
    ],
    [
      [16, 16, 16], 
      [16, 64, 16],
      [16, 16, 16]
    ],
  ], anisotropy=(4,4,40))

  cmp([
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
    [
      [1, 1, 1], 
      [1, 1, 1],
      [1, 1, 1]
    ],
  ], 
  [
    [
      [16, 16, 16], 
      [16, 16, 16],
      [16, 16, 16]
    ],
    [
      [16, 64, 16], 
      [16, 64, 16],
      [16, 64, 16]
    ],
    [
      [16, 16, 16], 
      [16, 16, 16],
      [16, 16, 16]
    ],
  ], anisotropy=(4,40,4))

def test_3d_scipy_comparison():
  randos = np.random.randint(0, 2, size=(10, 10, 10), dtype=np.uint32)
  labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2, randos.shape[2] + 2), dtype=np.uint32)
  # Scipy requires zero borders
  labels[1:-1,1:-1,1:-1] = randos

  print("INPUT")
  print(labels)

  print("MLAEDT")
  mlaedt_result = edt.edt(labels, black_border=False)
  print(mlaedt_result)

  print("SCIPY")
  scipy_result = ndimage.distance_transform_edt(labels)
  print(scipy_result)

  print("DIFF")
  print(np.abs(scipy_result == mlaedt_result))
  print(np.max(np.abs(scipy_result - mlaedt_result)))

  assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )