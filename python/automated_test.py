import pytest

import math

import edt
import numpy as np
from scipy import ndimage

INTEGER_TYPES = [
  np.uint8, np.uint16, np.uint32, np.uint64,
]

TYPES_NO_BOOL = INTEGER_TYPES + [ np.float32 ]

TYPES = TYPES_NO_BOOL + [ bool ]

@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("parallel", (1,2))
def test_one_d_simple(dtype, parallel):
  labels = np.array([ 0 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)

  result = edt.edt(labels, black_border=False, parallel=parallel)
  assert np.all(result == labels)

  labels = np.array([ 1 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)

  result = edt.edt(labels, black_border=False, parallel=parallel)
  assert np.all(result == np.array([ np.inf ]))

  labels = np.array([ 0, 1 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)

  result = edt.edt(labels, black_border=False, parallel=parallel)
  assert np.all(result == labels)

  labels = np.array([ 1, 0 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)

  result = edt.edt(labels, black_border=False, parallel=parallel)
  assert np.all(result == labels)

  labels = np.array([ 0, 1, 0 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)  

  result = edt.edt(labels, black_border=False, parallel=parallel)
  assert np.all(result == labels)  

  labels = np.array([ 0, 1, 1, 0 ], dtype=dtype)
  result = edt.edt(labels, black_border=True, parallel=parallel)
  assert np.all(result == labels)  

  result = edt.edt(labels, black_border=False, parallel=parallel)
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
    for parallel in (1,2):
      for dtype in types:
        print(dtype)
        labels = np.array(labels, dtype=dtype)
        ans = np.array(ans, dtype=np.float32)
        result = edt.edtsq(
          labels, anisotropy=anisotropy, 
          black_border=True, parallel=parallel
        )
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
      [  25,  25,  25,  25,  25 ], 
      [  36, 100, 100, 100,  36 ], 
      [  36, 144, 225, 144,  36 ], 
      [  36, 100, 100, 100,  36 ], 
      [  25,  25,  25,  25,  25 ], 
    ],
    anisotropy=(5.0, 6.0)
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
  for dtype in INTEGER_TYPES:
    print(dtype)
    randos = np.random.randint(0, 2, size=(3, 3), dtype=dtype)
    labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2), dtype=dtype)
    # Scipy requires zero borders
    labels[1:-1,1:-1] = randos

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
    for parallel in (1,2):
      for dtype in (np.uint32, bool):
        randos = np.random.randint(0, 2, size=(5, 5), dtype=dtype)
        labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2), dtype=dtype)

        print("INPUT")
        print(labels)

        print("MLAEDT")
        mlaedt_result = edt.edt(labels, black_border=False, parallel=parallel)
        print(mlaedt_result)

        print("SCIPY")
        scipy_result = ndimage.distance_transform_edt(labels)
        print(scipy_result)

        assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )

def test_three_d():  
  def cmp(labels, ans, types=TYPES, anisotropy=(1.0, 1.0, 1.0)):
    for parallel in (1,2):
      for dtype in types:
        print(dtype, anisotropy)
        labels = np.array(labels, dtype=dtype)
        ans = np.array(ans, dtype=np.float32)
        print(labels)
        print(ans)
        result = edt.edtsq(
          labels, anisotropy=anisotropy, 
          black_border=True, 
          parallel=parallel
        )
        assert np.all(result.T == ans) # written in human understandable order so needs transpose 

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
      [16, 16, 16],
      [16, 16, 16]
    ],
    [
      [16, 16, 16], 
      [16, 64, 16],
      [16, 16, 16]
    ],
    [
      [16, 16, 16], 
      [16, 16, 16],
      [16, 16, 16]
    ],
  ], anisotropy=(4,4,4))

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
      [25, 25, 25], 
      [25, 25, 25],
      [25, 25, 25]
    ],
    [
      [36, 36, 36], 
      [36,100, 36],
      [36, 36, 36]
    ],
    [
      [25, 25, 25], 
      [25, 25, 25],
      [25, 25, 25]
    ],
  ], anisotropy=(6,6,5))

@pytest.mark.parametrize("order", ("C", "F"))
@pytest.mark.parametrize("parallel", (1,2))
@pytest.mark.parametrize("dtype", (np.uint32, bool))
def test_3d_scipy_comparison(dtype, parallel, order):
  for _ in range(5):
    randos = np.random.randint(0, 2, size=(100, 100, 100), dtype=dtype)
    labels = np.zeros( (randos.shape[0] + 2, randos.shape[1] + 2, randos.shape[2] + 2), dtype=dtype, order=order)
    # Scipy requires zero borders
    labels[1:-1,1:-1,1:-1] = randos

    print("INPUT")
    print(labels)

    print("MLAEDT")
    mlaedt_result = edt.edt(labels, black_border=False, parallel=parallel)
    print(mlaedt_result)

    print("SCIPY")
    scipy_result = ndimage.distance_transform_edt(labels)
    print(scipy_result)

    print("DIFF")
    print(np.abs(scipy_result == mlaedt_result))
    print(np.max(np.abs(scipy_result - mlaedt_result)))

    assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )

def test_non_mutation_2d():
  """
  This example helped debug the error 
  caused by reading/writing to the same array.
  """
  x = np.array(
  [
   [  True, False,  True,  True,  ],
   [ False,  True,  True,  True,  ],
   [ False,  True,  True,  True,  ],
   [  True,  True,  True,  True,  ],
   [ False,  True,  True,  True,  ],], dtype=bool)
 
  compare_scipy_edt(x)

def test_dots(numdots=5, N=100, radius=20):
  img = np.zeros((N, N), dtype=bool)
  locations=np.random.randint(0, N-1, size=(numdots, 2), dtype=int)
  xx,yy = np.meshgrid(range(N), range(N), indexing='xy')

  for loc in locations:
    dx = xx - loc[0]
    dy = yy - loc[1]
    d = np.sqrt(dx ** 2 + dy ** 2)
    img[d <= radius] = True

  img[ :, 0] = 0
  img[ 0, :] = 0
  img[-1, :] = 0
  img[ :,-1] = 0

  compare_scipy_edt(img)

def compare_scipy_edt(labels):
  print("INPUT", labels.shape)
  print(labels)

  print("MLAEDT")
  mlaedt_result = edt.edt(labels, black_border=False)
  print(mlaedt_result)

  print("SCIPY")
  scipy_result = ndimage.distance_transform_edt(labels)
  print(scipy_result)

  print("DIFF")
  print(np.abs(scipy_result - mlaedt_result) < 0.000001)
  print("MAX Diff")
  print(np.max(np.abs(scipy_result - mlaedt_result)))

  assert np.all( np.abs(scipy_result - mlaedt_result) < 0.000001 )

def test_2d_even_anisotropy():
  labels = np.zeros( (15,15), dtype=bool, order='F')
  labels[2:12, 2:12] = True
  img = edt.edt(labels, anisotropy=(1,1))
  for i in range(1, 150):
    w = float(i)
    aimg = edt.edt(labels, anisotropy=(w, w))
    assert np.all(w * img == aimg)

def test_3d_even_anisotropy():
  labels = np.zeros( (15,15,15), dtype=bool, order='F')
  labels[2:12, 2:12, 5:10] = True
  img = edt.edt(labels, anisotropy=(1,1,1))
  for parallel in (1,2):
    for i in range(1, 150):
      w = float(i)
      aimg = edt.edt(labels, anisotropy=(w, w, w), parallel=parallel)
      assert np.all(w * img == aimg)

def test_2d_lopsided():
  def gen(x, y, order):
    x = np.zeros((x, y), dtype=np.uint32, order=order)
    x[0:25,5:50] = 3
    x[25:50,5:50] = 1
    x[60:110,5:50] = 2
    return x

  sizes = [
    (150, 150),
    (150,  75),
    (75,  150),
  ]

  for size in sizes:
    cres = edt.edt(gen(size[0], size[1], 'C'))
    fres = edt.edt(gen(size[0], size[1], 'F'))

    print(size)
    assert np.all(cres[:] == fres[:])

@pytest.mark.parametrize("size", [ (150, 150), (150, 75), (75, 150)])
def test_2d_lopsided_anisotropic(size):
  def gen(x, y, order):
    x = np.zeros((x, y), dtype=np.uint32, order=order)
    x[0:25,5:50] = 3
    x[25:50,5:50] = 1
    x[60:110,5:50] = 2
    return x

  cres = edt.edt(gen(size[0], size[1], 'C'), anisotropy=(2,3))
  fres = edt.edt(gen(size[0], size[1], 'F'), anisotropy=(2,3))
  assert np.all(np.isclose(cres, fres))

@pytest.mark.parametrize("size", [     
    (150, 150, 150),
    (150,  75,  23),
    (75,  150,  37),
])
def test_3d_lopsided(size):
  def gen(x, y, z, order):
    x = np.zeros((x, y, z), dtype=np.uint32, order=order)
    x[ 0:25,  5:50, 0:25] = 3
    x[25:50,  5:50, 0:25] = 1
    x[60:110, 5:50, 0:25] = 2
    return x

  cres = edt.edt(gen(size[0], size[1], size[2], 'C'))
  fres = edt.edt(gen(size[0], size[1], size[2], 'F'))
  assert np.all(np.isclose(cres, fres))

def test_3d_high_anisotropy():
  shape = (256, 256, 256)
  anisotropy = (1000000, 1200000, 40)

  labels = np.ones( shape, dtype=np.uint8)
  labels[0, 0, 0] = 0
  labels[-1, -1, -1] = 0

  resedt = edt.edt(labels, anisotropy=anisotropy, black_border=False)

  mx = np.max(resedt)
  assert np.isfinite(mx)
  assert mx <= (1e6 * 256) ** 2 + (1e6 * 256) ** 2 + (666 * 256) ** 2

  resscipy = ndimage.distance_transform_edt(labels, sampling=anisotropy)
  resscipy[ resscipy == 0 ] = 1
  resedt[ resedt == 0 ] = 1
  ratio = np.abs(resscipy / resedt)
  assert np.all(ratio < 1.000001) and np.all(ratio > 0.999999)

def test_all_inf():
  shape = (128, 128, 128)
  labels = np.ones( shape, dtype=np.uint8)
  res = edt.edt(labels, black_border=False, anisotropy=(1,1,1))
  assert np.all(res == np.inf)

def test_numpy_anisotropy():
  labels = np.zeros(shape=(128, 128, 128), dtype=np.uint32)
  labels[1:-1,1:-1,1:-1] = 1

  resolution = np.array([4,4,40])
  res = edt.edtsq(labels, anisotropy=resolution)

def test_voxel_connectivity_graph_2d():
  labels = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
  ])

  omni = 0b111111
  noxf = 0b111110
  noxb = 0b111101

  graph = np.array([
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
  ], dtype=np.uint8)

  dt = edt.edt(labels, voxel_graph=graph)
  assert np.all(dt == np.inf)

  dt = edt.edt(labels, voxel_graph=graph, black_border=True)
  assert np.all(dt == np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 1.5, 1.5, 1.5, 1.5, 0.5],
    [0.5, 1.5, 2.5, 2.5, 1.5, 0.5],
    [0.5, 1.5, 1.5, 1.5, 1.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  ]))

  graph = np.array([
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, noxf, noxb, omni, omni],
    [omni, omni, omni, omni, omni, omni],
    [omni, omni, omni, omni, omni, omni],
  ], dtype=np.uint8, order="C")
  dt = edt.edt(labels, voxel_graph=graph, black_border=True)

  ans = np.array([
    [1,        1,        1,        1,        1,        1],
    [1,        1.8027756,1.118034, 1.118034, 1.8027756,1],
    [1,        1.5,      0.5,      0.5,      1.5,      1],
    [1,        1.8027756,1.118034, 1.118034, 1.8027756,1],
    [1,        1,        1,        1,        1,        1]
  ])
  assert np.all(np.abs(dt - ans)) < 0.000002

  graph = np.asfortranarray(graph)
  dt = edt.edt(labels, voxel_graph=graph, black_border=True)
  assert np.all(np.abs(dt - ans)) < 0.000002

def test_small_anisotropy():
  d = np.array([
    [True, True ], 
    [True, False],
  ])
  res = edt.edt(d, anisotropy=[0.5, 0.5], black_border=False)

  assert np.all(np.isclose(res, [[np.sqrt(2) / 2, 0.5],[0.5, 0.0]]))

@pytest.mark.parametrize("weight", [
  0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 
  1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000.
])
def test_anisotropy_range(weight):
  img = np.ones((100,97,99), dtype=np.uint8)
  img[0,0,0] = 0

  res = edt.edt(img, anisotropy=(weight, weight, weight), black_border=False)

  sx = weight * (img.shape[0] - 1)
  sy = weight * (img.shape[1] - 1)
  sz = weight * (img.shape[2] - 1)

  max_val = res[99, 96, 98]
  expected = math.sqrt(sx*sx + sy*sy + sz*sz)

  assert math.isclose(expected, max_val, rel_tol=0.000001)

def test_nan_large_array():
  arr = np.ones((46342, 1))
  arr[0, 0] = 0
  res = edt.edt(arr)
  assert not np.any(np.isnan(res))

def test_column_off_by_one():
  img = np.array([[1,1], [2,1]], dtype=np.int64, order="C")
  res = edt.edt(img)
  ans = np.array([[1, 1.41421],[1, 1]], dtype=np.float32)
  assert np.all(np.isclose(res, ans))

@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("in_place", [ True, False ])
def test_each(dtype, in_place):
  labels = np.zeros((64,64,64), dtype=dtype)
  half = 32
  i = 1
  for x in range(2):
    for y in range(2):
      for z in range(2):
        labels[x*half:(x+1)*half, y*half:(y+1)*half, z*half:(z+1)*half ] = i
        i += 1

  mdt = edt.edt(labels) # multiple dt

  for label, dt in edt.each(labels, mdt, in_place=in_place):
    single = (labels == label) * mdt
    assert np.all(single == dt), label

@pytest.mark.parametrize("in_place", [ True, False ])
def test_each_random(in_place):
  labels = np.random.randint(0,75, size=(62,65,69))
  mdt = edt.edt(labels) # multiple dt

  for label, dt in edt.each(labels, mdt, in_place=in_place):
    single = (labels == label) * mdt
    assert np.all(single == dt), label  

def test_zero_trailing_2d():
  labels = np.array([[
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]], dtype=np.uint8)

  ans = np.array([[
    [9, 4, 1, 0],
    [4, 4, 2, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]], dtype=np.uint8)

  assert np.all(ans == edt.edtsq(labels))

@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_sdf(dtype):
  labels = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ], dtype=dtype)

  ans = edt.edt(labels) - edt.edt(labels == 0)
  res = edt.sdf(labels)
  assert np.all(res == ans)





