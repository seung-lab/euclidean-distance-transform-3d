import pytest

import edt
import numpy as np

def test_one_d_identity():
  labels = np.array([ 0 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)

  labels = np.array([ 1 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)

  labels = np.array([ 0, 1 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)

  labels = np.array([ 1, 0 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)


  labels = np.array([ 0, 1, 0 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)  

  labels = np.array([ 0, 1, 1, 0 ], dtype=np.uint32)
  result = edt.edt(labels)
  assert np.all(result == labels)  

def test_one_d():
  def cmp(labels, ans, anisotropy=1.0):
    labels = np.array(labels, dtype=np.uint32)
    ans = np.array(ans, dtype=np.float32)
    result = edt.edtsq(labels, anisotropy=anisotropy)    
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
    [ 1, 4, 9, 4, 1, 0, 1, 4, 9, 4, 1, 1, 4, 4, 1, 1 ]
  )

def test_two_d_ident():  
  def cmp(labels, ans, anisotropy=(1.0, 1.0)):
    labels = np.array(labels, dtype=np.uint32)
    ans = np.array(ans, dtype=np.float32)
    result = edt.edtsq(labels, anisotropy=anisotropy)    
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
  def cmp(labels, ans, anisotropy=(1.0, 1.0)):
    labels = np.array(labels, dtype=np.uint32)
    ans = np.array(ans, dtype=np.float32)
    result = edt.edtsq(labels, anisotropy=anisotropy)
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
    ]
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
    ]
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
    ])

  labels = np.ones( (5, 6), dtype=np.uint32)
  labels[3:,:] = 2 # rows 4-5 = 2

  cmp(labels, 
    [
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 4, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1 ], 
    ]
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
    ]
  )

  cmp(labels, 
    [
      [ 0, 0, 0, 0, 0, 0, 0 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
      [ 1, 4, 4, 4, 2, 1, 1 ], 
      [ 1, 4, 4, 4, 1, 1, 1 ], 
      [ 1, 1, 1, 1, 1, 1, 1 ], 
    ]
  )

def test_three_d():  
  def cmp(labels, ans, anisotropy=(1.0, 1.0, 1.0)):
    labels = np.array(labels, dtype=np.uint32)
    ans = np.array(ans, dtype=np.float32)
    result = edt.edtsq(labels, anisotropy=anisotropy)    
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
