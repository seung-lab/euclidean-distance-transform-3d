import time

import numpy as np
from scipy import ndimage

import edt

labels = np.ones(shape=(512, 512, 512), dtype=np.uint32)

start = time.time()
res = edt.edtsq(labels, anisotropy=(1,1,1))

for i in range(300):
  x = (labels == i) * res

print('Multi-label EDT: ', time.time() - start, ' sec.')

start = time.time()
ndimage.distance_transform_edt(labels)
print('ndimage EDT: ', time.time() - start, ' sec.')

