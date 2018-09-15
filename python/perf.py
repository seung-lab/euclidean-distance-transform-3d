import time

import numpy as np
from scipy import ndimage

import edt

labels = np.ones(shape=(512, 512, 512), dtype=np.uint32)

start = time.time()
res = edt.edtsq(labels, anisotropy=(1,1,1))
print('Multi-label EDT: ', time.time() - start, ' sec.')

binlabels = labels.astype(np.bool)

start = time.time()
res = edt.edtsq(binlabels, anisotropy=(1,1,1))
print('Binary EDT: ', time.time() - start, ' sec.')

start = time.time()
ndimage.distance_transform_edt(labels)
print('ndimage EDT: ', time.time() - start, ' sec.')

