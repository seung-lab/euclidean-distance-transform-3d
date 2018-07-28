import edt
import numpy as np
from scipy import ndimage

import time




# for d in (np.uint8, np.uint16, np.uint32, np.uint64):
labels = np.ones(shape=(8196, 8196), dtype=np.uint64)
start = time.time()
res = edt.edtsq(labels, anisotropy=(1,1))

# for i in range(300):
#   x = (labels == i) * res

print('Multi-label EDT: ', time.time() - start, ' sec.')


# x = np.ones((512,512,512), dtype=np.int32)

# x[1,1,1] = 0
start = time.time()
ndimage.distance_transform_edt(labels)
print('ndimage EDT: ', time.time() - start, ' sec.')

