import edt
import numpy as np
from scipy import ndimage

import time




# for d in (np.uint8, np.uint16, np.uint32, np.uint64):
labels = np.ones(shape=(512,512,512), dtype=np.uint16)
start = time.time()
print(edt.edt(labels))
print('Multi-label EDT: ', time.time() - start, ' sec.')


# x = np.ones((512,512,512), dtype=np.int32)

# x[1,1,1] = 0
start = time.time()
ndimage.distance_transform_edt(x)
print('ndimage EDT: ', time.time() - start, ' sec.')

