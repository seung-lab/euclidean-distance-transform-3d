import edt
import numpy as np
from scipy import ndimage

import time

for d in (np.uint8, np.uint16, np.uint32, np.uint64):
  labels = np.ones(shape=(4,4), dtype=d)
  start = time.time()
  print(edt.edt2dsq(labels))
  print(d, time.time() - start)


# x = np.ones((512,512,512), dtype=np.int32)

# x[1,1,1] = 0

# ndimage.distance_transform_edt(x)


