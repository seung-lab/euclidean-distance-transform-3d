import numpy as np
import edt

x = np.ones((512,512,512), dtype=np.uint8)

for i in range(10):
  ex = edt.edt(x, parallel=4)

print("done.")