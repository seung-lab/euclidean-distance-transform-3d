import numpy as np
import edt

x = np.ones((256,256,256), dtype=np.uint8)

for i in range(10):
  ex = edt.edt(x)

print("done.")