## Python Instructions for MLAEDT-3D

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions.

### Python Installation

*Requires a C++ compiler*

The installation process depends on `edt.cpp` for the Python bindings derived from `edt.pyx`. `edt.hpp` contains the algorithm implementation.

```bash
cd python
pip install numpy
python setup.py develop
```

### Recompiling `edt.pyx`

*Requires Cython and a C++ compiler*

```bash
cd python
cython -3 --cplus edt.pyx # generates edt.cpp
python setup.py develop # compiles edt.cpp and edt.hpp 
                        # together into a shared binary e.g. edt.cpython-36m-x86_64-linux-gnu.so
```

### Python Usage

Consult `help(edt)` after importing. The edt module contains: `edt`, `edtsq` which compute the euclidean and squared euclidean distance respectively and select dimension based on the shape of the numpy object fed to them. 1D, 2D, and 3D volumes are supported. 1D processing is extremely fast.  

If for some reason you'd like to use the specific 'D' function you want, `edt1d`, `edt1dsq`, `edt2d`, `edt2dsq`, `edt3d`, and `edt3dsq` are included.

```python
import edt
import numpy as np

labels = np.ones(shape=(512, 512, 512), dtype=np.uint32)
dt = edt.edt(labels, anisotropy=(6, 6, 30)) # e.g. for the S1 dataset by Kasthuri et al., 2014
```

