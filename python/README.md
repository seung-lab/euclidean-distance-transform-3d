## Python Instructions for MLAEDT-3D

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions.

### Python Installation

*Requires a C++ compiler*

The installation process depends on `edt.cpp` for the Python bindings derived from `edt.pyx`. `edt.hpp` contains the algorithm implementation.  

```bash
pip install numpy
pip install edt
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

Consult `help(edt)` after importing. The edt module contains: `edt` and `edtsq` which compute the euclidean and squared euclidean distance respectively. Both functions select dimension based on the shape of the numpy array fed to them. 1D, 2D, and 3D volumes are supported. 1D processing is extremely fast. Numpy boolean arrays are handled specially for faster processing.  

If for some reason you'd like to use a specific 'D' function, `edt1d`, `edt1dsq`, `edt2d`, `edt2dsq`, `edt3d`, and `edt3dsq` are available.  

The three optional parameters are `anisotropy`, `black_border`, and `order`. Anisotropy is used to correct for distortions in voxel space, e.g. if X and Y were acquired with a microscope, but the Z axis was cut more corsely.  

`black_border` allows you to specify that the edges of the image should be considered in computing pixel distances (it's also slightly faster).  

`order` allows the programmer to determine how the underlying array should be interpreted. `'C'` (C-order, XYZ, row-major) and `'F'` (Fortran-order, ZYX, column major) are supported. `'C'` order is the default.

`parallel` controls the number of threads. Set it <= 0 to automatically determine your CPU count.

```python
import edt
import numpy as np

# e.g. 6nm x 6nm x 30nm for the S1 dataset by Kasthuri et al., 2014
labels = np.ones(shape=(512, 512, 512), dtype=np.uint32, order='F')
dt = edt.edt(labels, anisotropy=(6, 6, 30), black_border=True, order='F', parallel=1) 
```

