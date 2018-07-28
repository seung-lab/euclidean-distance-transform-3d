## C++ Instructions for MLAEDT-3D

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions.

### Compiling

You only need `edt.hpp`, `test.cpp` is only there for testing.

```bash
make shared # compile edt.so
make test # compile ./test with debugging information
```

If you statically integrate `edt.hpp` into your own C++ program, I recommend compiler flags `-O3` and `-ffast-math` for optimal performance.

### Using C++ Functions

```cpp
#include "edt.hpp"

using namespace edt;

int* labels1d = new int[512]();
int* labels2d = new int[512*512]();
int* labels3d = new int[512*512*512]();

// ... populate labels ...

// 1d, 2d, and 3d anisotropic transforms 
float* dt = edt(labels1d, /*sx=*/512, /*wx=*/1.0); // wx = anisotropy in x direction
float* dt = edt(labels2d, /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0); 
float* dt = edt(labels3d, 
	/*sx=*/512, /*sy=*/512, /*sz=*/512,
	/*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0); 

// get the squared distance instead (avoids computing sqrt)
float* dt = edtsq(labels1d, /*sx=*/512, /*wx=*/1.0); // wx = anisotropy in x direction
float* dt = edtsq(labels2d, /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0); 
float* dt = edtsq(labels3d, 
	/*sx=*/512, /*sy=*/512, /*sz=*/512,
	/*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0); 
```