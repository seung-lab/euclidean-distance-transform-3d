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

// 1d, 2d, and 3d anisotropic transforms, wx = anisotropy on x-axis 
float* dt = edt<int>(labels1d, /*sx=*/512, /*wx=*/1.0, /*black_border=*/true); 
float* dt = edt<int>(labels2d, 
  /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0,
  /*black_border=*/true, /*parallel=*/1); 
float* dt = edt<int>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true, /*parallel=*/2); 

// get the squared distance instead (avoids computing sqrt)
float* dt = edtsq<int>(labels1d, /*sx=*/512, /*wx=*/1.0, /*black_border=*/true); 
float* dt = edtsq<int>(labels2d, 
  /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0,
  /*black_border=*/true, /*parallel=*/4); 
float* dt = edtsq<int>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true, /*parallel=*/8); 
```

### High Performance Binary Images

Binary images are treated specially in 2D and 3D to avoid executing the extra multi-label logic (1D is very fast even with it). This results in a substantial savings of perhaps 20-50% depending on the compiler. For a 512x512x512 cube filled with ones, on a 4.0 GHz linux machine with g++, I witnessed reductions from 9 sec. to 7 sec. (1.29x). On 2.8 GHz Mac OS with clang-902.0.39.2 I saw a reduction from 12.4 sec to 7.9 sec (1.56x).  

The code will easily handle all integer types, and the image only needs to be binary in the sense that there is a single non-zero label, it doesn't have to be ones.  

Boolean typed images are handled specially by a specialization of the edt function, so nothing different from above needs to be done. If you have an integer typed image, you'll need to use `binary_edt` or `binary_edtsq` instead to take advantage of this.  

You'll get slightly higher performance setting `black_border=true`.  


```cpp
#include "edt.hpp"

using namespace edt;

bool* labels2d = new bool[512*512]();
bool* labels3d = new bool[512*512*512]();

float* dt = edt<bool>(labels2d, 
  /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0,
  /*black_border=*/true); 
float* dt = edt<bool>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true); 


int* labels2d = new int[512*512]();
int* labels3d = new int[512*512*512]();

float* dt = binary_edt<int>(labels2d, /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0); 
float* dt = binary_edt<int>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true); 

```
