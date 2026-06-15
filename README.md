[![PyPI version](https://badge.fury.io/py/edt.svg)](https://badge.fury.io/py/edt) [![Tests](https://github.com/kevinjohncutler/edt/actions/workflows/tests.yml/badge.svg)](https://github.com/kevinjohncutler/edt/actions/workflows/tests.yml) [![DOI](https://zenodo.org/badge/142239683.svg)](https://zenodo.org/doi/10.5281/zenodo.10815870)

## Multi-Label Anisotropic 3D Euclidean Distance Transform (MLAEDT-3D)

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions. Curious what the output looks like? Download [the movie](https://github.com/seung-lab/euclidean-distance-transform-3d/blob/master/edt.mp4?raw=true)!

```python
import edt
import numpy as np

# e.g. 6nm x 6nm x 30nm for the S1 dataset by Kasthuri et al., 2014
labels = np.ones(shape=(512, 512, 512), dtype=np.uint32, order='F')
dt = edt.edt(
  labels, anisotropy=(6, 6, 30), 
  black_border=True,
  parallel=4 # number of threads, <= 0 sets to num cpu
)
# see more features below
```

### Use Cases  

1. Compute the distance transform of a volume containing multiple labels simultaneously and then query it using a fast masking operator.
2. Convert the multi-label volume into a binary image (i.e. a single label) using a masking operator and compute the ordinary distance transform.

### Python Installation

#### pip Binary Installation

```bash
pip install edt
```

Binaries are available for some platforms. If your platform is not supported, pip will attempt to install from source, in which case follow the instructions below.

#### pip Source Installation

*Requires a C++ compiler.*

```bash
sudo apt-get install g++ python3-dev
pip install numpy
pip install edt --no-binary :all:
```

### Python Usage

Consult `help(edt)` after importing. The module exposes `edt` and `edtsq`, which now delegate to the unified ND backend (`edt_nd` / `edtsq_nd`) and therefore support any dimensionality ≥1 without selecting specialised kernels. Boolean inputs are still handled efficiently.  

Legacy 1D/2D/3D entry points remain available through `edt.legacy` (clone the upstream repository into `original_repo/` and build it to enable them). For convenience the names (`edt`, `edtsq`, `sdf`, etc.) are re-exported as thin aliases that forward to the packaged legacy module once it is built. For new code, prefer the ND APIs.

The two optional parameters are `anisotropy` and `black_border`. Anisotropy is used to correct for distortions in voxel space, e.g. if X and Y were acquired with a microscope, but the Z axis was cut more corsely. 

`black_border` allows you to specify that the edges of the image should be considered in computing pixel distances (it's also slightly faster).  

```python
import edt
import numpy as np

# e.g. 6nm x 6nm x 30nm for the S1 dataset by Kasthuri et al., 2014
labels = np.ones(shape=(512, 512, 512), dtype=np.uint32, order='F')
dt = edt.edt(
  labels, anisotropy=(6, 6, 30), 
  black_border=True,
  parallel=1 # number of threads, <= 0 sets to num cpu
)
# signed distance function (0 is considered background)
sdf = edt.sdf(...) # same arguments as edt

# You can extract individual components of the distance transform
# in sequence rapidly using this technique. A random image may be slow
# as this uses "runs" of voxels to address only parts of the image at a time.
# in_place=True may be faster and more memory efficient, but is read-only.
for label, image in edt.each(labels, dt, in_place=True):
  process(image) # stand in for whatever you'd like to do

# Constrained connectivity via voxel_graph: each voxel encodes which
# of its 2*ndim directions are permissible as a bitmask (uint8).
# Voxels with an impermissible direction are treated as eroded
# by 0.5 in that direction instead of being 1 unit from black.
# WARNING: This is an experimental feature and uses 8x+ memory.
graph = np.zeros(labels.shape, dtype=np.uint8)
graph |= 0b00111111 # all 6 directions permissible (-z +z -y +y -x +x)
graph[122,334,312] &= 0b11111110 # +x isn't allowed at this location
graph[123,334,312] &= 0b11111101 # -x isn't allowed at this location
dt = edt.edt(
  labels, anisotropy=(6, 6, 30), 
  black_border=True,
  parallel=1, # number of threads, <= 0 sets to num cpu
  voxel_graph=graph,
) 
```

*Note on Memory Usage: Make sure the input array to edt is contiguous memory or it will make a copy. You can determine if this is the case with `print(data.flags)`.*

## C++ Instructions for MLAEDT-3D

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions. C++ function expect the input array to be in Fortran (column-major) order. If your array is in C (row-major) order, it will also work but you must
reverse the order of the dimension and anisotropy arguments (sx,sy,sz -> sz,sy,sx and wx,wy,wz -> wz,wy,wx).

### Compiling

You only need `src/edt.hpp` and `src/threadpool.h` for most purposes.

### C++ Examples

```cpp
#include "edt.hpp"

int* labels1d = new int[512]();
int* labels2d = new int[512*512]();
int* labels3d = new int[512*512*512]();

// ... populate labels ...

// 1d, 2d, and 3d anisotropic transforms, wx = anisotropy on x-axis 
float* dt = edt::edt<int>(labels1d, /*sx=*/512, /*wx=*/1.0, /*black_border=*/true); 
float* dt = edt::edt<int>(labels2d, 
  /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0,
  /*black_border=*/true, /*parallel=*/1); 
float* dt = edt::edt<int>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true, /*parallel=*/2); 

// get the squared distance instead (avoids computing sqrt)
float* dt = edt::edtsq<int>(labels1d, /*sx=*/512, /*wx=*/1.0, /*black_border=*/true); 
float* dt = edt::edtsq<int>(labels2d, 
  /*sx=*/512, /*sy=*/512, /*wx=*/1.0, /*wy=*/1.0,
  /*black_border=*/true, /*parallel=*/4); 
float* dt = edt::edtsq<int>(labels3d, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512,
  /*wx=*/4.0, /*wy=*/4.0, /*wz=*/40.0,
  /*black_border=*/true, /*parallel=*/8); 

// signed distance field edt::sdf and edt::sdfsq
float* dt = edt::sdf<int>(labels3d, 
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

### Motivation

<p style="font-style: italics;" align="center">
<img height=256 width=256 src="https://raw.githubusercontent.com/seung-lab/euclidean-distance-transform-3d/master/labeled-cube-kisuk-lee.png" alt="A Labeled 3D Image. Credit: Kisuk Lee" /><br>
<b>Fig. 1.</b> A labeled 3D connectomics volume. Credit: Kisuk Lee
</p>

The connectomics field commonly generates very large densely labeled volumes of neural tissue. Some algorithms, such as the TEASAR skeletonization algorithm [1] and its descendant [2] require the computation of a 3D Euclidean Distance Transform (EDT). We found that the [scipy](https://github.com/scipy/scipy/blob/f3dd9cba8af8d3614c88561712c967a9c67c2b50/scipy/ndimage/src/ni_morphology.c) implementation of the distance transform (based on the Voronoi method of Maurer et al. [3]) was too slow for our needs despite being relatively speedy. 

The scipy EDT took about 20 seconds to compute the transform of a 512x512x512 voxel binary image. Unfortunately, there are typically more than 300 distinct labels within a volume, requiring the serial application of the EDT. While cropping to the ROI does help, many ROIs are diagonally oriented and span the volume, requiring a full EDT. I found that in our numpy/scipy based implementation of TEASAR, EDT was taking approximately a quarter of the time on its own. The amount of time the algorithm spent per a block was estimated to be multiple hours per a core. 

It's possible to compute the EDT much more quickly by computing the distance transform for all labels in one pass by making it boundary aware. Since the distance transform does not result in overlapping boundaries, it is trivial to then extract individual ROIs by to querying the block with the shapes of individual labels. 

<p align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/euclidean-distance-transform-3d/master/two-routes.png" alt="Fig. 2. Extracting the distance transform of a single label from dense segmentation. (a) a 2d slice of dense segmentation (b) extraction of a single label into a binary image (c) simultaneous distance transform of all labels in (a) (d) distance transform of (b) which can be achieved by direct distance transform of (b) or by the multiplication of (b) with (c)." /><br>
<b>Fig. 2.</b> Extracting the distance transform of a single label from dense segmentation. <b>(a)</b> a 2d slice of dense segmentation <b>(b)</b> extraction of a single label into a binary image <b>(c)</b> simultaneous distance transform of all labels in (a) <b>(d)</b> distance transform of (b) which can be achieved by direct distance transform of (b) or by the multiplication of (b) with (c).
</p>

The implementation presented here uses concepts from the 1994 paper by T. Saito and J. Toriwaki [4] and uses a linear sweeping method inspired by the 1966 method of Rosenfeld and Pfaltz [4] and of Mejister et al [7] with that of Mejister et al's and Felzenszwald and Huttenlocher's 2012 [6] two pass linear time parabolic minmal envelope method. I later learned that this method was discovered as early as 1992 by Rein van den Boomgaard in his thesis. [10] I incorporate a few minor modifications to the algorithms to remove the necessity of a black border. My own contribution here is the modification of both the linear sweep and linear parabolic methods to account for multiple label boundaries. 

This implementation was able to compute the distance transform of a binary image in 7-8 seconds. When adding in the multiple boundary modification, this rose to 9 seconds. Incorporating the cost of querying the distance transformed block with masking operators, the time for all operators rose to about 90 seconds, well short of the over an hour required to compute 300 passes. 

### Basic EDT Algorithm Description

A naive implementation of the distance transform is very expensive as it would require a search that is O(N<sup>2</sup>) in the number of voxels. In 1994, Saito and Toriwaki (ST) showed how to decompose this search into passes along x, y, and z linear in the number of voxels. After the X-axis EDT is computed, the Y-axis EDT can be computed on top of it by finding the minimum x<sup>2</sup> + y<sup>2</sup> for each voxel within each column. You can extend this argument to N dimensions. The pertient issue is then finding the minima efficiently without having to scan each column a quadratic number of times.

Felzenszwalb and Huttenlocher (FH) [6] and others have described taking advantage of the geometric interpretation of the distance function, as a parabola. Using ST's decomposition of the EDT into three passes, each broken up by row, the problem becomes one dimensional. FH described computing each pass on each row by finding the minimal envelope of the space created by the parabolas that project from the vertices located at (i, f(i)) for a one dimensional image f.     

This method works by first scanning the row and finding a set of parabolas that constitue the lower envelope. Simultaneously during this linear scan, it computes the abscissa of the nearest parabola to the left and thereby defines the effective domain of each vertex. This linear reading scan is followed by a linear writing scan that records the height of the envelope at each voxel.  

This method is linear and relatively fast, but there's another trick we can do to speed things up. The first transformation is special as we have to change the binary image *f* into a floating point representation. FH recommended using an indicator function that records zeros for out of set and infinities for within set voxels on the first pass. However, this is somewhat cumbersome to reason about and requires an additional remapping of the image.

The original Rosenfeld and Pfaltz (RP) paper [5] demonstrated a remarkably simple two pass sweeping algorithm for computing the manhattan distance  (L1 norm, [visualized here](https://github.com/ljubobratovicrelja/distance-transform)). On the first pass, the L1 and the L2 norm agree as only a single dimension is involved. Using very simple operators and sweeping forward we can compute the increasing distance of a voxel from its leftmost bound. We can then reconcile the errors in a backward pass that computes the minimum of the results of the first pass and the distance from the right boundary. Mejister, Roerdink, and Hesselink (MRH) [7] described a similar technique for use with binary images within the framework set by ST.

In all, we manage to achieve an EDT in six scans of an image in three directions. The use of the RP and MRH inspired method for the first transformation saves about 30% of the time as it appears to use a single digit percentage of the CPU time. In the second and third passes, due to the read and write sequence of FH's method, we can read and write to the same block of memory, increasing cache coherence and reducing memory usage.  

### Multi-Label 1D RP and MRH Inspired Algorithm

The forward sweep looks like:  

    f(a_i) = 0               ; a_i = 0  
           = a_i + 1         ; a_i = 1, i > 0
           = inf             ; a_i = 1, i = 0

I modify this to include consideration of multi-labels as follows:  

    let a_i be the EDT value at i
    let l_i be the label at i (seperate 1:1 corresponding image)
    let w be the anisotropy value
    
    f(a_i, l_i) = 0          ; l_i = 0
    f(a_i, l_i) = a_i + w    ; l_i = l_i-1, l_i != 0

    f(a_i, l_i) = w          ; l_i != l_i-1, l_i != 0 
      f(a_i-1, l_i-1) = w    ; l_i-1 != 0
      f(a_i-1, l_i-1) = 0    ; l_i-1 = 0

The backwards pass is unchanged:  

    from n-2 to 1:
        f(a_i) = min(a_i, a_i+1 + 1)
    from 0 to n-1:
        f(a_i) = f(a_i)^2



### Anisotropy Support in FH Algorithm

A small update to the parabolic intercept equation is necessary to properly support anisotropic dimensions. The original equation appears on page 419 of their paper (reproduced here):

```
s = ((f(r) + r^2) - (f(q) + q^2)) / 2(r - q)

Where given parabola on an XY plane:
s:    x-coord of the intercept between two parabolas
r:    x-coord of the first parabola's vertex
f(r): y-coord of the first parabola's vertex
q:    x-coord of the second parabola's vertex
f(q): y-coord of the second parabola's vertex
```
However, this equation doesn't work with non-unitary anisotropy. The derivation of the proper equation follows.

```
1. Let w = the anisotropic weight
2. (ws - wq)^2 + f(q) = (ws - wr)^2 + f(r)
3. w^2(s - q)^2 + f(q) = w^2(s - r)^2 # w^2 can be pulled out after expansion
4. w^2( (s-q)^2 - (s-r)^2 ) = f(r) - f(q)
5. w^2( s^2 - 2sq + q^2 - s^2 + 2sr - r^2 ) = f(r) - f(q)
6. w^2( 2sr - 2sq + q^2 - r^2 ) = f(r) - f(q)
7. 2sw^2(r-q) + w^2(q^2 - r^2) = f(r) - f(q)
8. 2sw^2(r-q) = f(r) - f(q) - w^2(q^2 - r^2)

=> s = (w^2(r^2 - q^2) + f(r) - f(q)) / 2w^2(r-q)
```

As the computation of *s* is one of the most expensive lines in the algorithm, two multiplications can be saved by factoring *w<sup>2</sup> (r<sup>2</sup> - q<sup>2</sup>)* into *w<sup>2</sup> (r - q) (r + q)*. Here we save one multiplication in computing `r*r + q*q` and replace it with a subtraction. In the denominator of *s*, we avoid having to multiply by *w<sup>2</sup>* or compute *r-q* again. This trick prevents anisotropy support from adding substantial costs.

```
Let w2 = w * w
Let factor1 = (r - q) * w2
Let factor2 = r + q

Let s = (f(r) - f(q) + factor1 * factor2) / (2 * factor1);
```

### Multi-Label Felzenszwalb and Huttenlocher Variation

The parabola method attempts to find the lower envelope of the parabolas described by vertices (i, f(i)).

We handle multiple labels by running the FH method on contiguous blocks of labels independently. For example, in the following column:  

```
Y LABELS:  0 1 1 1 1 1 2 0 3 3 3 3 2 2 1 2 3 0
X AXIS:    0 9 9 1 2 9 7 0 2 3 9 1 1 1 4 4 2 0
FH Domain: 1 1 1 1 1 1 2 2 3 3 3 3 4 4 5 6 7 7
```

Each domain is processed within the envelope described below. This ensures that edges are labeled 1. Alternatively, one can preprocess the image and set differing label pixels to zero and run the FH method without changes, however, this causes edge pixels to be labeled 0 instead of 1. I felt it was nicer to let the background value be zero rather than something like -1 or infinity since 0 is more commonly used in our processes as background.

### Additional Parabolic Envelope  

*Applies to parameter `black_border=True`.*

The methods of RP, ST, and FH all appear to depend on the existence of a black border ringing the ROI. Without it, the Y pass can't propogate a min signal near the border. However, this approach seems wasteful as it requires about 6s<sup>2</sup> additional memory (for a cube) and potentially a copy into bordered memory. Instead, I opted to impose an envelope around all passes. For an image I with n voxels, I implicitly  place a vertex at (-1, 0) and at (n, 0). This envelope propogates the edge effect through the volume.  

To modify the first X-axis pass, I simply mark `I[0] = 1` and `I[n-1] = 1`. The parabolic method is a little trickier to modify because it uses the vertex location to reference `I[i]`. -1 and n are both off the ends of the array and therefore would crash the program. Instead, I add the following lines right after the second pass write:  

```cpp
// let w be anisotropy
// let i be the index
// let v be the abcissa of the parabola's vertex
// let n be the number of voxels in this row
// let d be the transformed (destination) image

d[i] = square(w * (i - v)) + I[v] // line 18, pp. 420 of FH [6]
envelope = min(square(w * (i+1)), square(w * (n - i))) // envelope computation
d[i] = min(envelope, d[i]) // application of envelope
```

These additional lines add about 3% to the running time compared to a program without them. I have not yet made a comparison to a bordered variant.  

### Performance vs. SciPy


<p style="font-style: italics;" align="center">
<img src="https://github.com/seung-lab/euclidean-distance-transform-3d/raw/master/benchmarks/edt-2.0.0_vs_scipy_1.2.1_snemi3d_extracting_labels.png" alt="Fig. 3: Extraction of Individual EDT from 334 Labeled Segments in SNEMI3D MIP 1 (512x512x100 voxels)" /><br>
<b>Fig. 3.</b> Extraction of Individual EDT from 334 Labeled Segments in SNEMI3D MIP 1 (512x512x100 voxels)
</p>

The above experiment was gathered using `memory_profiler` while running the below code once for each function.

```python3
import edt
from tqdm import tqdm
from scipy import ndimage

import numpy as np

labels = load_snemi3d_image()
uniques = np.unique(labels)[1:]

def edt_test():
  res = edt.edt(labels, order='F')
  for segid in tqdm(uniques):
    extracted = res * (labels == segid)

def ndimage_test():
  for segid in tqdm(uniques):
    extracted = (labels == segid)
    extracted = ndimage.distance_transform_edt(extracted)

```

### Environment Variables

Threading behavior can be controlled via environment variables or programmatically using `edt.configure()`.

**Runtime:**

| Variable | Default | Description |
| --- | --- | --- |
| `EDT_ADAPTIVE_THREADS` | `1` | Enable adaptive thread limiting based on array size. Set to `0` to always use the requested thread count. |
| `EDT_ND_MIN_VOXELS_PER_THREAD` | `50000` | Minimum voxels per thread for ND≥4 arrays. |
| `EDT_ND_MIN_LINES_PER_THREAD` | `32` | Minimum lines per thread for ND≥4 arrays. |
| `EDT_ND_PROFILE` | unset | Set to `1` to enable per-call profiling output. |

**Build-time:**

| Variable | Default | Description |
| --- | --- | --- |
| `EDT_MARCH_NATIVE` | `1` | Compile with `-march=native` for the current CPU. Set to `0` to disable. |

**Programmatic override** (takes priority over environment variables):

```python
import edt

# Disable adaptive thread limiting for this process
edt.configure(adaptive_threads=False)

# Lower thread thresholds for small arrays
edt.configure(min_voxels_per_thread=1000, min_lines_per_thread=4)
```

### References

1. M. Sato, I. Bitter, M.A. Bender, A.E. Kaufman, and M. Nakajima. "TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons". Proc. 8th Pacific Conf. on Computer Graphics and Applications. Oct. 2000. doi: 10.1109/PCCGA.2000.883951 ([link](https://ieeexplore.ieee.org/abstract/document/883951/))
2.  I. Bitter, A.E. Kaufman, and M. Sato. "Penalized-distance volumetric skeleton algorithm". IEEE Transactions on Visualization and Computer Graphics Vol. 7, Iss. 3, Jul-Sep 2001. doi: 10.1109/2945.942688 ([link](https://ieeexplore.ieee.org/abstract/document/942688/))
3. C. Maurer, R. Qi, and V. Raghavan. "A Linear Time Algorithm for Computing Exact Euclidean Distance Transforms of Binary Images in Arbitrary Dimensions". IEEE Transactions on Pattern Analysis and Machine Intelligence. Vol. 25, No. 2. February 2003. doi: 10.1109/TPAMI.2003.1177156 ([link](https://ieeexplore.ieee.org/abstract/document/1177156/))
4. T. Saito and J. Toriwaki. "New Algorithms for Euclidean Distance Transformation of an n-Dimensional Digitized Picture with Applications". Pattern Recognition, Vol. 27, Iss. 11, Nov. 1994, Pg. 1551-1565. doi: 10.1016/0031-3203(94)90133-3 ([link](http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Saito94.pdf))
5. A. Rosenfeld and J. Pfaltz. "Sequential Operations in Digital Picture Processing". Journal of the ACM. Vol. 13, Issue 4, Oct. 1966, Pg. 471-494. doi: 10.1145/321356.321357 ([link](https://dl.acm.org/citation.cfm?id=321357))
6. P. Felzenszwald and D. Huttenlocher. "Distance Transforms of Sampled Functions". Theory of Computing, Vol. 8, 2012, Pg. 415-428. doi: 10.4086/toc.2012.v008a019 ([link](http://cs.brown.edu/people/pfelzens/dt/))
7. A. Meijster, J.B.T.M. Roerdink, and W.H. Hesselink. (2002) "A General Algorithm for Computing Distance Transforms in Linear Time". In: Goutsias J., Vincent L., Bloomberg D.S. (eds) Mathematical Morphology and its Applications to Image and Signal Processing. Computational Imaging and Vision, vol 18. Springer, Boston, MA. doi: 10.1007/0-306-47025-X_36 ([link](http://fab.cba.mit.edu/classes/S62.12/docs/Meijster_distance.pdf))
8. H. Zhao. "A Fast Sweeping Method for Eikonal Equations". Mathematics of Computation. Vol. 74, Num. 250, Pg. 603-627. May 2004. doi: 10.1090/S0025-5718-04-01678-3 ([link](https://www.ams.org/journals/mcom/2005-74-250/S0025-5718-04-01678-3/))
9. H. Zhao. "Parallel Implementations of the Fast Sweeping Method". Journal of Computational Mathematics. Vol. 25, No.4, Pg. 421-429. July 2007. Institute of Computational Mathematics and Scientific/Engineering Computing. ([link](https://www.jstor.org/stable/43693378))  
10. "The distance transform, erosion and separability". https://www.crisluengo.net/archives/7 Accessed October 22, 2019. *(This site claims Rein van den Boomgaard discovered a parabolic method at the latest in 1992. Boomgaard even shows up in the comments! If I find his thesis, I'll update this reference.)*
