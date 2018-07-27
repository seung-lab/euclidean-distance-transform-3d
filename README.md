## Multi-Label 3D Euclidean Distance Transform (MLEDT-3D)

Compute the Euclidean Distance Transform of a 1d, 2d, or 3d labeled image containing multiple labels in a single pass with support for anisotropic dimensions.

### Python Installation

*Requires a C++ compiler*

```bash
cd python
pip install numpy
python setup.py develop
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

### C++ Usage

Include the edt.hpp header/implementation. The function names are underscored to avoid a namespace collision in Cython.   

```cpp
#include "edt.hpp"

int main () {

	int* labels = new int[3*3*3]();

	int sx = 3, sy = 3, sz = 3;
	float wx = 6, wy = 6, wz = 30; // anisotropy

	float* dt = edt::_edt3d(labels, sx, sy, sz, wx, wy, wz)

	return 0;
}
```

For compilation, I recommend the compiler flags `-O3` and `-ffast-math`.  


### Motivation

The connectomics field commonly generates very large densely labeled volumes of neural tissue. Some algorithms, such as the TEASAR skeletonization algorithm [1] require the computation of a 3D Euclidean Distance Transform. We found that the commodity implementation of the distance transform as implemented in [scipy](https://github.com/scipy/scipy/blob/f3dd9cba8af8d3614c88561712c967a9c67c2b50/scipy/ndimage/src/ni_morphology.c) (implementing the Voronoi based method of Maurer et al. [2]) was too slow for our needs.  

The relatively speedy scipy implementation took about 20 seconds to compute the transform of a 512x512x512 binary image. Unfortunately, there are typically more than 300 distinct labels within a volume, requiring the serial application of the EDT. While cropping to the ROI does help, many ROIs are diagonally oriented and span the volume, requiring a full EDT. I found that in our numpy/scipy based implementation of TEASAR, EDT was taking approximately a quarter of the time on its own. The amount of time the algorithm was taking per a block was estimated to be multiple hours per a core. 

Also concerning for our TEASAR implementation was that the scipy version did not have a parameter for handling anisotropic data sets (though in the discussion section Maurer et al is clear that their algorithm supports anisotropy with a simple modification). It's common for connectomics datasets to have anisotropic ratios of 10:1 (e.g. 4nm x 4nm x 40nm resolution image stacks), making this treatment important.  

I realized that it's possible to compute the EDT much more quickly by computing the distance transform in one pass by making it label boundary aware at slightly higher computational cost. Since the distance transform does not result in overlapping boundaries, it is trivial to then perform a fast masking operation to query the block for the appropriate distance transform. 

The implementation presented here uses concepts from the 1994 paper by T. Saito and Toriwaki [3] and combines the 1966 linear sweeping method of Rosenfeld and Pfaltz [4] with that of Felzenszwald and Huttenlocher's 2012 two pass linear time parabolic minmal envelope method [5]. I incorporate a few minor modifications to the algorithms to remove the necessity of a black border. My own contribution here is the modification of both the linear sweep and parabolic methods to account for multiple label boundaries.

This implementation was able to compute the distance transform of a binary image in 7-8 seconds. When adding in the multiple boundary modification, this rose to 9 seconds. 

### Basic EDT Algorithm Description

A naive implementation of the distance transform is very expensive as it would require a search that is O(N^2) in the number of voxels. In 1994, Saito and Toriwaki (ST) showed how in principle to decompose this search into three passes, along x, y, and z linear in the number of voxels. After the X-axis EDT is computed, the Y-axis EDT can be computed on top of it by finding the minimum x^2 + y^2 for each pixel within each column. You can extend this argument to N dimensions, but we will halt at Z. The question is then, how to find that minimum efficiently without having to scan each column a quadratic number of times to find that minimum? 

Felzenszwalb and Huttenlocher (FH) and others have described taking advantage of the geometric interpretation of the distance function, as a parabola. Using ST's decomposition of the EDT into three passes, each broken up by row, the problem becomes one dimensional. FH showed that it is possible to compute each pass on each row by finding the minimal envelope of the space created by the parabolas that project from the vertices located (i, f(i)) in a one dimensional image f.   

This method works by first scanning the row and finding a set of parabolas that constitue this lower envelope. During this linear scan, computes the abscissa of the nearest parabola to the left and thereby defines the effective domain of each selected parabola. This linear reading scan is followed by a linear writing scan that records the height of the envelope in each voxel.  

This method is linear and relatively fast, but there's another trick we can do. The first transformation is special as we have to change it from a binary image into something else. If we stayed with FH's method, it was recommended that we use an indicator function that records zeros and infinities on the first pass. However, this is somewhat cumbersome to reason about and is not really adding additional speed as it requires an additional remapping of the image.

However, if we look deeper into the past, the original Rosenfeld and Pfaltz (RP) paper demonstrated a remarkably simple two pass sweeping algorithm for computing the L1 norm ([visualized here](https://github.com/ljubobratovicrelja/distance-transform)). For our very first pass, the L1 and the L2 norm agree as only a single dimension is involved. Using very simple operators, and moving monotonically forward we can compute the increasing distance of a pixel from it's leftmost bound. We can then reconcile that in a backward pass that computes the minimum of the results of the first pass and the distance from the right boundary.  

In all, we manage to achieve an EDT in six scans of an image in three directions. The use of RP's method for the first transformation saves about 30% of the time as it appears to a single digit percentage of the algorithm. In the second and third passes, due to the read and write sequence of FH's method, we can read and write to the same block of memory, increasing cache coherence and reducing memory usage.  

### Modifications to the EDT Algorithm



### Side Notes on Further Performance Improvements

If this scheme seems good to you, but you don't care about multi-segment, it is possible to reduce memory usage down to 1x (down from 2x). Additionally, it is possible to convert to nearly all integer operations when using the squared distance. 

The most expensive operation appears to be the Z scan of our 512x512x512 float cubes. This is almost certainly because of L1 cache misses. The X scan has a stride of 4 bytes and is very fast. The Y scan has a stride of 512\*4 bytes (2 KiB), and is also very fast. The Z scan has a stride of 512\*512\*4 bytes (1 MiB) and seems to add 3-5 seconds to the running time. The L1 cache on the tested computer is 32kB. I considered using half precision floats, but that would only bring it down to 512KiB, which is still a cache miss.

### References

1. M. Sato, et al. "TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons". Proc. 8th Pacific Conf. on Computer Graphics and Applications. Oct. 2000. doi: 10.1109/PCCGA.2000.883951
2. C. Maurer, et al. "A Linear Time Algorithm for Computing Exact Euclidean Distance Transforms of Binary Images in Arbitrary Dimensions". IEEE Transactions on Pattern Analysis and Machine Intelligence. Vol. 25, No. 2. February 2003. doi: 10.1109/TPAMI.2003.1177156
3. T. Saito and J. Toriwaki. "New Algorithms for Euclidean Distance Transformation of an n-Dimensional Digitized Picture with Applications". Pattern Recognition, Vol. 27, Issue 11, Nov. 1994, Pg. 1551-1565. doi: 10.1016/0031-3203(94)90133-3
4. A. Rosenfeld and J. Pfaltz. "Sequential Operations in Digital Picture Processing". Journal of the ACM. Vol. 13, Issue 4, Oct. 1966, Pg. 471-494. doi: 10.1145/321356.321357
5. P. Felzenszwald and D. Huttenlocher. "Distance Transforms of Sampled Functions". Theory of Computing, Vol. 8, 2012, Pg. 415-428. doi: 10.4086/toc.2012.v008a019

