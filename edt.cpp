#include <cmath>
#include <cstdint>
#include <stdio.h>

#include <string.h>
#include <algorithm>

#define sq(x) ((x) * (x))


void print2d(float* dest, int n) {
  for (int i = 0; i < n*n; i++) {
    if (i % n == 0 && i > 0) {
      printf("\n");
    }
    printf("%.2f, ", dest[i]);
  }

  printf("\n\n");
}



template <typename T>
void edt_1d_single_seg(T* f, float *d,  const int segid, const int n, const int stride, const float anistropy) {
  d[0] = (float)(f[0] == segid); // 0 or 1
  for (int i = 1; i < n; i++) {
    d[i] = (f[i] == segid) * (d[i - 1] + 1.0);
  }

  d[n - 1] = (float)(f[n - 1] == segid);
  for (int i = n - 2; i >= 1; i--) {
    d[i] = anistropy * std::fmin(d[i], d[i + 1] + 1);
  }
}


/* 1D Euclidean Distance Transform for Multiple Segids
 *
 * Map a row of segids to a euclidean distance transform.
 * Zero is considered a universal boundary as are differing
 * segids. Segments touching the boundary are mapped to 1.
 */
template <typename T>
void squared_edt_1d_multi_seg(T* segids, float *d, const int n, const int stride, const float anistropy) {
  int i;

  T working_segid = segids[0];

  d[0] = (float)(working_segid > 0) * anistropy; // 0 or 1
  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid) {
      d[i] = d[i - stride] + anistropy;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = anistropy;
      working_segid = segids[i];
    }
  }

  d[n - stride] = (float)(segids[n - stride] > 0) * anistropy;
  for (i = (n - 2) * stride; i >= stride; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}


/* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 * Essentially, the distance function can be 
 * modeled as the lower envelope of parabolas
 * that spring mainly from edges of the shape
 * you want to transform. The array is scanned
 * to find the parabolas, then a second scan
 * writes the correct values.
 *
 * O(N) time complexity.
 *
 * I (wms) make a few modifications for our use case
 * of executing a euclidean distance transform on
 * a 3D anisotropic image that contains many segments
 * (many binary images). This way we do it correctly
 * without running EDT > 100x in a 512^3 chunk.
 *
 * Parameters:
 *   *f: the image ("sampled function" in the paper)
 *    n: size of image
 *    wx, wy, wz: anistropy e.g. (4nm, 4nm, 40nm)
 * 
 * Returns: distance transform of f
 */
void squared_edt_1d_parabolic(float* f, float *d, const int n, const int stride, const float anistropy) {
  int k = 0;
  int* v = new int[n]();
  float* ranges = new float[n + 1]();

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  float s;
  for (int i = 1; i < n; i++) {
    s = ((f[i * stride] + sq(i)) - (f[v[k] * stride] + sq(v[k]))) / ( 2.0 * (i - v[k]));
    while (s <= ranges[k]) {
      k--;
      s = ((f[i * stride] + sq(i)) - (f[v[k] * stride] + sq(v[k]))) / ( 2.0 * (i - v[k]));
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) {
      k++;
    }

    d[i * stride] = sq(anistropy * (i - v[k])) + f[v[k] * stride];
    envelope = std::fminf(sq(i + 1), sq(n - i));
    d[i * stride] = std::fminf(envelope, d[i * stride]);
  }

  delete [] v;
  delete [] ranges;
}

// /* Df(x,y,z) = min( wx^2 * (x-x')^2 + Df|x'(y,z) )
//  *              x'                   
//  * Df(y,z) = min( wy^2 * (y-y') + Df|x'y'(z) )
//  *            y'
//  * Df(z) = wz^2 * min( (z-z') + i(z) )
//  *          z'
//  * i(z) = 0   if voxel out of set (f[p] == 0)
//  *        inf if voxel in set (f[p] == 1)
//  */
// template <typename T>
// float* dt3d(T* input, 
//   const size_t sx, const size_t sy, const size_t sz, 
//   const float wx, const float wy, const float wz) {

//   const size_t sxy = sx * sy;

//   float *f = new float[sx]();
//   float *xaxis = new float[sx * sy * sz]();
//   for (int z = 0; z < sz; z++) {
//     for (int y = 0; y < sy; y++) { 
//       indicator<T>(input + sx * y + sxy * z, f, sx);
//       dt(f, (xaxis + sx * y + sxy * z), sx, 1, wx);
//     }
//   }

//   delete []f;

//   float *yaxis = new float[sx * sy * sz]();
//   for (int z = 0; z < sz; z++) {
//     for (int x = 0; x < sx; x++) {
//       printf("x: %d z: %d\n", x, z);
//       dt((xaxis + x + sxy * z), (yaxis + x + sxy * z), sy, sx, wy);
//     }
//   }

//   // delete []xaxis;

//   float *zaxis = new float[sx * sy * sz]();
//   for (int y = 0; y < sy; y++) {
//     for (int x = 0; x < sx; x++) {
//       dt((yaxis + x + sx * y), (zaxis + x + sx * y), sz, sxy, wz);
//     }
//   }

//   return yaxis;
//   // return xaxis;
// }


template <typename T>
float* dt2d(T* input, 
  const size_t sx, const size_t sy,
  const float wx, const float wy) {

  float *xaxis = new float[sx * sy]();
  for (int y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>(input, (xaxis + sx * y), sx, 1, wx); 
  }

  float *yaxis = new float[sx * sy]();
  for (int x = 0; x < sx; x++) {
    squared_edt_1d_parabolic((xaxis + x), (yaxis + x), sy, sx, wy);
  }

  delete [] xaxis;

  return yaxis;
}


// void test1d(int n) {
//   float* input = new float[n]();
//   for (int i = 0; i < n; i++) {
//     input[i] = 1.0;
//   }
  
//   float* output = new float[n]();
//   indicator<float>(input, input, n);


//   dt(input, output, n, 1, 1.0);

//   for (int i = 0; i < n; i++) {
//     printf("%.2f, ", output[i]);
//   }
//   printf("\n");

//   delete []input;
//   delete []output;
// }

void test2d(int n) {
  int N = n*n;
  int* input = new int[N]();
  
  for (int i = 0; i < N; i++) {
    input[i] = 1;
  }

  float* dest = dt2d<int>(input, n,n, 1.,1.);

  // print2d(dest, n);

  delete [] dest;
  delete [] input;
}

// void test3d(int n) {
//   int N = n*n*n;
//   int* input = new int[N]();
//   memset(input, 1, N * sizeof(int));

//   float* dest = dt3d<int>(input, n,n,n, 1.,1.,1.);

//   for (int i = 0; i < n*n*n; i++) {
//     if (i % n == 0 && i > 0) {
//       printf("\n");
//     }
//     if (i % (n*n) == 0 && i > 0) {
//       printf("\n");
//     }
//     printf("%.2f, ", dest[i]);
//   }

//   printf("\n\n\n");

//   delete []dest;
// }


void printint(int16_t *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%d, ", f[i]);
  }
}


void printflt(float *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.2f, ", f[i]);
  }
}

void test_multiseg_1d() {
  int size = 1024 * 1024 * 100;
  int16_t* segids = new int16_t[size]();

  for (int i = 0; i < size / 2; i++) {
    segids[i] = 1;
  }

  for (int i = size / 2; i < size; i++) {
    segids[i] = 2;
  }

  segids[10] = 3;

  segids[15] = 0;

  float* d = new float[size]();

  squared_edt_1d_multi_seg<int16_t>(segids, d, size, 1, 1.0);

  printint(segids, 20);
  printf("\n");
  printflt(d, 20);
}

int main() {

  test2d(1024*4);

  return 0;
}

