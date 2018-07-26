#include "edt.hpp"


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

  input[12] = 0;

  float* dest = dt2d<int>(input, n,n, 1.,1.);

  print2d(dest, n);

  delete [] dest;
  delete [] input;
}

void test3d(int n) {
  int N = n*n*n;
  int* input = new int[N]();
  
  for (int i = 0; i < N; i++) {
    input[i] = 1;
  }

  input[13] = 0;

  float* dest = dt3d<int>(input, n,n,n, 1.,1.,1.);

  // for (int i = 0; i < n*n*n; i++) {
  //   if (i % n == 0 && i > 0) {
  //     printf("\n");
  //   }
  //   if (i % (n*n) == 0 && i > 0) {
  //     printf("\n");
  //   }
  //   printf("%.2f, ", dest[i]);
  // }

  // printf("\n\n\n");

  delete []dest;
}

void print(int *in, float* f, float* ans, int n) {
	printf("in: ");
	printint(in, n);
	printf("\nout: ");
	printflt(f, n);
	printf("\nans: ");
	printflt(ans, n);
	printf("\n");
}


void print(float *in, float* f, float* ans, int n) {
	printf("in: ");
	printflt(in, n);
	printf("\nout: ");
	printflt(f, n);
	printf("\nans: ");
	printflt(ans, n);
	printf("\n");
}


void assert(float *xform, float *ans, int n) {
	for (int i = 0; i < n; i++) {
		if (xform[i] != ans[i]) {
			printf("oh no!\n");
			return;
		}
	}
}

void test_one_d_x () {
	int bordered_single[7] = { 0, 1, 1, 1, 1, 1, 0 };
	float ans1[7] = { 0., 1., 4., 9., 4., 1., 0. };
	float *xform = new float[7]();

	squared_edt_1d_multi_seg<int>(bordered_single, xform, 7, 1, 1.0);
	print(bordered_single, xform, ans1, 7);
	assert(xform, ans1, 7);

	float ans1_a[7] = { 0., 4., 16., 36., 16., 4., 0. };
	squared_edt_1d_multi_seg<int>(bordered_single, xform, 7, 1, 2.0);
	print(bordered_single, xform, ans1_a, 7);
	assert(xform, ans1_a, 7);

	int unbordered_single[7] = { 1, 1, 1, 1, 1, 1, 1 };
	float ans2[7] = { 1., 4., 9., 16., 9., 4., 1. };
	squared_edt_1d_multi_seg<int>(unbordered_single, xform, 7, 1, 1.0);
	print(unbordered_single, xform, ans2, 7);
	assert(xform, ans2, 7);

	delete [] xform;
	xform = new float[13]();
	int unbordered_3x[13] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 3, 1, 1 };
	float ans3[13] = { 1., 4., 9., 4., 1., 1., 4., 4., 1., 0., 1., 1., 1. };
	squared_edt_1d_multi_seg<int>(unbordered_3x, xform, 13, 1, 1.0);
	print(unbordered_3x, xform, ans3, 13);
	assert(xform, ans3, 13);
}

void test_one_d_parabola () {
	float bordered_single[7] = { 0, 100, 100, 100, 100, 100, 0 };
	float ans1[7] = { 0., 1., 4., 9., 4., 1., 0. };
	float *xform = new float[7]();

	squared_edt_1d_parabolic(bordered_single, xform, 7, 1, 1.0);
	print(bordered_single, xform, ans1, 7);
	assert(xform, ans1, 7);

	float ans1_a[7] = { 0., 4., 16., 36., 16., 4., 0. };
	squared_edt_1d_parabolic(bordered_single, xform, 7, 1, 2.0);
	print(bordered_single, xform, ans1_a, 7);
	assert(xform, ans1_a, 7);

	float unbordered_interrupted[7] = { 1, 1, 1, 0, 1, 1, 1 };	
	float ans1_b[7] = { 1., 1., 1., 0., 1., 1., 1. };
	squared_edt_1d_parabolic(unbordered_interrupted, xform, 7, 1, 1.0);
	print(unbordered_interrupted, xform, ans1_b, 7);
	assert(xform, ans1_b, 7);

	float unbordered_single[7] = { 100, 100, 100, 100, 100, 100, 100 };
	float ans2[7] = { 1., 4., 9., 16., 9., 4., 1. };
	squared_edt_1d_parabolic(unbordered_single, xform, 7, 1, 1.0);
	print(unbordered_single, xform, ans2, 7);
	assert(xform, ans2, 7);

	delete [] xform;
	xform = new float[13]();
	float unbordered_3x[13] = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 3, 1, 1 };
	float ans3[13] = { 1., 4., 9., 4., 1., 1., 4., 4., 1., 0., 1., 1., 1. };
	squared_edt_1d_parabolic(unbordered_3x, xform, 13, 1, 1.0);
	print(unbordered_3x, xform, ans3, 13);
	assert(xform, ans3, 13);
}


int main () {
	// try {
	// 	test_one_d_parabola();
	// }
	// catch (char const *c) {
	// 	printf("%s", c);
	// }

	test3d(512);
}