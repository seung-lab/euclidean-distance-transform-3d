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

// void test2d(int n) {
//   int N = n*n;
//   int* input = new int[N]();
  
//   for (int i = 0; i < N; i++) {
//     input[i] = 1;
//   }

//   float* dest = dt2d<int>(input, n,n, 1.,1.);

//   print2d(dest, n);

//   delete [] dest;
//   delete [] input;
// }

void print(int *in, float* f, float* ans, int n) {
	printf("in: ");
	printint(in, n);
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

int main () {
	try {
		test_one_d_x();
	}
	catch (char const *c) {
		printf("%s", c);
	}
}