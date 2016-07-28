#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <complex>
#include <vector>

using namespace std;

extern "C" void cuRotateInit(
	float** dev_input,
	float** dev_output,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz
	);

extern "C" void cuRotate(
	float** input,
	float** output,
	float** dev_input,
	float** dev_output,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	float theta,
	float phi
	);

extern "C" void cuRotateFree(
  float** dev_input,
  float** dev_output
  );

#endif