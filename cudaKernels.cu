
#include "./utils.h"
#include "./cudaKernels.cuh"

// #include "timing.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
/******
* CUDA 3D volume Rotation
*/


__device__ float my_roundf (float a)
{
    float fa = fabsf (a);
    float t = ((fa >= 0.5f) && (fa <= 8388608.0f)) ? 0.5f : 0.0f;
    return copysignf (truncf (fa + t), a);
}
__global__ void
d_render(float *d_output,
         float *d_input,
         float theta,
         float phi,
         unsigned int nx,
         unsigned int ny,
         unsigned int nz,
		 unsigned int log2_sizeimg
         // float ST,
         // float CT,
         // float SP,
         // float CP
          )
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int nxny = nx * ny;
    int z = tid / nxny + 1;
    int x = tid % nx + 1;
    int y = ( tid % nxny ) / nx + 1;

    float ST = sinf(theta);
    float CT = cosf(theta);
    float SP = sinf(phi);
    float CP = cosf(phi);

    int p1 = (nx + 1)/2 + 1;
    int p2 = (ny + 1)/2 + 1;
    int p3 = (nz + 1)/2 + 1;

    // Apply the rotation, nearest neighbor
    // int xx = my_roundf( x*CT + z*ST - CT*p1 + p1 - ST*p3 );
    // if (xx > nx || xx < 1) return;
    // int yy = my_roundf(- x*SP*ST + y*CP + z*SP*CT + SP*ST*p1 - CP*p2 - SP*CT*p3 + p2);
    // if (yy > ny || yy < 1) return;
    // int zz = my_roundf(- x*CP*ST - y*SP + z*CP*CT + CP*ST*p1 + SP*p2 - CP*CT*p3 + p3);
    // if (zz > nz || zz < 1) return;
    // Apply the rotation, nearest neighbor
    int xx = my_roundf( x*CT - y*ST*SP - z*ST*CP - p1*CT + p2*ST*SP + p3*ST*CP + p1);
    if (xx > nx || xx < 1) return;
    int yy = my_roundf( y*CP - z*SP - p2*CP + p3*SP + p2 );
    if (yy > ny || yy < 1) return;
    int zz = my_roundf( x*ST + y*CT*SP + z*CT*CP - p1*ST - p2*CT*SP - CT*CP*p3 + p3 );
    if (zz > nz || zz < 1) return;
    /*
    * Apply inverse rotation and find the nearest neighbor in the 
    * ORIGINAL image instead of in the output image.
    */

    
    // if (xx > nx || xx < 1 || yy > ny || yy < 1 || zz > nz || zz < 1) {
    //  return;
    // }
    unsigned int idx = ((zz - 1) * (nx * ny)) + ((yy - 1) * nx) + xx - 1;
  // float voxel = tex3D( tex, x - 1, y - 1, z - 1 );
    float voxel = d_input[ idx ];
    d_output[tid] = voxel;
}

__global__ void
d_render_cumulate(float *d_output,
         const float *d_input,
         float theta,
         float phi,
         unsigned int nx,
         unsigned int ny,
         unsigned int nz,
         const unsigned int numAngles,
		 unsigned int log2_sizeimg
          )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nxny = nx * ny;
    const int z = tid / nxny + 1;
    const int x = tid % nx + 1;
    const int y = ( tid % nxny ) / nx + 1;

    float ST = sinf(theta);
    float CT = cosf(theta);
    float SP = sinf(phi);
    float CP = cosf(phi);


    int p1 = (nx + 1)/2 + 1;
    int p2 = (ny + 1)/2 + 1;
    int p3 = (nz + 1)/2 + 1;

    /*
    * Apply inverse rotation and find the nearest neighbor in the 
    * ORIGINAL image instead of in the output image.
    */
    int xx = roundf( x*CT - z*ST - p1*CT + p3*ST + p1);
    if (xx > nx || xx < 1) return;
    int yy = roundf( - x*SP*ST + y*CP - z*SP*CT + p1*SP*ST - p2*CP + p3*SP*CT + p2 );
    if (yy > ny || yy < 1) return;
    int zz = roundf( x*CP*ST + y*SP + z*CP*CT - p1*CP*ST - p2*SP - p3*CP*CT + p3 );
    if (zz > nz || zz < 1) return;

    
    // if (xx > nx || xx < 1 || yy > ny || yy < 1 || zz > nz || zz < 1) {
    //  return;
    // }
  unsigned int idx = ((zz - 1) * (nx * ny)) + ((yy - 1) * nx) + xx - 1;
  // float voxel = tex3D( tex, x - 1, y - 1, z - 1 );
  float voxel = d_input[ idx ];
  // d_output[tid] += voxel;
    // Apply a texture lookup
    // d_output[tid] = voxel;
  atomicAdd( &d_output[tid], voxel );
  // check exception
  if (isnan(d_output[tid]) || isinf(d_output[tid])) {
	  d_output[tid] = 0;
  }
}

void cuRotateInit(
	float** dev_input,
	float** dev_output,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz
	) {
  
  gpuErrchk(cudaMalloc( dev_input, sizeof(float) * nx * ny * nz ));
    gpuErrchk(cudaMalloc( dev_output, sizeof(float) * nx * ny * nz ));

}

void cuRotate(
	float** input,
	float** output,
	float** dev_input,
	float** dev_output,
	unsigned int nx,
	unsigned int ny,
	unsigned int nz,
	float theta,
	float phi
	) {

  // Copy input data
  gpuErrchk( cudaMemcpy( 
      *dev_input,
      *input,
      sizeof(float) * nx * ny * nz,
      cudaMemcpyHostToDevice
      ) );
  gpuErrchk( cudaMemcpy( 
      *dev_output,
      *output,
      sizeof(float) * nx * ny * nz,
      cudaMemcpyHostToDevice
      ) );
	d_render<<< nx * ny * nz / 256, 256 >>> (
		*dev_output,
		*dev_input,
		theta,
		phi,
		nx, ny, nz, 0/* log2_sizeimg */
		);
	gpuErrchk( cudaMemcpy(
		*output,
		*dev_output,
		sizeof(float) * nx * ny * nz,
		cudaMemcpyDeviceToHost
		));
}

void cuRotateFree(
  float** dev_input,
  float** dev_output
  ) {
  gpuErrchk( cudaFree(*dev_output) );
  gpuErrchk( cudaFree(*dev_input) );
}