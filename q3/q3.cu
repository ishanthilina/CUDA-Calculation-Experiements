#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>

//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#else
typedef float Real;
#endif


#define MATRIX_SIZE 12

__global__ void cuda_simple_mat_mul(float* _pA, float* _pB, float* _pC) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	
	//check for bounds
	if(row < MATRIX_SIZE && col < MATRIX_SIZE)
	{
		float sum = 0.f;
		for (int i = 0; i < MATRIX_SIZE; i++)
			sum += _pA[row * MATRIX_SIZE + i] * _pB[i * MATRIX_SIZE + col];

		_pC[row * MATRIX_SIZE + col] = sum;
	}
}