// clear;rm a.out; nvcc -O3 q3.cu ;./a.out -c
// 
// clear;rm a.out; nvcc -O3 -D DP -L /usr/local/cuda/lib -lcuda q3.cu ;./a.out -c

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>

#define MATRIX_DIM 1600


//Code to check for GPU errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#else
typedef float Real;
#endif


static unsigned long inKB(unsigned long bytes)

{ return bytes/1024; }



static unsigned long inMB(unsigned long bytes)

{ return bytes/(1024*1024); }


/**
 * Used to print memory states in the GPU
 */
 static void printStats()

 {

 	size_t free, total;

 	CUresult res = cuMemGetInfo(&free, &total);

 	if(res != CUDA_SUCCESS){
 		printf("!!!! cuMemGetInfo failed! (status = %x)", res);
 		return;

 	}

 	printf("---------------------------------------------------------------\n");

 	printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));

 	printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));

 	printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
 	printf("---------------------------------------------------------------\n");

 }


 __global__ void cuda_simple_mat_mul(Real* A, Real* B, Real* C) {

 	int col = threadIdx.x + blockIdx.x * blockDim.x;
 	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//check for bounds
 	if(row < MATRIX_DIM && col < MATRIX_DIM)
 	{
 		Real sum = 0.f;

 		for (int i = 0; i < MATRIX_DIM; i++)
 			sum += A[row * MATRIX_DIM + i] * B[i * MATRIX_DIM + col];

 		C[row * MATRIX_DIM + col] = sum;
 	}
 }

/**
 * Initializes the given matrix to a set of float/Double values between 1-2
 */
 void init_matrix(Real matrix[MATRIX_DIM][MATRIX_DIM])
 {
 	for(int i=0; i < MATRIX_DIM; i++)
 	{
 		for(int j=0; j < MATRIX_DIM; j++)
 		{
 			matrix[i][j] = 1 + (Real)rand()/(Real)RAND_MAX;
 		}
 	}
 }

 void print_matrix(Real matrix[MATRIX_DIM][MATRIX_DIM])
 {

 	for(int i = 0; i < MATRIX_DIM; i++)
 	{
 		printf("[");

 			for(int j  = 0; j < MATRIX_DIM; j++)
 			{
			#ifdef DP
 				printf("%20.18f ", matrix[i][j]);
    		#else
 				printf("%f ", matrix[i][j]);
    		#endif

 				
 			}
 			printf("] \n");
 		}
 		printf("\n");
 	}

/**
 * Shows the usage of the program.
 */
 void print_usage(){
 	printf("Wrong usage!\n");
 }

 int main(int argc, char const *argv[])
 {

 	if(argc<2){
 		print_usage();
 	}

 	srand(time(NULL));
 	// Create the matrices
 	static Real A[MATRIX_DIM][MATRIX_DIM]; 
 	static Real B[MATRIX_DIM][MATRIX_DIM]; 
 	static Real C[MATRIX_DIM][MATRIX_DIM]; 
 	// Initialize the matrices
 	init_matrix(A);
 	init_matrix(B);
 	// print_matrix(A);
 	// print_matrix(B);


 	if (0 == strcmp(argv[1], "-s"))
 	{
 		printf("serial mode\n");
 	}
 	else if (0 == strcmp(argv[1], "-p"))
 	{
 		printf("pthread mode\n");
 	}
 	else if (0 == strcmp(argv[1], "-c"))
 	{
 		// If the tiled mode needs to be enabled
 		if (argc>2 && 0 == strcmp(argv[2], "-t")){
 			printf("cuda tiled mode\n");
 		}
 		else{
 			printf("cuda mode\n");

 			long matrix_size=MATRIX_DIM*MATRIX_DIM*sizeof(Real);
 			printf("%ld\n",matrix_size );

 			Real* _A;
 			gpuErrchk(cudaMalloc((void**) &_A, matrix_size));
 			printStats();

 		}
 		
 	}
 	else{
 		print_usage();
 	}
 	return 0;
 }