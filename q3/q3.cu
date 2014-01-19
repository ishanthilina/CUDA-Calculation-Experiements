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

#define MATRIX_DIM 1800

#define MIN_ERROR 0.1

// CUDA related
#define BLOCK_SIZE 32


//Code to check for GPU errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),\
		 file, line);
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

 	printf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), \
 		inMB(free));

 	printf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), \
 		inMB(total));

 	printf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, \
 		100.0*(total - free)/(double)total);
 	printf("---------------------------------------------------------------\n");

 }

/**
 * Carries out a simple square matrix multiplication where each thread
 * calculates a single item in the resulting matrix.
 * @param A First matrix
 * @param B Second matrix
 * @param C Results matrix
 */
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

/**
 * Prints the given matrix to the stdout
 */
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
 * Compares the given two matrices.
 */
 void compare_matrices(Real matrix1[MATRIX_DIM][MATRIX_DIM],\
 	Real matrix2[MATRIX_DIM][MATRIX_DIM])
 {
 	for(int i = 0; i < MATRIX_DIM; i++)
 	{
 		for(int j = 0; j < MATRIX_DIM; j++)
 		{
 			if((matrix1[i][j] - matrix2[i][j] > MIN_ERROR) &&
 				(matrix1[i][j] - matrix2[i][j] > 0))
 			{
 				printf("Error i=%d : j=%d mat1=%f mat2=%f\n",i,j,\
 					matrix1[i][j], matrix2[i][j]);
 				return;
 			}
 		}
 	}

 	printf("Matrices Match! \n");
 } 
/**
 * carries out a serial matrix multiplication
 */
 void serial_mat_mul(Real A[MATRIX_DIM][MATRIX_DIM], \
 	Real B[MATRIX_DIM][MATRIX_DIM], Real C[MATRIX_DIM][MATRIX_DIM])	{
 	float sum;
 	for (int row=0; row<MATRIX_DIM; row++){
 		for (int col=0; col<MATRIX_DIM; col++){
 			sum = 0.f;
 			for (int n=0; n<MATRIX_DIM; n++){
 				sum += A[row][n]*B[n][col];
 			}
 			C[row][col] = sum;
 		}
 	}
 }

/**
 * Shows the usage of the program.
 */
 void print_usage(){
 	printf("Wrong usage!\n");
 }

/**
 * Does a matrix multiplication using the "tiled" approach in the GPU
 * @param A First matrix
 * @param B Second matrix
 * @param C Results matrix
 */
 __global__ void cuda_tiled_mat_mul(Real * A, Real * B, Real * C) {
 	
 	float CValue = 0;

 	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
 	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

 	__shared__ Real As[BLOCK_SIZE][BLOCK_SIZE];
 	__shared__ Real Bs[BLOCK_SIZE][BLOCK_SIZE];

 	for (int k = 0; k < (BLOCK_SIZE + MATRIX_DIM - 1)/BLOCK_SIZE; k++) {
 		// check ranges for the matrices and check for left out parts where
 		//  MATRIX_DIM is not an exact multiplication of tile size(BLOCK_SIZE)
 		if (k*BLOCK_SIZE + threadIdx.x < MATRIX_DIM && Row < MATRIX_DIM){

 			As[threadIdx.y][threadIdx.x] = A[Row*MATRIX_DIM + \
 				k*BLOCK_SIZE + threadIdx.x];
 		}  
 		else{

 			As[threadIdx.y][threadIdx.x] = 0.0;
 		}                                                   

 		if (k*BLOCK_SIZE + threadIdx.y < MATRIX_DIM && Col < MATRIX_DIM){
 			
 			Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + \
 				threadIdx.y)*MATRIX_DIM + Col];
 		}
 		else{

 			Bs[threadIdx.y][threadIdx.x] = 0.0;
 		}                                                   

 		// Wait till all the threads finish before calculating the results
 		__syncthreads();

 		for (int n = 0; n < BLOCK_SIZE; ++n) 
 			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

 		__syncthreads();
 	}

 	// Calculate the result
 	if (Row < MATRIX_DIM && Col < MATRIX_DIM) 
 		C[((blockIdx.y * blockDim.y + threadIdx.y)*MATRIX_DIM)+\
 			(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;

 }


 int main(int argc, char const *argv[])
 {

 	if(argc<2){
 		print_usage();
 	}

 	// Initialize the random seed
 	srand(time(NULL));

 	// Create the matrices
 	static Real A[MATRIX_DIM][MATRIX_DIM]; 
 	static Real B[MATRIX_DIM][MATRIX_DIM]; 
 	static Real C[MATRIX_DIM][MATRIX_DIM]; 
 	static Real serial_C[MATRIX_DIM][MATRIX_DIM]; 
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

 		long matrix_size=MATRIX_DIM*MATRIX_DIM*sizeof(Real);
 			// printf("%ld\n",matrix_size );

 		Real* _A;
 		gpuErrchk(cudaMalloc((void**) &_A, matrix_size));
 		// printStats();

 		Real* _B;
 		gpuErrchk(cudaMalloc((void**) &_B, matrix_size));
 		// printStats();

 		Real* _C;
 		gpuErrchk(cudaMalloc((void**) &_C, matrix_size));
 		// printStats();

 		// copy the matrices to device
 		cudaMemcpy(_A, A, matrix_size, cudaMemcpyHostToDevice);
 		cudaMemcpy(_B, B, matrix_size, cudaMemcpyHostToDevice);

 		// If the tiled mode needs to be enabled
 		if (argc>2 && 0 == strcmp(argv[2], "-t")){
 			printf("cuda tiled mode\n");

 			// set the grid and block sizes
 			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
 			dim3 dimGrid;
 			dimGrid.x = (MATRIX_DIM + dimBlock.x - 1)/dimBlock.x;
 			dimGrid.y = (MATRIX_DIM + dimBlock.y - 1)/dimBlock.y;

 			// execute the workload in the GPU
 			cuda_tiled_mat_mul<<<dimGrid , dimBlock>>>(_A,_B,_C);

 			// Copy back the result
 			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);

 			// get the serial output
 			serial_mat_mul(A,B,serial_C);

 			// print_matrix(serial_C);
 			// print_matrix(C);

 			// Compare the reuslts
 			compare_matrices(serial_C,C);

 			// free device memory
 			cudaFree(_A);
 			cudaFree(_B);
 			cudaFree(_C);

 		}
 		else{
 			printf("cuda mode\n");

 			int K=100;			
 			
 			dim3 threadBlock(BLOCK_SIZE,BLOCK_SIZE);
 			dim3 grid(K,K);

 			// call the GPU
 			cuda_simple_mat_mul<<<grid,threadBlock>>>(_A,_B,_C);

 			// Copy back the result
 			cudaMemcpy(C,_C,matrix_size,cudaMemcpyDeviceToHost);

 			// get the serial output
 			serial_mat_mul(A,B,serial_C);

 			// print_matrix(serial_C);
 			// print_matrix(C);

 			compare_matrices(serial_C,C);

 			// free device memory
 			cudaFree(_A);
 			cudaFree(_B);
 			cudaFree(_C);


 		}
 		
 	}
 	else{
 		print_usage();
 	}
 	return 0;
 }
