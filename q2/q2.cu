// clear;rm a.out; nvcc -O3 -arch sm_20 q2.cu ; ./a.out -c

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>


// CUDA related
#define THREADS_PER_BLOCK 512
#define CALCS_PER_THREAD 50

// PThread related
#define MAX_PTHREADS 8

#define VECTOR_SIZE 100000000  //1e8
// #define VECTOR_SIZE 500  //1e8


//Help code for switching between Single Precision and Double Precision
#ifdef DP
typedef double Real;
#else
typedef float Real;
#endif

/**
 * Shows the usage of the program.
 */
 void print_usage(){
 	printf("Wrong usage!\n");
 }

 void print_vector(Real vector[]){
 	printf("---------------------------------------------------------------\n");
 	for(long i=0;i<VECTOR_SIZE;i++){
		#ifdef DP
 		printf("%ld --> %20.18f\n",i,vector[i]);

        #else
 		printf("%ld --> %f\n",i,vector[i]);


        #endif
 	}
 	printf("---------------------------------------------------------------\n");
 }

/**
 * Initializes a given vector with values between 1 and 2
 * @param vector The vector that needs to be initialized
 */
 void initialize_vector(Real vector[]){
 	for(long i=0;i<VECTOR_SIZE;i++){
 		vector[i]=(rand() / (float) RAND_MAX)+1;
 	}
 }

/**
 * Does a serial calculation of the dot product of the two given vectors
 * @param  vector1 
 * @param  vector2 
 * @return Dot product value of the vectors
 */
 Real serial_calculation(Real vector1[], Real vector2[]){

 	Real result;

 	for(long i=0;i<VECTOR_SIZE;i++){
 		result += vector1[i] * vector2[i];
 	}

 	return result;
 }

/**
 * prints the value of pi
 * @param pi   calculated value for pi
 * @param from the name of the method that the pi value was calculated. ex- CUDA
 */
 void print_product(Real pi, char *from){
	#ifdef DP
 	char *latter=" result is %20.18f\n";
    #else
 	char *latter=" result is %f\n";
    #endif

 	char *to_print = (char *)malloc(\
			strlen(from)+strlen(latter)+1);//+1 for 0-terminator
 	strcpy(to_print, from);
 	strcat(to_print, latter);
 	printf(to_print,pi);

    // free memory
 	free(to_print);


 }

//struct for parameter passing between pthread calls
 struct pthread_arg_struct {
 	int tid;
 	int total_threads;
 	Real *vector1;
 	Real *vector2;
 };


 void *pthread_calculation(void *arguments){

 	struct pthread_arg_struct *args = (struct pthread_arg_struct *)arguments;
 	int total_threads = args -> total_threads;
	int tid = args -> tid;       //obtain the value of thread id
	Real *vector1=args -> vector1;
	Real *vector2=args -> vector2;

	Real *result = (Real *)malloc(sizeof(Real));
	*result=0;

	// printf("%d\n",tid );
	// print_vector(vector1);
	// print_vector(vector2);

	// calculate the range to be multiplied
	int chunk_size = VECTOR_SIZE/total_threads;
	int lowerbound=chunk_size*tid;			// lowest index to be calculated
	int upperbound=lowerbound+chunk_size-1;	// highest index to be calculated
	// printf("%d calculates from %d to %d\n",tid, lowerbound,upperbound );

	for(int i=lowerbound;i<=upperbound;i++){
		*result+=vector1[i]*vector2[i];
	}

	// allocate the leftover vector elements to master
	if(0==tid && (0!=VECTOR_SIZE%total_threads)){
		for(int i=chunk_size*total_threads;i<=VECTOR_SIZE;i++){
			*result+=vector1[i]*vector2[i];
		}

	}

	// printf("In the end thread %d total is %f\n",tid,*result );
   	pthread_exit((void *)result);     //return the in count

   }

/**
 * Vector dot product code for a single CUDA thread. Function assumes that 
 * the VECTOR_SIZE is completely divisible by CALCS_PER_THREAD
 * 
 * @param vector1 First vector - An array of Real
 * @param vector2 Second vector - An array of Real
 * @param result  Used to return the result - An array of Real
 */
 __global__ void cuda_thread_task(Real vector1[], Real vector2[], Real result[]) {
 	 	

 	unsigned long start_point = threadIdx.x + blockDim.x * blockIdx.x;


	// calculate the range to be multiplied
 	long lowerbound=start_point*CALCS_PER_THREAD;
 	
 	long upperbound=lowerbound+CALCS_PER_THREAD-1;
	// printf("%ld - %ld - %ld \n", start_point,lowerbound,upperbound);
 	for(long index=lowerbound;index<=upperbound;index++){
 		// printf("%ld - %ld - %ld - %ld\n", start_point,lowerbound,upperbound,index);
		result[index] = vector1[index]*vector2[index];
 		result[1] = 13;
 		// vector1[index]=12;

 	}
 		printf("2-Hello thread %d\n", threadIdx.x);
 	// result[start_point]=23;

 	


 }

 int main(int argc, char const *argv[])
 {
	// check the inputs and set the mode
	// int execution_mode=-1;
 	if(argc<2){
 		print_usage();
 	}
	// initialize the vectors
	// printf("%d\n",VECTOR_SIZE);
 	static Real vector1[VECTOR_SIZE];
 	static Real vector2[VECTOR_SIZE];
 	initialize_vector(vector1);
 	initialize_vector(vector2);

	// print_vector(vector1);
	// print_vector(vector2);

	// if a serial execution is needed
 	if(0==strcmp(argv[1],"-s")){
 		printf("serial mode\n");
		// printf("%f\n",serial_calculation(vector1,vector2) );
 		print_product(serial_calculation(vector1,vector2),"SERIAL");
 	}
	// if a parallel execution is needed
 	else if(0==strcmp(argv[1],"-p")){
 		print_product(serial_calculation(vector1,vector2),"SERIAL");

 		printf("parallel mode\n");

 		int num_of_threads;
		// check whether the given # of threads is valid
 		if(argc !=3){
 			print_usage();
 			return -1;
 		}
 		num_of_threads=atoi(argv[2]);
 		if(num_of_threads>MAX_PTHREADS){
 			printf("[ERROR-PTHREADS] - Only up to 8 threads can be created\n");
 			return -1;
 		}

		// printf("Creating %d threads\n", num_of_threads);
 		pthread_t threads[num_of_threads];
 		int rc;
 		long t;
 		void *status;
 		Real result=0;

   		//initialize the threads
 		for(t=0;t<num_of_threads;t++){
 			struct pthread_arg_struct* args=(\
 				struct pthread_arg_struct*)malloc(sizeof *args);

 			args->total_threads=num_of_threads;
 			args->tid=t;
 			args-> vector1=vector1;
 			args-> vector2=vector2;

 			rc = pthread_create(&threads[t], NULL, pthread_calculation,(void *)args);
 			if (rc){
 				printf("ERROR; return code from pthread_create() is %d\n", rc);
 				exit(-1);
 			}
 		}

   		//join the threads
 		for(t=0;t<num_of_threads;t++){
 			pthread_join(threads[t], &status);
            result+=*(Real*)status;            //keep track of the total in count
            // printf("Thread: %ld %f\n",t,result );

        }

        print_product(result,"PTHREADS");

    }
	// if CUDA execution is needed
    else if(0==strcmp(argv[1],"-c")){
    	print_product(serial_calculation(vector1,vector2),"SERIAL");

    	printf("cuda mode\n");

		//Allocate vectors in device memory
    	size_t size = VECTOR_SIZE * sizeof(Real);
    	Real* _vector1;
    	cudaMalloc((void**) &_vector1, size);
    	Real* _vector2;
    	cudaMalloc((void**) &_vector2, size);
    	Real* _results;
    	cudaMalloc((void**) &_results, size);

		// Allocate memory for results in the host memory
    	Real* results = (Real*)malloc(size);

		//copy vectors from host memory to device memory
    	cudaMemcpy(_vector1, vector1,size,cudaMemcpyHostToDevice);
    	cudaMemcpy(_vector2, vector2,size,cudaMemcpyHostToDevice);
    	// cudaMemcpy(_results, results,size,cudaMemcpyHostToDevice);


    	long num_of_grids=(VECTOR_SIZE/(THREADS_PER_BLOCK*CALCS_PER_THREAD))+1;
    	printf("#of Grids = %ld\n",num_of_grids );
		// carry out the calculations
    	cuda_thread_task\
    	<<<num_of_grids,THREADS_PER_BLOCK>>>(_vector1,_vector2,_results);

		// copy the results back from the device memory to host memory
    	cudaMemcpy(results,_results,size,cudaMemcpyDeviceToHost);

		// free device memory
    	cudaFree(_vector1);
    	cudaFree(_vector2);
    	cudaFree(_results);

		// calculate the final result
    	Real result=0;
    	for(int i=0;i<VECTOR_SIZE;i++){
    		result+=results[i];
    		// if(results[i]!=0.0){
    		// 	printf("%f\n",results[i] );

    		// }
    	}

    	print_product(result,"CUDA");


    }
    else{
    	print_usage();
    }
    return 0;
}