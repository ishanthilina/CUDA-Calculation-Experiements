// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>

#define NUM_THREADS 8         //number of threads
#define TOT_COUNT 10000055      //total number of iterations
#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi


/**
A random number generator. 
Guidance from from http://stackoverflow.com/a/3067387/1281089
**/
float randNumGen(){

   int random_value = rand(); //Generate a random number   
   float unit_random = random_value / (float) RAND_MAX; //make it between 0 and 1 
   return unit_random;
}

/**
The task allocated to a thread
**/
void *doCalcs(void *threadid)
{
   long longTid;
   longTid = (long)threadid;
   
   int tid = (int)longTid;       //obtain the integer value of thread id

   //using malloc for the return variable in order make
   //sure that it is not destroyed once the thread call is finished
   float *in_count = (float *)malloc(sizeof(float));
   *in_count=0;
   
   //get the total number of iterations for a thread
   float tot_iterations= TOT_COUNT/NUM_THREADS;
   
   int counter=0;
   
   //calculation
   for(counter=0;counter<tot_iterations;counter++){
      float x = randNumGen();
      float y = randNumGen();
      
      float result = sqrt((x*x) + (y*y));
      
      if(result<1){
         *in_count+=1;         //check if the generated value is inside a unit circle
      }
      
   }
   
   //get the remaining iterations calculated by thread 0
   if(tid==0){
      float remainder = TOT_COUNT%NUM_THREADS;
      
      for(counter=0;counter<remainder;counter++){
      float x = randNumGen();
      float y = randNumGen();
      
      float result = sqrt((x*x) + (y*y));
      
      if(result<1){
         *in_count+=1;         //check if the generated value is inside a unit circle
      }
      
   }
   }


   //printf("In count from #%d : %f\n",tid,*in_count);
   
   pthread_exit((void *)in_count);     //return the in count
}



__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	clock_t start, stop;

	pthread_t threads[NUM_THREADS];
   	int rc;
   	long t;
   	void *status;
   	float tot_in=0;

	// float host[BLOCKS * THREADS];
	// float *dev;
	// curandState *devStates;

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);

	start = clock();

	// cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts
	
	// cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	// gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	// cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results 

	// float pi_gpu;
	// for(int i = 0; i < BLOCKS * THREADS; i++) {
	// 	pi_gpu += host[i];
	// }

	// pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// PThreads
	for(t=0;t<NUM_THREADS;t++){
     rc = pthread_create(&threads[t], NULL, doCalcs, (void *)t);
     if (rc){
       printf("ERROR; return code from pthread_create() is %d\n", rc);
       exit(-1);
       }
     }

   //join the threads
   for(t=0;t<NUM_THREADS;t++){
           
      pthread_join(threads[t], &status);
      //printf("Return from thread %ld is : %f\n",t, *(float*)status);
      
      tot_in+=*(float*)status;            //keep track of the total in count
     
     }
     
   float pthread_pi=4*(tot_in/TOT_COUNT);
   

   //End of PThreads 
	

	start = clock();
	float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

	// printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	printf("PThread estimate of PI = %f [error of %f]\n",pthread_pi,pthread_pi - PI);
	// return 0;
	// 
	/* Last thing that main() should do */
   pthread_exit(NULL);
}

