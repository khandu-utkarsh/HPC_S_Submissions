// nvcc -std=c++11  vectorVectorProduct.cu -o vectorVectorProduct -Xcompiler -fopenmp
// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <iostream>

const int THREADS_PER_BLOCK = 1024;
int BLOCKS_COUNT = 4;
int N_items = BLOCKS_COUNT * THREADS_PER_BLOCK;


void vec_mult(double* c, const double* a, const double* b, long N){
  double localSum = 0.;
  //#pragma omp parallel for schedule(static) reduction(+:localSum)
  for (long i = 0; i < N; i++) {
    localSum += a[i] * b[i];
  }
  //printf("Local Sum: %f\n",localSum);
  c[0] = localSum;
  //printf("Local Sum c: %f\n",c[0]);
}

__global__
void vec_vec_mult_kernel(double* c, const double* a, const double* b, long N){
  //!c should be of size 1
  //!a and b should be of same size

  //!Creating a shared variable for a block
  __shared__ double partialInner[THREADS_PER_BLOCK];

  //!Global threads index
  int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(global_thread_index < N)
  {
    partialInner[threadIdx.x] = a[global_thread_index] * b[global_thread_index];
    //printf("Cuda thread id: %d | Local Prod: %f |Curr Local c: %f\n",global_thread_index, a[global_thread_index] * b[global_thread_index],partialInner[threadIdx.x]);
  }
  else
  {
    partialInner[threadIdx.x] = 0.;
  }

  __syncthreads();  //!Waiting for all the threads of a block to finish up.


  // x >>= 1 means " set x to itself shifted by one bit to the right "
  // means divide x by 2 in each iteration
  for (int limit = THREADS_PER_BLOCK/2; limit > 0; limit >>= 1) 
  {
    if ( threadIdx .x < limit) 
    {
      partialInner[threadIdx.x] += partialInner[threadIdx .x + limit];
      //printf("Current Value of Limit: %d | Local thread id: %d | Local Element Value: %f\n",limit, threadIdx.x, partialInner[threadIdx.x]);
    }
    __syncthreads();
  }

  if(threadIdx.x == 0)
  {
    c[blockIdx.x] = partialInner[0];
    //printf("Cuda Out: %f\n",c[blockIdx.x]);
    //printf("Partial Inner: %f\n",partialInner[threadIdx.x]);
  }

}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  //long N = (1UL<<25); // 2^25
  long N = (1UL<<18); // 2^10 //!Not working for more than 2^18, check it and fix it
  N_items = N;
  BLOCKS_COUNT = N_items/THREADS_PER_BLOCK;

  std::cout << "Blocks Count: " << BLOCKS_COUNT <<  " |Threads per block: " << THREADS_PER_BLOCK << " |No of entities in vector: " << N_items << std::endl;
  //printf("BLOCKS COUNT: %d\n",BLOCKS_COUNT);
  //printf("Threads per block: %d\n",THREADS_PER_BLOCK);
  //printf("N_items, N: %d, %d\n",N_items,N);


  //!For 1d vectors
  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));

  //!Scalers
  double* z_compCuda = (double*) malloc(BLOCKS_COUNT * sizeof(double));
  double* z_cpu = (double*) malloc(1 * sizeof(double));


  //#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) 
  {
      x[i] = drand48();
      y[i] = drand48();
  }
  z_compCuda[0] = 0.;
  z_cpu[0] = 0.;

  double tt = omp_get_wtime();
  vec_mult(z_cpu, x, y, N_items);
  //printf("True output = %f\n", z_cpu[0]);
  double seqtTime = omp_get_wtime()-tt;
  printf("CPU:  %f s\n", seqtTime);

  double *x_d, *y_d, *z_cuda;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  Check_CUDA_Error("malloc y failed");

  //!Allocating scaler
  cudaMalloc(&z_cuda, BLOCKS_COUNT*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);

  double ttinner = omp_get_wtime();
  vec_vec_mult_kernel<<<BLOCKS_COUNT, THREADS_PER_BLOCK>>>(z_cuda, x_d, y_d, N_items);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(z_compCuda, z_cuda, BLOCKS_COUNT*sizeof(double), cudaMemcpyDeviceToHost);
  for(int i = 1; i < BLOCKS_COUNT; ++i)
  {
    z_compCuda[0] += z_compCuda[i];
  }
  tt = omp_get_wtime()-tt;
  //printf("Value returned by cuda: %f\n",z_compCuda[0]);
  printf("Outer time GPU: %f s, Inner time GPU: %f\n", tt, ttinner);
  //printf("GPU: %f s, %f s\n", omp_get_wtime()-tt, ttinner);
  printf("Speed Up Outer: %f\n",seqtTime/tt);
  printf("Speed Up Inner: %f\n",seqtTime/ttinner);
  int memoryOperations = 2*N + BLOCKS_COUNT * THREADS_PER_BLOCK * 10 +  BLOCKS_COUNT;
  double bandwidth = memoryOperations * sizeof(double) /(tt * 1000000000);
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;


  //printf("Speed Up without memory copying time: %f\n",seqtTime/(ttinner));  

  //double err = 0;
  //for (long i = 0; i < N; i++) err += fabs(z_compCuda[i]-z_cpu[i]);
  double err = fabs(z_compCuda[0]-z_cpu[0]);  
  printf("Error = %f\n", err);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_cuda);

  free(x);
  free(y);
  free(z_compCuda);
  free(z_cpu);

  return 0;
}