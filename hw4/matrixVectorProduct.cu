// nvcc -std=c++11  matrixVectorProduct.cu -o matrixVectorProduct -Xcompiler -fopenmp
// $ nvcc -arch=sm_61 gpu03.cu -o gpu03 -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <iostream>

const int THREADS_PER_BLOCK = 1024;
// int BLOCKS_COUNT = 16;
// //int N_items = BLOCKS_COUNT * THREADS_PER_BLOCK;


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

void matrix_matrix_mult(double* b, const double* A, const double* x, long N, long rowsCount)
{
  //#pragma omp parallel for schedule(static)
  for(int iRow = 0; iRow < rowsCount; ++iRow)
  {
    // printf("Current Row: %d\n",iRow);
    // for(int i = 0; i < N; ++i)
    // {
    //   printf("b: %f, A: %f, x: %f\n",b[iRow + i],A[iRow * N + i], x[i]);
    // }
    vec_mult(&b[iRow], &A[iRow * N], x, N);
  } 
}

__global__
void matrix_matrix_mult_kernel(double* b, const double* A, const double* x, long N, long rowsCount)
{
  //!c should be of size 1
  //!a and b should be of same size

  //!Creating a shared variable for a block
  __shared__ double partialInner[THREADS_PER_BLOCK];

  //!Global threads index
  int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(global_thread_index < N * rowsCount)
  {
    partialInner[threadIdx.x] = A[global_thread_index] * x[global_thread_index % N];
    //printf("Cuda thread id: %d | Local Prod: %f |Local Index: %d |Curr Local c: %f\n",global_thread_index, A[global_thread_index] * x[global_thread_index%N], threadIdx.x, partialInner[threadIdx.x]);
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


  __syncthreads();
  if(threadIdx.x == 0)
  {
    b[blockIdx.x] = partialInner[0];
    //printf("Cuda Out: %f, | Block Index: %d\n",b[blockIdx.x], blockIdx.x);
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

int main()
{
  //int THREADS_PER_BLOCK = 4;
  long N = (1UL<<12);
  long rowsCount = (1UL<<11);
  
  int BLOCKS_COUNT = N * rowsCount/THREADS_PER_BLOCK;
  //int N_items = BLOCKS_COUNT * THREADS_PER_BLOCK;


    //long N = (1UL<<25); // 2^25
    //long N = (1UL<<18); // 2^10 //!Not working for more than 2^18, check it and fix it
    //N_items = N;

    //!Columns
    //int factor = 2;
    //int N = THREADS_PER_BLOCK * factor;
    //int rowsCount = BLOCKS_COUNT / N;
    std::cout << "Blocks Count: " << BLOCKS_COUNT <<  " |Threads per block: " << THREADS_PER_BLOCK << " |No of columns: " << N << " |No of rows: " << rowsCount << std::endl;

    // printf("BLOCKS COUNT: %d\n",BLOCKS_COUNT);
    // printf("Threads per block: %d\n",THREADS_PER_BLOCK);
    // printf("Columns, N: %d | Rows: %d\n",N, rowsCount);

    //!Number of columns be N
    //!Number of rows be M

    //long rowsCount = (1UL<<18);

    //!For 1d vectors
    double* x = (double*) malloc(N * sizeof(double));

    //!For 2d matrices, storing it as a row major,
    double* A = (double*) malloc((rowsCount * N) * sizeof(double));

    //!For output vectors
    double* b = (double*) malloc(rowsCount * sizeof(double));

    //!For output vectors
    double* b_cuda_copied = (double*) malloc(BLOCKS_COUNT * sizeof(double));

    //!Initializing the memory for the matrix and vector
    //#pragma omp parallel for schedule(static)
    for (long i = 0; i < rowsCount * N; i++) {A[i] = drand48();}
    //#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {x[i] = drand48();}
    //#pragma omp parallel for schedule(static)
    for (long i = 0; i < rowsCount; i++)
    {
      b[i] = 0.;
    }
    
    // for (long i = 0; i < rowsCount * N; i++) 
    // {
    //   printf("Matrix Element Index: %d, Matrix Element Value: %f\n",i, A[i]);//drand48();}
    // }
    // for (long i = 0; i < N; i++)
    // {
    //   printf("x vector element: %d, x vector element value: %f\n",i, x[i]);
    // }

    //double* z_compCuda = (double*) malloc(BLOCKS_COUNT * sizeof(double));
    //z_compCuda[0] = 0.;

    double seqt = omp_get_wtime();
    matrix_matrix_mult(b, A, x, N, rowsCount);
    seqt = omp_get_wtime() - seqt;
    //printf("True output = %f\n", z_cpu[0]);
    printf("CPU %f s\n", seqt);

    //!Allocating memory for CUDA
    double *x_d, *A_d, *b_cuda;
    
    cudaMalloc(&x_d, N*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    
    cudaMalloc(&A_d, (rowsCount * N)*sizeof(double));
    Check_CUDA_Error("malloc A failed");

    //!Allocating for output
    cudaMalloc(&b_cuda, BLOCKS_COUNT*sizeof(double));

    double tt = omp_get_wtime();
    cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, (rowsCount * N)*sizeof(double), cudaMemcpyHostToDevice);

    double ttinner = omp_get_wtime();
    matrix_matrix_mult_kernel<<<BLOCKS_COUNT, THREADS_PER_BLOCK>>>(b_cuda, A_d, x_d, N, rowsCount);
    cudaDeviceSynchronize();

    //printf("Exited cuda kernel\n");


    ttinner = omp_get_wtime() - ttinner;
    cudaMemcpy(b_cuda_copied, b_cuda, BLOCKS_COUNT*sizeof(double), cudaMemcpyDeviceToHost);

    //!Fix this function
    for(int iRow = 0; iRow < rowsCount; ++iRow)
    {
      double localSum = 0.;
      for(int iBlock = 0; iBlock < BLOCKS_COUNT/rowsCount; ++iBlock)
      {
        int currentBlockIndex = iRow * BLOCKS_COUNT/rowsCount + iBlock;
        //printf("Current Block Index: %d, Current Row Index: %d, Current Local Block Index: %d\n", currentBlockIndex, iRow, iBlock);
        localSum += b_cuda_copied[currentBlockIndex];
      }
      b_cuda_copied[iRow * BLOCKS_COUNT/rowsCount] = localSum;
    }
    tt = omp_get_wtime() -tt;
    // for(int i = 0; i < BLOCKS_COUNT; ++i)
    // {
    //   printf("Index: %d, Vlaue: %f\n",i,b_cuda_copied[i]);
    // }
    printf("Outer time GPU: %f s, Inner time GPU: %f\n", tt, ttinner);
    printf("Speed Up Outer: %f\n",seqt/tt);
    printf("Speed Up Inner: %f\n",seqt/ttinner);
    
    int memoryOperations = N + N * rowsCount + BLOCKS_COUNT * THREADS_PER_BLOCK * 10 +  BLOCKS_COUNT;
    double bandwidth = memoryOperations * sizeof(double) /(tt * 1000000000);
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    double err = 0.;
    for(long iRow = 0; iRow < rowsCount; ++iRow)
    {
      err += fabs(b_cuda_copied[iRow * BLOCKS_COUNT/rowsCount] - b[iRow]);
    }

    // for(long iRow = 0; iRow < rowsCount; ++iRow)
    // {
    //   printf("Cpu | Index: %d, Value: %f\n",iRow, b[iRow]);
    //   printf("Gpu | Index: %d, Value: %f\n",iRow * BLOCKS_COUNT/rowsCount, b_cuda_copied[iRow * BLOCKS_COUNT/rowsCount]);
    // }


    //double err = 0;
    //for (long i = 0; i < N; i++) err += fabs(z_compCuda[i]-z_cpu[i]);
    //double err = fabs(b_cuda_copied[0]-b_cuda[0]);
    printf("Error = %f\n", err);

    cudaFree(x_d);
    cudaFree(A_d);
    cudaFree(b_cuda);


    free(x);
    free(A);
    free(b);
    free(b_cuda_copied);

  return 0;
}