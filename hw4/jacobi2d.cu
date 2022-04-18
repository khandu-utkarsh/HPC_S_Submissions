// nvcc -std=c++11  jacobi2d.cu -o jacobi2d -Xcompiler -fopenmp

#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <algorithm>



//Seq code

void seqJacobiIteration(double * u_new, double * u_curr,double * f, int Ne, int N_dis, double h)
{
    //std::cout << "Inside seq" << std::endl;
    
    //!Go to each row and then each column
    for(long rowIndex = 1; rowIndex < N_dis+1; ++rowIndex)
    {
        for(long columnIndex = 1; columnIndex < N_dis+1; ++columnIndex)
        {
            u_new[rowIndex * Ne + columnIndex] = 1./4. * (h * h * f[rowIndex * Ne + columnIndex] + 
                                                        u_curr[(rowIndex - 1 ) * Ne+ columnIndex] + 
                                                        u_curr[rowIndex * Ne + (columnIndex - 1)] +
                                                        u_curr[(rowIndex + 1) * Ne + columnIndex] + 
                                                        u_curr[rowIndex * Ne + (columnIndex + 1)]);
            //std::cout << "Index Seq: " << rowIndex * Ne + columnIndex << "  Value: " <<u_new[rowIndex * Ne + columnIndex] << std::endl;
        }
    }
}


__global__
void jacobi_kernel(double* u_in, double* u_out, const double* f, double * u_error, long N, long N_row, long N_col, double h2)
{
    //printf("Inside kernel\n");
    //!Current Block Index
    int threadColIndex = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int threadRowIndex =1 +  blockIdx.y * blockDim.y + threadIdx.y;

    //!Memory index,
    int memIndx = threadRowIndex * N_col + threadColIndex;

    if(memIndx >= N)    return;
    //!Indexing is correct


    //printf("Row Id: %d | Col Id: %d |Memory Index: %d\n",threadRowId,threadColId,memIndx);
    //printf("Block Dim in x: %d, Block Dim in y: %d\n",blockDim.x, blockDim.y);
    
    // printf("MemIndex: %d | X Block Idx : %d | Y Block Idx: %d | X Thread Idx: %d | Y Thread Idx: %d |, (i, j): (%d,%d)|\n",
    //         memIndx,
    //         blockIdx.x,
    //         blockIdx.y,
    //         threadIdx.x,
    //         threadIdx.y,
    //         threadRowIndex,
    //         threadColIndex);
  
    u_out[memIndx] =   0.25 * ( h2* f[threadRowIndex * N_col + threadColIndex] + 
                        u_in[(threadRowIndex -1) * N_col + threadColIndex] + 
                        u_in[(threadRowIndex +1) * N_col + threadColIndex] +
                        u_in[(threadRowIndex) * N_col + threadColIndex + 1] +
                        u_in[(threadRowIndex) * N_col + threadColIndex - 1]);
     //u_out[memIndx] =    f[threadRowIndex * N_col + threadColIndex]; 
    //                     u_in[(threadRowIndex -1) * N_col + threadColIndex] + 
    //                     u_in[(threadRowIndex +1) * N_col + threadColIndex] +
    //                     u_in[(threadRowIndex) * N_col + threadColIndex + 1] +
    //                     u_in[(threadRowIndex) * N_col + threadColIndex - 1];

    //printf("Index: %d, Computed U_out: %f\n",memIndx,u_out[memIndx]);

    //printf("U_updated value: %f | Index: %f,\n",u_out[memIndx],memIndx);
    
    __syncthreads();  //!Waiting for all the threads of a block to finish up.

    //!Check for error
    //printf("Computing error")
    u_error[memIndx] = fabs(u_out[memIndx] - u_in[memIndx]);
    u_in[memIndx] = u_out[memIndx];
    //printf("Index: %d| Value: %f\n",memIndx, u_in[memIndx]);
    // for(int i = 0; i < N; ++i)
    // {
    //     u_in[i] = u_out[i];

    // }
}


void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}


int main(int argc, char** argv) 
{
    int N_discpts = 1024;
    int N_row = N_discpts + 2;
    int N_column = N_discpts + 2;
    int N = N_row * N_column;
    double h = 1./(N_discpts + 1);
    double h2 = h*h;
    printf("Number of discretization points: %d\n",N_discpts);
    //printf("Rows: %d| Columns: %d\n",N_row, N_column);

    //!For 2d
    double* u_host_in = (double*) malloc(N * sizeof(double));
    double* u_host_out = (double*) malloc(N * sizeof(double));
    double* f_host = (double*) malloc(N * sizeof(double));
    double* u_cuda_output = (double*) malloc(N * sizeof(double));
    double* u_error = (double*) malloc(N * sizeof(double));

    //!Initializing the memory for the u and f;
    //#pragma omp parallel for schedule(static)
    for(int i = 0; i < N; ++i)
    {
        u_host_in[i] = 0.;
        u_host_out[i] = 0.;
        f_host[i] = 1.;
        u_cuda_output[i] = 0.;
    }


    double *u_device_in, *u_device_out, *f_device, *u_error_cuda;

    cudaMalloc(&u_device_in, N*sizeof(double));
    Check_CUDA_Error("malloc u_device_in failed");
    
    cudaMalloc(&u_device_out, N*sizeof(double));
    Check_CUDA_Error("malloc u_device_out failed");
    
    cudaMalloc(&f_device, N*sizeof(double));
    Check_CUDA_Error("malloc f_device failed");

    cudaMalloc(&u_error_cuda, N*sizeof(double));
    Check_CUDA_Error("malloc u_error_device failed");

    //!Copying from host to device
    double cudaTime = 0., seqTime = 0.;
    double t1 = omp_get_wtime();
    cudaMemcpy(u_device_in, u_host_in, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_device, f_host, N*sizeof(double), cudaMemcpyHostToDevice);
    double t2 = omp_get_wtime();
    cudaTime = t2 - t1;
    //cudaMemcpy(u_device_out, u_host_out, N*sizeof(double), cudaMemcpyHostToDevice);


    //!Memory for cuda
    int blockDimInX = 32;
    int blockDimInY = 32;
    dim3 blockShape(blockDimInX,blockDimInY);
    dim3 gridShape(N_discpts/blockDimInX,N_discpts/blockDimInY);

    std::cout << "Gird Shape: " << "Blocks in x_dir: " <<  N_discpts/blockDimInX << " Blocks in y_dir: " << N_discpts/blockDimInY << 
                 " | Block Shape: " << "Threads in x_dir: " << blockDimInX << " Threads in y_dir: " << blockDimInY << std::endl;
    
    double lastIterError = 0.;
    for(int iter = 0; iter < 2000; ++iter)
    {
        double maxError = -1;
        //std::cout << "Calling kernel" << std::endl;
        double t3 = omp_get_wtime();
        jacobi_kernel<<<gridShape,blockShape>>>(u_device_in, u_device_out, f_device, u_error_cuda, N, N_row, N_column, h2);
        cudaDeviceSynchronize();
        cudaMemcpy(u_error, u_error_cuda, N*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(u_cuda_output, u_device_out, N*sizeof(double), cudaMemcpyDeviceToHost);
        double t4 = omp_get_wtime();
        cudaTime += t4 - t3;

        double t5 = omp_get_wtime();
        seqJacobiIteration(u_host_out, u_host_in, f_host,N_column, N_discpts, h);
        double t6 = omp_get_wtime();
        seqTime = t6 - t5;

        // for(int i = 0; i < N; ++i)
        // {
        //     std::cout << "Index: " << i << " |Cuda  "  << u_cuda_output[i] << " |CPU  "  << u_host_out[i] << std::endl;
        // }

        double compareError = -1;
        for(int i = 0; i < N; ++i)
        {
            double diff = u_cuda_output[i] - u_host_out[i];
            if(diff > compareError)
            {
                compareError = diff;
            }
        }
        iter++;

        double t7 = omp_get_wtime();
        for(long i = 0; i < N; ++i)
        {
            u_host_in[i] = u_host_out[i];
        }
        double t8 = omp_get_wtime();
        seqTime += t8 - t7;

        for(int i = 0; i < N; ++i)
        {
            if(maxError < u_error[i])
            {
                maxError = u_error[i];
            }
        }
        lastIterError = maxError;
        //if(iter == 1999 || iter%10 == 0)
        //{
            std::cout << "Iteration: " << iter << " |Seq relative error: " << compareError << "  |Residual Error: " << lastIterError << std::endl;
       // }
    }
    //printf("Exited cuda kernel\n");
    //printf("Maximum Iterations used are: %d\n",iter);

    //std::cout << "|Time taken by sequential algo: " << seqTime << "s" << " |Time taken by cuda algo: " << cudaTime << "s"<<std::endl;
    //std::cout << "Speed Up: " << cudaTime/seqTime << std::endl;

    cudaFree(u_device_in);
    cudaFree(u_device_out);
    cudaFree(f_device);
    cudaFree(u_error_cuda);

    free(u_host_in);
    free(u_host_out);
    free(f_host);
    free(u_cuda_output);
    free(u_error);
    return 0;
}
