#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>



//#define THREADS_COUNT 40
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }

  //!Printing sequential
  //  std::cout << "Printing sequential " << std::endl;
  // for (long i = 0; i < n; i++)
  //{
  // std::cout << "Index i = " << i << "  PS Value = " << prefix_sum[i] << std::endl;
  // }
}

void scan_omp(long* prefix_sum, const long* A, long n, int threadsCount) {
  // TODO: implement multi-threaded OpenMP scan

  double eachChunkSize = n/double(threadsCount);
  long chunkSize = std::ceil(eachChunkSize);
  //std::cout << "Chunk Size is " << chunkSize << std::endl;

  prefix_sum[0] = 0;
  //double ps = omp_get_wtime();
  #pragma omp parallel shared(A, prefix_sum, n, chunkSize) num_threads(threadsCount)
  {
    #pragma omp single //!This will ensure that single thread creates tasks, which will then be executed by all threads
    {
      for(long iThread = 0; iThread < threadsCount; ++iThread)
      {
          #pragma omp task
          {
            long start_index = 0;
            if(iThread == 0)
            {
              start_index = iThread * chunkSize + 1;
            }
            else
            {
              start_index = iThread * chunkSize;
            }
            long end_index = std::min((iThread +1) * chunkSize, n);
            for(long currInd = start_index; currInd < end_index; ++currInd)
            {
              if(currInd == start_index)
              {
                prefix_sum[currInd] = A[currInd -1];
              }
              else
              {
                prefix_sum[currInd] = prefix_sum[currInd - 1] + A[currInd - 1];
              }
            }
          }
      
      }
    }
  }
  //double pe = omp_get_wtime();
  //std::cout << "Time Elapsed in parallel function: " << pe - ps << std::endl;
  
  // std::cout << "Prefix Sum, before correction: " << std::endl;
  // for (long i = 0; i < n; i++)
  // {
  //   std::cout <<"Index i = " << i << "  PS Value = " << prefix_sum[i] << std::endl;
  // }

  //!Correction Loop
  //double cs = omp_get_wtime();
  for(long iThread = 1; iThread < threadsCount; ++iThread)
  {
    long start_index = iThread * chunkSize;
    long end_index = std::min((iThread +1) * chunkSize, n);
    long corrTerm = prefix_sum[start_index-1];
    //std::cout << "start index " << start_index << " end index " << end_index << std::endl;
    #pragma omp parallel shared(A, prefix_sum, n, chunkSize, corrTerm) num_threads(threadsCount)
    {
      #pragma omp for      
      for(long currInd = start_index; currInd < end_index; ++currInd)
      {
        prefix_sum[currInd]  += corrTerm;
      }
    }
  }
  //double ce = omp_get_wtime();
  //std::cout << "Time Elapsed in correction function: " << ce - cs << std::endl;

  // std::cout << "Prefix Sum, after correction: " << std::endl;
  // for (long i = 0; i < n; i++)
  // {
  //   std::cout <<"Index i = " << i << "  PS Value = " << prefix_sum[i] << std::endl;
  // }


}

int main() {

  for(int th = 1; th < 32; ++th)
  {
    long N = 100000000; //!Given
    //long N = 20; //!Testing
    long* A = (long*) malloc(N * sizeof(long));
    long* B0 = (long*) malloc(N * sizeof(long));
    long* B1 = (long*) malloc(N * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = rand(); //!Given
    //for (long i = 0; i < N; i++) A[i] = i; //!Testing

    double tt = omp_get_wtime();
    scan_seq(B0, A, N);
    double seq = omp_get_wtime() - tt;
  //  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

    tt = omp_get_wtime();
    scan_omp(B1, A, N, th);
    double par = omp_get_wtime() - tt;
  //  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);


    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  //  printf("error = %ld\n", err);

  //  std::cout << "Speed up = " << seq/par << std::endl;
    std::cout << "Threads Used: " << th << " | Seq Time: " << seq << " | Parallel Time: " << par << " | Error: " << err << "| Speed Up: " << seq/par << std::endl;

    free(A);
    free(B0);
    free(B1);

  }
  return 0;
}

