#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

inline void CopyFromNewToCurr(int p, long N, double *u_curr, double * u_new)
{
    #pragma omp parallel for num_threads(p) 
    for(long i = 0; i < N*N; ++i)
    {
        u_curr[i] = u_new[i];
    }
}


double maxNorm(long N, double * u_curr, double * u_new)
{
    double maxValue = -DBL_MAX;
    for(long iRow = 0; iRow < N * N; ++iRow)
    {
        double diff = fabs(u_new[iRow] - u_curr[iRow]);
        if(diff > maxValue)
        {
            maxValue = diff;
        }
    }
    return maxValue;
}


int main(int argc, char** argv) 
{
    //int nrThreads[10] = {1, 4, 8, 12, 16, 29, 24, 28, 32, 36, 40, 48, 64};

    long N = 100;
    long Ne = N + 2;
    long maxIterations = 2000;
    //Discretization Points
    double h = 1./(N+1);


    //!Row major storage,
    double* f = (double*) malloc(Ne * Ne * sizeof(double)); // N
    double* u_curr = (double*) malloc(Ne * Ne * sizeof(double)); //u
    double* u_new = (double*) malloc(Ne * Ne * sizeof(double)); //u_new

    double seqTime = 1;
    for(int nrTIndex = 0; nrTIndex < 18; nrTIndex++) 
    {
        int nrT = 1;
        if(nrTIndex == 0)
            nrT = 1;
        else
            nrT = 4 * nrTIndex;

        //!Initialize f values with 1;
        #pragma omp parallel for num_threads(nrT)
        for (long i = 0; i < Ne*Ne; i++)
        {
            f[i] = 1;
            u_curr[i] = 0;
            u_new[i] = 0;
        }   

        //!Loop over iterations
        //bool converged = false;
        //double finalError = 0.0;
#ifdef _OPENMP
        double t = omp_get_wtime();
#endif
        for(long iter = 0; iter < maxIterations; ++iter)
        {
            //std::cout << "Curr ItCount: " << iter << std::endl;
            //!Go to each row and then each column
            #pragma omp parallel for num_threads(nrT) collapse(2)
            for(long rowIndex = 1; rowIndex < N+1; ++rowIndex)
            {
                for(long columnIndex = 1; columnIndex < N+1; ++columnIndex)
                {
                    //std::cout << "Curr ItCount: " << iter << std::endl;
                    if((rowIndex + columnIndex)%2 == 0) //!Red Point
                    {
                        //std::cout << "Row Index: " << rowIndex << " Column Index: " << columnIndex << std::endl;                    
                        u_new[rowIndex * Ne + columnIndex] = 1./4. * (h * h * f[rowIndex * Ne + columnIndex] + 
                                                                    u_curr[(rowIndex - 1) * Ne+ columnIndex] + 
                                                                    u_curr[rowIndex * Ne + (columnIndex - 1)] +
                                                                    u_curr[(rowIndex + 1) * Ne + columnIndex] + 
                                                                    u_curr[rowIndex * Ne + (columnIndex + 1)]);
                    }
                }
            }
            //!At this stage, all info for the red points have been updated, now we can update the y points
            #pragma omp parallel for num_threads(nrT) collapse(2)
            for(long rowIndex = 1; rowIndex < N+1; ++rowIndex)
            {
                for(long columnIndex = 1; columnIndex < N+1; ++columnIndex)
                {
                    if((rowIndex + columnIndex)%2 != 0) //!Black Points
                    {
                        u_new[rowIndex * Ne + columnIndex] = 1./4. * (h * h * f[rowIndex * Ne + columnIndex] + 
                                                                    u_new[(rowIndex - 1 ) * Ne+ columnIndex] + 
                                                                    u_new[rowIndex * Ne + (columnIndex - 1)] +
                                                                    u_new[(rowIndex + 1) * Ne + columnIndex] + 
                                                                    u_new[rowIndex * Ne + (columnIndex + 1)]);
                    }
                }
            }
            //!All row and columns done, update the curr with the new data and proceed afterwards
            // double error = maxNorm(Ne, u_curr, u_new);
            // std::cout << "Iteration Count: " << iter+1 << " Current Residual: " << error << std::endl;
            // if(error < 1e-6)
            // {
            //     converged = true;
            //     finalError = error;
            //     std::cout << "Converged  " << "Iterations Count: " << iter+1 << " Final Residual: " << error << std::endl;            
            //     break;
            // }

            CopyFromNewToCurr(nrT, Ne, u_curr, u_new);
        }
#ifdef _OPENMP
        t = omp_get_wtime() - t;
        if(nrT == 1)
            seqTime = t;
        printf("Gauss Seidel Method, N = %d, Num threads = %d, time elapsed = %f, speedup = %f\n", N,nrT, t, seqTime/t);
#endif
    }
    // if(converged == false)
    // {
    //     std::cout << "Didn't converged  " << " Final Residual: " << finalError << std::endl;                    
    // }
    free(f);
    free(u_curr);
    free(u_new);
    return 0;
}
