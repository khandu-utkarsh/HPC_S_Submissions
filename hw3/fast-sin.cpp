#include <math.h>
#include <stdio.h>
#include "utils.h"
#include <iostream>
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static  double c3  = -1/(((double)2)*3);
static  double c5  =  1/(((double)2)*3*4*5);
static  double c7  = -1/(((double)2)*3*4*5*6*7);
static  double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static  double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + c9*x^9 + c11*x^11


// coefficients in the Taylor series expansion of sin(x)
static  double c2  = -1/(((double)2));
static  double c4  =  1/(((double)2)*3*4);
static  double c6  = -1/(((double)2)*3*4*5*6);
static  double c8  =  1/(((double)2)*3*4*5*6*7*8);
static  double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6*x^6 + c8*x^8 + c10*x^10


// Functions for computing reference values -- for comparison
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}
void cos4_reference(double* cosx, const double* x) {
  for (long i = 0; i < 4; i++) cosx[i] = cos(x[i]);
}
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //

// Functions using unvectorized talyor series reference values
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}
void cos4_taylor(double* cosx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x0 = 1;
    double x1  = x[i];
    double x2  = x1 * x1;
    double x4  = x2 * x2;
    double x6  = x4 * x2;
    double x8  = x6 * x2;
    double x10  = x8 * x2;

    double s = x0;
    s += x2  * c2;
    s += x4  * c4;
    s += x6  * c6;
    s += x8  * c8;
    s += x10 * c10;
    cosx[i] = s;
  }
}
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //


// Functions using intrin datstructures for vectorization
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x1, x2, x3;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);

//  std::cout << "Inside AVX Function" << std::endl;

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)

//  std::cout << "Inside SSE Function" << std::endl;

   int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3, x5,x7, x9, x11;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5  = _mm_mul_pd(x3, x2);
    x7  = _mm_mul_pd(x5, x2);
    x9  = _mm_mul_pd(x7, x2);
    x11  = _mm_mul_pd(x9, x2);
    __m128d s = x1;

    // s = 
    // _mm_add_pd(// x1 + x3 + x5 + x7 + x9
    //             _mm_add_pd(     // x1 + x3 + x5 + x7 + x9
    //                         _mm_add_pd( // x1 + x3 + x5 + x7
    //                                     _mm_add_pd(  // x1 + x3 + x5
    //                                                 _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 ))),  //!x1 and x3
    //                                               _mm_mul_pd(x5 , _mm_set1_pd(c5 ))), // x1 + x3 + x5
    //                                   _mm_mul_pd(x7 , _mm_set1_pd(c7 ))), // x1 + x3 + x5 + x7
    //                       _mm_mul_pd(x9 , _mm_set1_pd(c9 ))),  // x1 + x3 + x5 + x7 + x9
    //           _mm_mul_pd(x11 , _mm_set1_pd(c11 ))); // x1 + x3 + x5 + x7 + x9
    // _mm_store_pd(sinx+i, s);

    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 )));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}
void cos4_intrin(double* cosx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  __m256d x0, x1, x2;
  x0  = _mm256_set1_pd(1.);
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);

//  std::cout << "Inside AVX Function" << std::endl;

  __m256d s = x0;
  s = _mm256_add_pd(s, _mm256_mul_pd(x2 , _mm256_set1_pd(c2 )));
  _mm256_store_pd(cosx, s);
#elif defined(__SSE2__)

//  std::cout << "Inside SSE Function" << std::endl;

   int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x0, x1, x2, x4, x6,x8, x10;
    x0  = _mm_set1_pd (1.);
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x4  = _mm_mul_pd(x2, x2);
    x6  = _mm_mul_pd(x4, x2);
    x8  = _mm_mul_pd(x6, x2);
    x10  = _mm_mul_pd(x8, x2);
    __m128d s = x0;

    s = _mm_add_pd(s, _mm_mul_pd(x2 , _mm_set1_pd(c2 )));
    s = _mm_add_pd(s, _mm_mul_pd(x4 , _mm_set1_pd(c4 )));
    s = _mm_add_pd(s, _mm_mul_pd(x6 , _mm_set1_pd(c6 )));
    s = _mm_add_pd(s, _mm_mul_pd(x8 , _mm_set1_pd(c8 )));
    s = _mm_add_pd(s, _mm_mul_pd(x10 , _mm_set1_pd(c10 )));
    _mm_store_pd(cosx+i, s);
  }
#else
  cos4_reference(cosx, x);
#endif
}
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //

// Function using custom vector class
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s.StoreAligned(sinx);
}

//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //


// Function for computing error
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
//  ------------------------------------------------------------------------------------------------------------------------------------------------------  //
double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}



//!Main -- Driver function
int main() {
  Timer tt;
  long N = 1000000;
  //long N = 1;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* x_red = (double*) aligned_malloc(N*sizeof(double));

  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));

  double* cosx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* cosx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* cosx_intrin = (double*) aligned_malloc(N*sizeof(double));

  //!Iota and Multiplication factor
  int* iFactors = (int*) aligned_malloc(N*sizeof(int));
  int* mFactors = (int*) aligned_malloc(N*sizeof(int));

  for (long i = 0; i < N; i++) {
    //x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4] //!Given
    x[i] = (drand48()-0.5) * 4 * M_PI; // [-2pi,2pi]
    x_red[i] = x[i];
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;

    cosx_ref[i] = 0;
    cosx_taylor[i] = 0;
    cosx_intrin[i] = 0;


    //!Computing this iota and multiplication factor
    iFactors[i] = 1;
    mFactors[i] = 0;
    int iotaFactor = 0;
    int multiplicationFactor = 1;
    {

      if(x_red[i] < 0)
      {
        multiplicationFactor = -1;
        x_red[i] = multiplicationFactor * x_red[i];
      }

      while(x_red[i] > M_PI/4)
      {
        x_red[i] = x_red[i] - M_PI/2;
        iotaFactor++;
      }

      if(iotaFactor%4 == 0)
      {
        iotaFactor = 0;
        multiplicationFactor *= 1; 
      }
      else if(iotaFactor%4 == 1)
      {
        iotaFactor = 1;
        multiplicationFactor *= 1; 
      }
      else if(iotaFactor%4 == 2)
      {
        iotaFactor = 0;
        multiplicationFactor *= -1; 
      }
      else if(iotaFactor%4 == 3)
      {
        iotaFactor = 1;
        multiplicationFactor *= -1;
      }
    }
    iFactors[i] = iotaFactor;
    mFactors[i] = multiplicationFactor;

  }

  //   //!Processing x array, so that it is in the range of [-pi/4 to pi/4] and noting how many multiplication of i are done

  // for (long i = 0; i < N; i++) 
  // {
  //   int iotaFactor = 0;
  //   int multiplicationFactor = 1;
  //   if(x_red[i] < 0)
  //   {
  //     multiplicationFactor = -1;
  //     x_red[i] = multiplicationFactor * x_red[i];
  //   }

  //   while(x_red[i] > M_PI/4)
  //   {
  //     x_red[i] = x_red[i] - M_PI/2;
  //     iotaFactor++;
  //   }

  //   if(iotaFactor%4 == 0)
  //   {
  //     iotaFactor = 0;
  //     multiplicationFactor *= 1; 
  //   }
  //   else if(iotaFactor%4 == 1)
  //   {
  //     iotaFactor = 1;
  //     multiplicationFactor *= 1; 
  //   }
  //   else if(iotaFactor%4 == 2)
  //   {
  //     iotaFactor = 0;
  //     multiplicationFactor *= -1; 
  //   }
  //   else if(iotaFactor%4 == 3)
  //   {
  //     iotaFactor = 1;
  //     multiplicationFactor *= -1;
  //   }
  //   iFactors[i] = iotaFactor;
  //   mFactors[i] = multiplicationFactor;
  // }

  // std::cout << "i Factor "<< iFactors[0] << std::endl;
  // std::cout << "m Factor "<< mFactors[0] << std::endl;  
  // std::cout << "Original x "<< x[0] << std::endl;
  // std::cout << "Reduced x "<< x_red[0] << std::endl;


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) 
    {      
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  auto refTime = tt.toc();
  printf("Reference time: %6.4f\n", refTime);

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4)
    {
      sin4_taylor(sinx_taylor+i, x_red+i);
      cos4_taylor(cosx_taylor+i, x_red+i);
    }
    //!Keeping it out so that blocking can be done by compiler
    for (long i = 0; i < N; i++)
    {
      if(iFactors[i] == 1) //!Pick cos value
      {
        sinx_taylor[i] = cosx_taylor[i] * mFactors[i];
      }
      else
      {
        sinx_taylor[i] = sinx_taylor[i] * mFactors[i];
      }
    }
  }
  auto talyorTime = tt.toc();
  printf("Taylor time:    %6.4f      Error: %e\n", talyorTime, err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) 
    {
      sin4_intrin(sinx_intrin+i, x_red+i);
      cos4_intrin(cosx_intrin+i, x_red+i);
    }
    for (long i = 0; i < N; i++)
    {  
      if(iFactors[i] == 1) //!Pick cos value
      {
        sinx_intrin[i] = cosx_intrin[i] * mFactors[i];
      }
      else
      {
        sinx_intrin[i] = sinx_intrin[i] * mFactors[i];
      }
    }
  }
  auto intrinTime = tt.toc();
  printf("Intrin time:    %6.4f      Error: %e\n", intrinTime, err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  std::cout << "Speed Up compared to Talyor Time: " << talyorTime/intrinTime << std::endl;
  std::cout << "Speed Up compared to Reference Time (C++ built in fxn): " << refTime/intrinTime << std::endl;


  aligned_free(x);
  aligned_free(x_red);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
  aligned_free(cosx_ref);
  aligned_free(cosx_taylor);
  aligned_free(cosx_intrin);


  aligned_free(iFactors);
  aligned_free(mFactors);
}
