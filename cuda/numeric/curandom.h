/* Copyright (c) 2023-2026 Gilad Odinak */
/* Random number generation routines    */
#ifndef CURANDOM_H
#define CURANDOM_H
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* cu_lrng returns a pseudo-random real number uniformly distributed
 * between 0.0 and 1.0, exclusive on both ends.
 * Lehmer random number generator - Steve Park & Dave Geyer
 * https://www.cs.wm.edu/~vjtorc/software/park/rngs.c
 */
static inline __host__ __device__ float cu_lrng(int* restrict seed)
{
    const int32_t modulus = 2147483647; /* 0x7FFFFFFF Prime number */
    const int32_t multiplier = 48271;   /* Another Prime number    */
    const int32_t q = modulus / multiplier;
    const int32_t r = modulus % multiplier;
    int32_t t = multiplier * (*seed % q) - r * (*seed / q);
    int32_t v = (t > 0) ? t : t + modulus;
    *seed = v;
    float num = ((float) v / modulus);
    if (num >= 1.0f) num = nextafterf(1.0f,0.0f); /* Avoid float rounding */
    return num;
}

/* Return a random number following a uniform distribution
 */
static inline __host__ __device__ float cu_urand(int* restrict seed, float min, float max)
{
    float num = cu_lrng(seed) * (max - min) + min;
    return num;
}

/* Return a random number following a normal distribution
 * with the provided mean and standard deviation.
 */
static inline __host__ __device__ float cu_nrand(int* restrict seed, float mean, float stddev)
{
    /* Box-Muller transform */
    float r1 = cu_lrng(seed);
    if (r1 < 1e-13) r1 = 1e-13;
    float r2 = cu_lrng(seed);
    float z = sqrt(-2.0 * log(r1)) * sin(2.0 * M_PI * r2);
    float num =  mean + stddev * z; /* Shift and scale */
    return num;
}

#endif
