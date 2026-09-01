/* Copyright (c) 2023-2024 Gilad Odinak */
/* Random number generation routines    */
#ifndef RANDOM_H
#define RANDOM_H
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Initializes the random number generator */
extern int32_t lrng_seed;
void init_lrng(int seed);
int get_lrng_seed(void);

/* lrng returns a pseudo-random real number uniformly distributed 
 * between 0.0 and 1.0, exclusive on both ends.
 * Lehmer random number generator - Steve Park & Dave Geyer
 */
static inline float lrng(void)
{
    const int32_t modulus = 2147483647; /* 0x7FFFFFFF   */
    const int32_t multiplier = 48271;   /* Prime number */
    const int32_t q = modulus / multiplier;
    const int32_t r = modulus % multiplier;
    int32_t t = multiplier * (lrng_seed % q) - r * (lrng_seed / q);
    int32_t v = (t > 0) ? t : t + modulus;
    lrng_seed = v;
    float num = ((float) v / modulus);
    if (num >= 1.0f) num = nextafterf(1.0f, 0.0f); /* Avoid float rounding */
    return num;
}

/* Return a random number following a uniform distribution
 */
static inline float urand(float min, float max) 
{
    float num = lrng() * (max - min) + min;
    return num;
}

/* Return a random number following a normal distribution
 * with the provided mean and standard deviation.
 */
static inline float nrand(float mean, float stddev) 
{
    /* Box-Muller transform */
    float r1 = lrng();
    if (r1 < 1e-13) r1 = 1e-13;
    float r2 = lrng();
    float z = sqrt(-2.0 * log(r1)) * sin(2.0 * M_PI * r2);
    float num =  mean + stddev * z; /* Shift and scale */
    return num;
}

#endif
