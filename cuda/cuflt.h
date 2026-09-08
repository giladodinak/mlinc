/* Copyright (c) 2026 Gilad Odinak */
/* Basic CUDA operations for manipulation floating point numbers */
#ifndef CUFLT_H
#define CUFLT_H

#include <cuda_runtime.h>

/* Sets an array of floats in GPU memory to zero.
 * 
 * Sets each element of the array pointed to by v to zero. 
 *
 * Parameters:
 *   v : Pointer to the array of floats to be cleared.
 *   n : Number of elements in the array.
 */
static inline void cufltclr(void* restrict v, int n)
{
    cudaMemset(v,0,n * sizeof(float));
}

/* Copies an array of n floats.
 *
 * Copies n floats from the source array pointed to by s to the destination
 * array pointed to by d. The source and destination arrays should not overlap.
 * The source and destination may each be in CPU or GPU memory.
 *
 * Parameters:
 *   d : Pointer to the destination array.
 *   s : Pointer to the source array.
 *   n : Number of elements to copy.
 */
static inline void cufltcpy(void* restrict d, const void* restrict s, int n)
{
    cudaMemcpy(d,s,n * sizeof(float),cudaMemcpyDefault);
}

#endif
