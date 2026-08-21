/* Copyright (c) 2023-2024 Gilad Odinak */
/* Basic operations for manipulation floating point numbers */
#ifndef FLOAT_H
#define FLOAT_H

#ifdef USE_DOUBLE
#define float   double
#define fabsf   fabs
#define logf    log
#define log10f  log10
#define log1pf  log1p
#define expf    exp
#define FMTF "%lf"
#else
#define FMTF "%f"
#endif

#include <memory.h> /* memcpy() memmove() memset() */

/* Sets an array of floats to zero.
 * 
 * Sets each element of the array pointed to by v to zero. 
 * It is not guaranteed that (float) 0.0 is represented as all zero bytes, 
 * so memset() is not used.
 *
 * Parameters:
 *   v : Pointer to the array of floats to be cleared.
 *   n : Number of elements in the array.
 */
static inline void fltclr(void* restrict v_, int n)
{
    float* restrict v = (float* restrict) v_;
    for (int i = 0; i < n; i++) v[i] = 0.0;
}

/* Copies an array of n floats.
 *
 * Copies n floats from the source array pointed to by s to the destination
 * array pointed to by d. The source and destination arrays should not overlap.
 *
 * Parameters:
 *   d : Pointer to the destination array.
 *   s : Pointer to the source array.
 *   n : Number of elements to copy.
 */
static inline void fltcpy(void* restrict d, const void* restrict s, int n)
{
    memcpy(d,s,n * sizeof(float));
}

/* Moves an array of floats; like fltcpy but overlapping move is okay.
 *
 * Moves n floats from the source array pointed to by s to the destination
 * array pointed to by d. Unlike fltcpy, overlapping of source and destination
 * arrays is allowed.
 *
 * Parameters:
 *   d : Pointer to the destination array.
 *   s : Pointer to the source array.
 *   n : Number of elements to move.
 */
static inline void fltmove(void* d, const void* s, int n)
{
    memmove(d,s,n * sizeof(float));
}

#endif
