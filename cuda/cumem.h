/* Copyright (c) 2026 Gilad Odinak */
/* GPU Memory allocation functions */

#ifndef MEM_H
#define MEM_H
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" { /* nvcc is a c++ compiler */
#endif

void* cuallocmem_int(size_t M, size_t N, int S, const char* typ,
                     const char* func, const char* file, int line);

/* Allocates contiguous CUDA device memory for an array of MxN elements of
 * type T, and initializes all bytes to zero.
 *
 * Parameters:
 *   M : Number of rows in the array.
 *   N : Number of columns in the array.
 *   T : Type of the elements in the array (e.g., int, float, double, char).
 *
 * Returns:
 *   A pointer to the allocated CUDA device memory block. If memory allocation
 *   or initialization fails, the function prints an error message and
 *   terminates the program.
 *
 * Examples:
 *   cuallocmem(128, 64, float)
 *   cuallocmem(1, 1000, char)
 */

#define cuallocmem(M,N,T) \
                cuallocmem_int(M,N,sizeof(T),#T,__FUNCTION__,__FILE__,__LINE__)

/* Frees the CUDA device memory allocated by the cuallocmem function.
 *
 * Parameters:
 *   p : Pointer to the CUDA device memory block to be freed.
 *
 * Notice:
 *  It is okay to pass a NULL pointer to this function; in that case, it
 *  does nothing.
 */
inline static void cufreemem(void *p) { if (p != NULL) cudaFree(p); }

#ifdef __cplusplus
}
#endif

#endif
