/* Copyright (c) 2023-2024 Gilad Odinak */
/* Memory data structures and functions */

#ifndef MEM_H
#define MEM_H
#include <stdlib.h>  /* declares free() */

void* allocmem_int(size_t M, size_t N, int S, char* typ, 
                                       const char* func, char* file, int line);

/* Allocates contiguous memory for an array of MxN elements of type T,
 * and initializes all elements to zero.
 *  
 * Parameters:
 *   M : Number of rows in the array.
 *   N : Number of columns in the array.
 *   T : Type of the elements in the array (e.g., int, float, double, char).
 * 
 * Returns:
 *   A pointer to the allocated memory block. If memory allocation fails, the
 *   function prints an error message and terminates the program.
 * 
 * Examples:
 *   allocmem(128, 64, float)
 *   allocmem(1, 1000, char)
 * 
 * Notice:
 *   If the representation of float zero (or double) is not binary zero, 
 *   the memory will be correctly initialized to all floating point zeros
 *   only if T is float (or double).
 */

#define allocmem(M,N,T) \
                  allocmem_int(M,N,sizeof(T),#T,__FUNCTION__,__FILE__,__LINE__)

/* Frees the memory allocated by the allocmem function.
 *
 * Parameters:
 *   p : Pointer to the memory block to be freed.
 *
 * Notice: 
 *  It is okay to pass a NULL pointer to this function; it that case, it
 *  does nothing.
 */
inline static void freemem(void *p) { if (p != NULL) free(p); }

#endif
