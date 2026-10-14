/* Copyright (c) 2023-2024 Gilad Odinak */
/* Memory data structures and functions  */
#include <stdio.h>
#include <stdlib.h>  /* calloc() free()             */
#include <memory.h>  /* memcpy() memmove() memset() */
#include <strings.h> /* strcmp()                    */
#include "float.h"
#include "mem.h"

/* Checks whether float representaion of 0.0 is all zero bytes */
static const char bZero[sizeof(float)] = {0}; /* All bytes are zero        */
static const float fZero = 0.0; /* Compiler's representation of float zero */

void* allocmem_int(size_t M, size_t N, int S, const char* typ, 
                   const char* func, const char* file, int line)
{
    void* p = calloc(M * N,S);
    if (p == NULL) {
        fflush(stdout);
        fprintf(stderr,"\nIn function '%s': "
                "out of memory at file '%s' line %d\n",func,file,line);
        exit(-1);
    }
    /* calloc returns pointer to memory that is initialized to binary zeros */
    if (memcmp(&fZero,&bZero,sizeof(fZero))) {
        /* float/double not represented as binary zeros - initialize here   */
        if(!strcmp("float",typ))
            fltclr(p,M * N);
        else 
        if(!strcmp("double",typ))
            for (size_t i = 0; i < M * N; i++)
                ((double *) p)[i] = 0.0;
    } 
    return p;
}
