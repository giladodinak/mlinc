/* Copyright (c) 2026 Gilad Odinak */
/* GPU Memory functions            */
#include <stdio.h>
#include <cuda_runtime.h>
#include "float.h"
#include "cumem.h"

void* cuallocmem_int(size_t M, size_t N, int S, const char* typ,
                     const char* func, const char* file, int line)
{
    (void) typ;
    void* p = NULL;
    size_t size = M * N * S;
    cudaError_t err = cudaMalloc(&p,size);
    if (err != cudaSuccess) {
        fflush(stdout);
        fprintf(stderr,"\nIn function '%s': "
                "CUDA memory allocation failed at file '%s' line %d: %s\n",
                func,file,line,cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaMemset(p,0,size);
    if (err != cudaSuccess) {
        cudaFree(p);
        fflush(stdout);
        fprintf(stderr,"\nIn function '%s': "
                "CUDA memory initialization failed at file '%s' line %d: %s\n",
                func,file,line,cudaGetErrorString(err));
        exit(-1);
    }
    return p;
}
