/* Copyright (c) 2026 Gilad Odinak */
/* GPU execution context           */
#ifndef CUCTX_H
#define CUCTX_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Creates a cuBLAS handle when a varibale of this class is instansiated.
 * The handle is released when the variable is destroyed (goes out of scope).
 * Initializes cuda stream handle, which is passed to kernel launch routines.
 */
class CudaContext {
public:
    cublasHandle_t handle;
    cudaStream_t stream;

    CudaContext() : handle(NULL), stream(0)
    {
        /* cublasCreate sets cuBLAS stream to the default null (0) stream */
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,"cublasCreate failed (%d)\n",(int) status);
            exit(1);
        }
    }

    ~CudaContext()
    {
        if (handle != NULL)
            cublasDestroy(handle);
    }
};

/* This function initializes one cuBLAS handle, and one CUDA stream per
 * CPU thread that are returned by all invocations of this function from
 * that same thread.
 *
 * Notice that this file is included from .cu files and has C++ semantics:
 * Althogh this function is included by many compilation units there is
 * only one copy of the handle variable in the calling thread's local storage.
 *
 * This design supports only one (the default) GPU device.
 */
inline CudaContext& cuda_context(void)
{
    static thread_local CudaContext ctx;
    return ctx;
}

/* Returns the current cuBLAS handle */
inline cublasHandle_t cublas_handle(void)
{
    return cuda_context().handle;
}

/* Returns the current CUDA stream */
inline cudaStream_t cuda_stream(void)
{
    return cuda_context().stream;
}

/* Sets a new CUDA stream and returns the previous one */
inline cudaStream_t cuda_set_stream(cudaStream_t stream)
{
    CudaContext& ctx = cuda_context();
    cudaStream_t old_stream = ctx.stream;
    ctx.stream = stream;
    cublasStatus_t status = cublasSetStream(ctx.handle,ctx.stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"cublasSetStream failed (%d)\n",(int)status);
        exit(1);
    }
    return old_stream;
}

#endif
