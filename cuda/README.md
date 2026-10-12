# CUDA Source Tree

This directory contains CUDA replacements for selected files in the `src` tree.

## 1. `cuda` tree and `src` tree precedence

The CUDA tree mirrors the normal source tree.

For example:

```text
src/model/dense.c
cuda/model/dense.cu
```

If both files exist, the CUDA version supersedes the CPU version:

```text
src/model/dense.c   -> not compiled
cuda/model/dense.cu -> compiled to dense.o
```

In other words, when both `file.c` in `src` and `file.cu` in `cuda` would
produce the same `file.o`, the CUDA object is used.

Files in the two trees are compiled differently:

```text
src/*.c   -> gcc
cuda/*.cu -> nvcc
```

All resulting object files are linked together with `g++`.

### Header precedence

For files compiled from the CUDA tree, CUDA include directories are searched
before the normal source include directories.

Therefore, if both exist:

```text
src/model/file.h
cuda/model/file.h
```

then a CUDA source file containing:

```c
#include "file.h"
```

uses:

```text
cuda/model/file.h
```

The CUDA header supersedes the `src` header 
**only for files compiled from the CUDA tree**.

If no CUDA version of a header exists, a CUDA source file can still include
the normal header from `src`.

For example:

```text
src/model/array.h
cuda/model/dense.cu
```

with no:

```text
cuda/model/array.h
```

allows:

```c
#include "array.h"
```

in `dense.cu` to use the `src` version.

This permits CUDA replacements to be introduced selectively: only files
that require a CUDA-specific implementation need to exist in the CUDA tree.

## 2. CUDA function qualifiers

| Qualifier | Executes on | Called from | Typical use |
|---|---|---|---|
| none / `__host__` | CPU | CPU | Normal C/C++ function |
| `__device__` | GPU | GPU | Helper function used by GPU code |
| `__host__ __device__` | CPU or GPU | CPU or GPU | Small helper that needs both CPU and GPU versions |
| `__global__` | GPU | CPU using `<<< >>>` | CUDA kernel entry point |

For example:

```c
__host__ __device__ float sigmoid1(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
```

can be called from either CPU code or GPU code.

A CUDA kernel is declared with `__global__`:

```c
__global__ void sigmoid_kernel(float *a, int n)
{
    ...
}
```

and is launched from CPU code using CUDA's `<<< >>>` syntax.

## 3. `blockIdx.x`, `blockDim.x`, and `threadIdx.x`

CUDA launches many threads at once. Threads are grouped into blocks, and blocks form a grid.

Three built-in values are commonly used to determine which element a thread should process:

```c
blockIdx.x
blockDim.x
threadIdx.x
```

### `threadIdx.x`

`threadIdx.x` is the thread's index **within its block**.

If a block contains 256 threads:

```text
threadIdx.x = 0 ... 255
```

### `blockIdx.x`

`blockIdx.x` is the block's index **within the grid**.

If 100 blocks are launched:

```text
blockIdx.x = 0 ... 99
```

### `blockDim.x`

`blockDim.x` is the number of threads in each block.

For:

```c
kernel<<<100,256>>>(...);
```

every thread sees:

```text
blockDim.x == 256
```

### Global thread index

These are commonly combined as:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This gives each thread a unique linear index across the entire grid.

For example:

```c
__global__ void relu_kernel(float *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && a[i] < 0.0f)
        a[i] = 0.0f;
}
```

The first block handles indices starting at 0, the second block starts at `blockDim.x`, and so on.

## 4. `extern __shared__`

CUDA shared memory is fast memory shared by all threads in one block.

Dynamic shared memory is declared inside a kernel as:

```c
extern __shared__ float tmp[];
```

The array size is not specified in the declaration. Its size is supplied
when the kernel is launched.

For example:

```c
__global__ void softmax_kernel(float *a, int K)
{
    extern __shared__ float tmp[];

    ...
}
```

can be launched with:

```c
softmax_kernel<<<BLKCNT,THDCNT,THDCNT * sizeof(float)>>>(a,K);
```

The third launch argument:

```c
THDCNT * sizeof(float)
```

allocates that many bytes of shared memory **for each block**.

Therefore, if:

```text
THDCNT = 256
sizeof(float) = 4
```

each block receives:

```text
256 * 4 = 1024 bytes
```

and inside the kernel:

```c
tmp[0] ... tmp[255]
```

are available to all threads in that block.

Each block receives its own independent shared-memory array.

Threads in the same block commonly coordinate access to shared memory with:

```c
__syncthreads();
```

## 5. CUDA kernel invocation: `<<< >>>`

A `__global__` function is launched using:

```c
kernel<<<grid_size,block_size>>>(arguments);
```

For example:

```c
sigmoid_kernel<<<BLKCNT,THCNT>>>(a,n);
```

means:

```text
BLKCNT blocks per grid
THCNT threads per block
```

The full launch form is:

```c
kernel<<<grid_size,block_size,shared_memory_bytes,stream>>>(arguments);
```

The four fields are:

```text
1. grid_size            number of blocks
2. block_size           number of threads per block
3. shared_memory_bytes  dynamic shared memory allocated per block
4. stream               CUDA stream used for the launch
```

The third and fourth arguments are optional.

For example:

```c
softmax_kernel<<<BLKCNT,THDCNT,MEMSIZE)>>>(a,K);
```

launches:

```text
BKCNT   - number of blocks
THDCNT  - threads per block
MEMSIZE - number of bytes of shared memory per block
```

with the default CUDA stream.

A common one-dimensional launch calculation is:

```c

int blkcnt = (n + thdcnt - 1) / thdcnt;
kernel<<<blkcnt,thdcnt>>>(a,n);
```

Because the final block may contain more threads than required, kernels
normally guard memory access:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n) {
    ...
}
```
