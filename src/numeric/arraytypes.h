/* Copyright (c) 2023-2024 Gilad Odinak */
/* Array type definitions */
#ifndef ARRAYTYPES_H
#define ARRAYTYPES_H

/* Define types of one and two dimensional arrays of unspecified dimensions.
 * These are dynamically cast to explicit dimensions within each function.
 *
 * For example
 *     fArr2D a2d; // a2d is two-dimensional array of unspecified dimensions
 *     typedef float (*ArrNK)[K]; // two-dimensional array NxK (e.g. N=3 K=5)
 *     ArrNK a2dNK = (ArrNK) a2d; // a2dNK is the addrress of 2-D array 3x5
 *     a2dNK[2][4] access an entry in the array
 * Notice that a2dNK memory is contiguous just like when declaring
 * a two-dimensional array  float arr[3][5]
 *
 * When passing an array by reference to a function, only the last dimension
 * can be specified. By convention, the first dimension is noted in the typedef
 * and in comments.
 *
 */
typedef int (*iVec);
typedef int (*iArr2D)[];
typedef float (*fVec);
typedef float (*fArr2D)[];
#endif
