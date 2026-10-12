/* Copyright (c) 2023-2024 Gilad Odinak */
/* Data normalization functions    */
#ifndef NORMALIZE_H
#define NORMALIZE_H
#include "arraytypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Calculates the mean and standard deviation of an array of feature vectors 
 * by feature (i.e., by column) and returns these values in mean and sdev 
 * vectors respectively.
 * 
 * Parameters:
 *   x        - Array of M vectors, each having D elements
 *   M        - Number of vectors in x
 *   D        - Number of elements in each vector 
 *   mean     - Returns a vector of D or D-1 values, each is the mean of 
 *              the values of the corresponding column in x array
 *   sdev     - Returns a vector of D or D-1 values, each is the standard 
 *              deviation from the mean of the corresponding column in x array
 *   exc_last - If not 0, the last column of x (i.e., last element of vectors)
 *              is excluded from the calculation. In that case, mean and sdev
 *              have D-1 elements.
 * 
 * This function calculates element-wise mean and standard deviation for all 
 * feature vectors in the dataset passed in x[][], and returns those mean and 
 * standard deviation arrays. The calculation excludes the last column in x[][]
 * (the bias) if exc_last is not zero.
 */
void calculate_mean_sdev(const fArr2D x_/*[M][D]*/, 
                         int M, int D, 
                         fVec mean/*[D | D-1]*/, 
                         fVec sdev/*[D | D-1]*/, 
                         int exc_last);

/* Normalizes an array of feature vectors by feature (column-wise) in place, by
 * subtracting the corresponding mean and dividing by the corresponding sdev.
 *
 * Parameters:
 *   x        - Array of M vectors, each having D elements
 *   B        - Number of vectors in x
 *   D        - Number of elements in each vector
 *   mean     - A vector of D or D-1 values, each is the mean of the values 
 *              of the corresponding column in x array
 *   sdev     - A vector of D or D-1 values, each is the standard deviation
 *              from the mean of the corresponding column in x array
 *   exc_last - If not 0, the last column of x (i.e., last element of vectors)
 *              is excluded from the calculation. In that case, mean and sdev 
 *              have D-1 elements, and only the first D-1  elements in each
 *              vector in x are normalized.
 */
void normalize(fArr2D x_/*[B][D]*/, 
               int B, int D, 
               const fVec mean_/*[D | D-1]*/, 
               const fVec sdev_/*[D | D-1]*/, 
               int exc_last);


#ifdef __cplusplus
}
#endif

#endif
