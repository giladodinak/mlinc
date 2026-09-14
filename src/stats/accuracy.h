/* Copyright (c) 2023-2024 Gilad Odinak */
/* Accuracy evalutation functions  */
#ifndef ACCURACY_H
#define ACCURACY_H
#include "arraytypes.h"

/* Calculates the R-squared factor between predicted and true values.
 * 
 * This function calculates the numerator of the R-squared factor between M 
 * predicted values and M true values. The result is a value between 0 and M, 
 * where 0 indicates no fit, and M indicates a perfect fit.
 * 
 * Parameters:
 *   yp - Predicted values array of size MxN
 *   yt - True values array of size MxN
 *   M  - Number of samples (rows)
 *   N  - Number of features (columns)
 * 
 * Returns:
 *   R-squared factor numerator value
 * 
 * Reference:
 *   https://en.wikipedia.org/wiki/Coefficient_of_determination
 */
float R2_sum(const fArr2D yp_/*[M][N]*/, 
             const fArr2D yt_/*[M][N]*/, 
             int M, int N);

/* Calculates the accuracy factor between predicted classes and true labels.
 * 
 * This function calculates the numerator of the accuracy factor between M 
 * predicted classes and M true labels. The result is a value between 0 and M, 
 * where 0 indicates no match and M indicates a perfect match.
 * 
 * Parameters:
 *   yp - Predicted classes array of size MxK
 *   yt - One-hot encdoed true labels array of size MxK
 *   M  - Number of samples (rows)
 *   K  - Number of classes (columns)
 * 
 * Note:
 *   This function assume the one-hot encoded vectors in yt have only one
 *   element set to 1 and all other elements set to 0. It does not check
 *   all values of yt to ensure that is the case.
 *
 * Returns:
 *   Accuracy factor numerator value
 */
float match_sum(const fArr2D yp_/*[M][K]*/,
                const fArr2D yt_/*[M][K]*/, 
                int M, int K);

#endif
