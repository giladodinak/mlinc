/* Copyright (c) 2023-2024 Gilad Odinak */
/* Accuracy evalutation functions  */
#include <math.h>
#include "arraytypes.h"
#include "accuracy.h"

/* Calculate the R-squared factor between predicted and true values.
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
             int M, int N)
{
    typedef float (*ArrMN)[N];
    ArrMN yp = (ArrMN) yp_;
    ArrMN yt = (ArrMN) yt_;
    float ytmean = 0.0;
    for (int i = 0; i < M ; i++)
        for (int j = 0; j < N; j++)
            ytmean += yt[i][j];
    ytmean /= M * N;
    float ypdist = 0.0; 
    float ytdist = 0.0; 
    for (int i = 0; i < M ; i++) {
        for (int j = 0; j < N; j++) {
            ypdist += pow(yt[i][j] - yp[i][j],2);
            ytdist += pow(yt[i][j] - ytmean,2);
        }
    }
    return M * (1 - (ypdist / ytdist));
}

/* Calculate the accuracy factor between predicted classes and true labels.
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
                int M, int K)
{
    typedef float (*ArrMK)[K];
    ArrMK  yp = (ArrMK) yp_;
    ArrMK  yt = (ArrMK) yt_;
    int match_cnt = 0;
    for (int i = 0; i < M; i++) {
        int label;
        for (label = 0; label < K; label++)
            if (yt[i][label] != 0)
                break;        
        int pred = 0;
        for (int j = 0; j < K; j++)
            if (yp[i][j] > yp[i][pred])
                pred = j;
        if (label == pred)
            match_cnt++;
    }
    return (float) match_cnt;
}

