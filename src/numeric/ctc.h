/* Copyright (c) 2023-2024 Gilad Odinak */
/* Connectionist Temporal Classification functions */
#ifndef CTC_H
#define CTC_H
#include "array.h"

/* Note yp, alpha, beta, prob values are in log scale  */
typedef struct ctc_s {
    int T;        /* Number of time steps (batch size) */
    int L;        /* Number of lables (classes)        */
    int blank;    /* Blank class index                 */
    fArr2D yp;    /* output probabilities [T][L]       */
    iVec ypc;     /* predicted lables from yp [T]      */
    int ypclen;   /* Actual number of labels in ypc    */
    iVec ytc;     /* true lables from yt [T]           */
    int ytclen;   /* Actual number of labels in ytc    */
    iVec label;   /* padded true labels [2*T+1]        */
    int S;        /* Actual length of padded label     */
    fArr2D alpha; /* forward probabilities [T][2*T+1]  */
    fArr2D beta;  /* backward probabilities [T][2*T+1] */
    fVec prob;    /* Final probabilities [T]           */
} CTC;

#ifdef __cplusplus
extern "C" {
#endif
    
/* Creates a Contectionist Temporal Classification loss calculator.
 * 
 * Parameters:
 *  T     - Number of time steps (number of input vectors)
 *  L     - Number of distinct labels (input vectors dimension), including blank
 *  blank - The index of the blank label (0 .. L-1)
 */
CTC* ctc_create(int T, int L, int blank);

/* Frees memory allocated by ctc_create()
 */
void ctc_free(CTC* ctc);

/* Calculates ctc loss for a batch of probability vectors 
 * and correspoding class labels.
 *
 * Parameters:
 *   yp - array of T vectors, each having L class probabilities
 *   yt - array of up to T one-hot encoded labels. If there are
 *        less than T labels, pad with blank labels to the end.
 * 
 * Returns:
 *   ctc loss value
 *
 * Note: 
 *   This function merges repeated identical labels, so instead of 
 *   padding with blank labels at the end, labels can be duplicated.
 *   for example, assuming blank label index is 0, and there are six
 *   time steps, the (one hot encoded) labels 5 1 2 can be encoded 
 *   as either 5 1 2 0 0 0 or 5 5 1 2 2 0 or 5 5 1 1 2 2.  If the 
 *   alignment is known, the late approch yields better results
 *   when batch size is smaller than sequence length.
 * 
 */
float ctc_loss(CTC* ctc,
               const fArr2D yp_/*[T][L]*/, 
               const fArr2D yt_/*[T][L]*/,
               int T, int L);

/* Calculates the gradient of the ctc loss with respect to prediction (dL/dy)
 * for predicted vectors and their true labels.
 *
 * Parameters:
 *   yp - array of T vectors, each having L class probabilities
 *   yt - array of up to T one-hot encoded labels. If there are
 *        less than T labels, blank labels pad the time steps
 *        between them. Blank label index value is L.
 *
 * Returns:
 *   dy - array of T vectors, each having L gradients (output)
 */
void dLdy_ctc_loss(CTC* ctc,
                   const fArr2D yp_/*[T][L]*/,
                   const fArr2D yt_/*[T][L]*/,
                   fArr2D dy_/*[T][L]*/,
                   int T, int L);

/* Calculates the accuracy factor between predicted classes and true labels.
 * 
 * This function calculates the numerator of the accuracy factor between T 
 * predicted classes and T true labels. The result is a value between 0 and T, 
 * where 0 indicates no match and T indicates a perfect match.
 * This numer is propotional to the edit distance of the two label sequences
 * after consequitve identical labels are merged.
 * 
 * Parameters:
 *   yp - Predicted classes array of size TxL
 *   yt - One-hot encdoed true labels array of size TxL
 *   T  - Number of samples (rows)
 *   L  - Number of classes (columns)
 * 
 * Note:
 *   This function uses the predicted and true labels calculated by ctc_loss,
 *   and ignores the passed in yp, yt arrays.
 *
 * Returns:
 *   Accuracy factor numerator value
 */
float ctc_accuracy(CTC* ctc,
                   const fArr2D yp_/*[T][L]*/,
                   const fArr2D yt_/*[T][L]*/, 
                   int T, int L);


#ifdef __cplusplus
}
#endif

#endif
