/* Copyright (c) 2023-2024 Gilad Odinak */
/* LSTM (reccurrent) neural network layer data structures and functions */
#ifndef LSTM_H
#define LSTM_H

#include "arraytypes.h"

typedef struct lstm_s {
  int D;           /* Input vector dimension                        */
  int S;           /* Number of units, size of hidden state         */
  int B;           /* Number of input vectors in a batch            */
  /* Note that B also is the sequence length (number of time steps) */
  int stateful;    /* 1: maintain state between batches             */
  /* Stateful mode preserves the final hidden and cell state across
   * forward passes only. Gradients do NOT propagate across sequences
   * (i.e., truncated BPTT with truncation at sequence boundaries).
   */
  char use_bias;   /* 1 - add bias, 0 - do not add bias             */
  char training;   /* 1 - training mode, 0 - inference only         */
  fArr2D Wf;       /* GPU: Weights matrix [D][S]                    */
  fArr2D Wi;       /* GPU: Weights matrix [D][S]                    */
  fArr2D Wc;       /* GPU: Weights matrix [D][S]                    */
  fArr2D Wo;       /* GPU: Weights matrix [D][S]                    */
  fArr2D Uf;       /* GPU: Recurrent weights [S][S]                 */
  fArr2D Ui;       /* GPU: Recurrent weights [S][S]                 */
  fArr2D Uc;       /* GPU: Recurrent weights [S][S]                 */
  fArr2D Uo;       /* GPU: Recurrent weights [S][S]                 */
  fVec bf;         /* GPU: Bias vector [S]                          */
  fVec bi;         /* GPU: Bias vector [S]                          */
  fVec bc;         /* GPU: Bias vector [S]                          */
  fVec bo;         /* GPU: Bias vector [S]                          */
  fArr2D f;        /* GPU: Forget gate matrix [B][S]                */
  fArr2D i;        /* GPU: Input gate matrix [B][S]                 */
  fArr2D o;        /* GPU: Output gate matrix [B][S]                */
  fArr2D cc;       /* GPU: Cell candidate matrix [B+1][S]           */
  fArr2D c;        /* GPU: Cell matrix [B+1][S]                     */
  fArr2D h;        /* GPU: Hidden state matrix [B+1][S]             */
  fVec ph;         /* GPU: Previous batch hidden state [S]          */
  fVec pc;         /* GPU: Previous batch cell state [S]            */
  /* Gradients (only if training == 1) */
  fArr2D gWf;      /* GPU: Weight gradients [D][S]                  */
  fArr2D gWi;      /* GPU: Weight gradients [D][S]                  */
  fArr2D gWc;      /* GPU: Weight gradients [D][S]                  */
  fArr2D gWo;      /* GPU: Weight gradients [D][S]                  */
  fArr2D gUf;      /* GPU: Recurrent gradients [S][S]               */
  fArr2D gUi;      /* GPU: Recurrent gradients [S][S]               */
  fArr2D gUc;      /* GPU: Recurrent gradients [S][S]               */
  fArr2D gUo;      /* GPU: Recurrent gradients [S][S]               */
  fVec gbf;        /* GPU: Bias gradients [S]                       */
  fVec gbi;        /* GPU: Bias gradients [S]                       */
  fVec gbc;        /* GPU: Bias gradients [S]                       */
  fVec gbo;        /* GPU: Bias gradients [S]                       */
} LSTM;

#ifdef __cplusplus
extern "C" {
#endif

LSTM* lstm_create(int units, int stateful, int use_bias);
void lstm_init(LSTM* l, int input_dim, int batch_size, int training);
void lstm_set_batch_size(LSTM* l, int batch_size);
void lstm_free(LSTM* l);
void lstm_reset(LSTM* l);

/* X must point to CUDA device memory. Returns CUDA device memory. */
fArr2D lstm_forward(LSTM* restrict l,
                    const fArr2D restrict X/*[B][D]*/, int lyr);

/* dY, X, and dX (when non-NULL) must point to CUDA device memory. */
void lstm_backward(LSTM* restrict l,
                   const fArr2D restrict dY/*[B][S]*/,
                   const fArr2D restrict X/*[B][D]*/,
                   fArr2D restrict dX/*[B][D]*/,
                   int lyr);

#ifdef __cplusplus
}
#endif

#endif
