/* Copyright (c) 2023-2024 Gilad Odinak */
/* Simple test program for the LSTM layer implementation */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "adamw.h"
#include "lstm.h"

/* Trains a Multi Layer LSTM to predict 
 * the values of f(x) = 0.6 * (sin(x) + 0.4 * sin(1.6 + 1.5 * x))
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16,
 * three layers will be created with size of 32, 16, and 1.
 *
 * range[0] - lowest input (x) value
 * range[1] - highest input (x) value (exclusive)
 * range[2] - increment between x values
 */
static inline float f(float x) { return 0.6 * (sin(x) + 0.4 * sin(1.6 + 1.5 * x)); }
int test_lstm(const float range[3], const int layers[], int layers_cnt,
                          float learning_rate, float weight_decay, int epochs)
{    
    char* title = "f(x) = 0.6 * (sin(x) + 0.4 * sin(1.6 + 1.5 * x))";
    const int L = layers_cnt + 1;
    const int M = (int) ((range[1] - range[0]) / range[2] + 0.5);
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 1;  /* Input vector dimension                     */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values                         */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range[0];
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        yt[i][0] = f(x);
        x += range[2];
    }
    /* Create layers */
    LSTM* l[L];
    for (int j = 0; j < L - 1; j++)
        l[j] = lstm_create(layers[j],1,1);
    l[L - 1] = lstm_create(N,1,1);

    /* Initialize layers */
    lstm_init(l[0],D,M,1);
    for (int j = 1; j < L; j++)
        lstm_init(l[j],layers[j - 1],M,1);
    
    /* Allocate memory for gradients and AdamW state */
    fArr2D dy[L];     /* Gradients with respect to the inputs */
    fArr2D mW[L][8];
    fArr2D vW[L][8];
    fVec mb[L][4];
    fVec vb[L][4];
    for (int j = 0; j < L; j++) {
        dy[j] = allocmem(l[j]->B,l[j]->S,float);
        for (int k = 0; k < 4; k++) {
            mW[j][k] = allocmem(l[j]->D,l[j]->S,float); /* mWf mWi mWc mWo */
            vW[j][k] = allocmem(l[j]->D,l[j]->S,float); /* vWf vWi vWc vWo */
            mb[j][k] = allocmem(1,l[j]->S,float);
            vb[j][k] = allocmem(1,l[j]->S,float);
        }
        for (int k = 4; k < 8; k++) {
            mW[j][k] = allocmem(l[j]->S,l[j]->S,float); /* mUf mUi mUc mUo */
            vW[j][k] = allocmem(l[j]->S,l[j]->S,float); /* vUf vUi vUc vUo */
        }
    }

    float losses[epochs];
    
    const float lr = learning_rate;
    const float wd = weight_decay;
    int update_step = 0; /* adamw update step */ 
    for (int i = 0; i < epochs; i++) {
        fArr2D yp[L]; /* pointers to layers' prediction arrays */
        /* Forward pass */
        yp[0] = lstm_forward(l[0],X,0);
        for (int j = 1; j < L; j++)
            yp[j] = lstm_forward(l[j],yp[j - 1],j);
        fltcpy(y,yp[L - 1],M * N);
        float loss = mean_square_error(y,yt,M,N);
        losses[i] = loss;
        printf("\rEpoch %5d loss %10.3f\r",i+1,loss<999999?loss:999999);
        fflush(stdout);
        /* Backward pass */
        dLdy_mean_square_error(y,yt,dy[L - 1],M,N);
        for (int j = L - 1; j > 0; j--)
            lstm_backward(l[j],dy[j],yp[j - 1],dy[j - 1],j);
        lstm_backward(l[0],dy[0],X,NULL,0);
        /* Update weights */
        update_step++;
        for (int j = 0; j < L; j++) {
            adamw_update(l[j]->Wf,l[j]->gWf,mW[j][0],vW[j][0],
                              l[j]->D,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Wi,l[j]->gWi,mW[j][1],vW[j][1],
                              l[j]->D,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Wc,l[j]->gWc,mW[j][2],vW[j][2],
                              l[j]->D,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Wo,l[j]->gWo,mW[j][3],vW[j][3],
                              l[j]->D,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Uf,l[j]->gUf,mW[j][4],vW[j][4],
                              l[j]->S,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Ui,l[j]->gUi,mW[j][5],vW[j][5],
                              l[j]->S,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Uc,l[j]->gUc,mW[j][6],vW[j][6],
                              l[j]->S,l[j]->S,lr,wd,update_step);
            adamw_update(l[j]->Uo,l[j]->gUo,mW[j][7],vW[j][7],
                              l[j]->S,l[j]->S,lr,wd,update_step);
        }
    }
    printf("\n");
    printf("X:  ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",X[i][0]);
    printf("\nyt: ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",yt[i][0]);
    printf("\ny:  ");
    for (int i = 0; i < M; i++)
        printf("%6.1f ",y[i][0]);
    printf("\n");
    for (int j = 0; j < L; j++) {
        lstm_free(l[j]);
        freemem(dy[j]);
        for (int k = 0; k < 8; k++) {
            freemem(mW[j][k]);
            freemem(vW[j][k]);
        }
        for (int k = 0; k < 4; k++) {
            freemem(mb[j][k]);
            freemem(vb[j][k]);
        }
    }
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float x[M];
        for (int i = 0; i < M; i++)
            x[i] = X[i][0];
        plot_graph(x,(float*)y,(float*)yt,M,
                   epochs,losses,NULL,NULL,NULL,title); 
    }
#else
    (void) losses;
    (void) title;
#endif
    return 0;
}

int main()
{
    init_lrng(42);
    const int layers[3] = {32,16,32};
    const float range[3] = {-10.0,10.0,0.1};
    test_lstm(range,layers,3,0.00001,0.001,20000);
    return 0;
}

