/* Copyright (c) 2023-2024 Gilad Odinak */
/* Simple test program for the Dense layer implementation */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "dense.h"

static void dense_update_lin(DENSE* l, float lr);

/* Trains a Multi Layer Perceptronn to predict 
 * the values of f(x) = (x**2 + 10* sin(x))
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16,
 * three layers will be created with size of 32, 16, and 1.
 *
 * range[0] - lowest input (x) value
 * range[1] - highest input (x) value (exclusive)
 * range[2] - increment between x values
 */
int test_dense(const float range[3], const int layers[], int layers_cnt,
                                               float learning_rate, int epochs)
{    
    char* title = "f(x) = (x**2 + 10* sin(x))";
    const int L = layers_cnt + 1;
    const int M = (int) ((range[1] - range[0]) / range[2] + 0.5);
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 1;  /* Input vector dimension                    */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values                        */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range[0];
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        yt[i][0] = (pow(x,2) + 10.0 * sin(x));
        x += range[2];
    }
    /* Create layers */
    DENSE* l[L];
    for (int j = 0; j < L - 1; j++)
        l[j] = dense_create(layers[j],"relu",1);
    l[L - 1] = dense_create(N,"none",1);

    /* Initialize layers */
    dense_init(l[0],D,M,1);
    for (int j = 1; j < L; j++)
        dense_init(l[j],layers[j - 1],M,1);

    /* Allocate memory for input gradients */
    fArr2D dy[L];
    for (int j = 0; j < L; j++)
        dy[j] = allocmem(l[j]->B,l[j]->S,float);

    float losses[epochs];
    
    for (int i = 0; i < epochs; i++) {
        fArr2D yp[L]; /* pointers to layers' prediction arrays */
        /* Forward pass */
        yp[0] = dense_forward(l[0],X,0);
        for (int j = 1; j < L; j++)
            yp[j] = dense_forward(l[j],yp[j - 1],j);
        fltcpy(y,yp[L - 1],M * N); /* Save final forward pass result */
        float loss = mean_square_error(y,yt,M,N);
        losses[i] = loss;
        printf("epoch %5d loss %10.3f\r",i+1,loss<999999?loss:999999);
        fflush(stdout);
        /* Backward pass */
        dLdy_mean_square_error(y,yt,dy[L - 1],M,N);
        for (int j = L - 1; j > 0; j--)
            dense_backward(l[j],dy[j],yp[j - 1],dy[j - 1],0);
        dense_backward(l[0],dy[0],X,NULL,0);
        /* Update weights */
        for (int j = 0; j < L; j++)
            dense_update_lin(l[j],learning_rate);
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
    for (int i = 0; i < L; i++) {
        dense_free(l[i]);
        freemem(dy[i]);
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

/* Updates dense layer's weights and bias in a linear way. */
static void dense_update_lin(DENSE* l, float lr)
{
    typedef float (*ArrDS)[l->S];
    ArrDS Wx = (ArrDS) l->Wx;
    ArrDS gWx = (ArrDS) l->gWx;
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->S; j++)
            Wx[i][j] -= lr * gWx[i][j];

    if (l->use_bias)
        for (int j = 0; j < l->S; j++)
            l->b[j] -= lr * l->gb[j];
}

int main()
{
    init_lrng(42);
    const int layers[3] = {64,128,16};
    const float range[3] = {0.0,5.0,0.1};
    test_dense(range,layers,3,0.0001,200000);

    return 0;
}

