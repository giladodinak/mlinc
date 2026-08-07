/* Copyright (c) 2023-2024 Gilad Odinak */
/* Simple test program for the Multi Layer Neural Network implementation */
#include <stdio.h>
#include <math.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "transformer.h"
#include "model.h"
#include "irisfile.h"
#include "modelio.h"

/* Trains a Multi Layer Perceptron to predict
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
int test_dense_regression(const float range[3], 
                          const int layers[], int layers_cnt, char* optimizer,
                          float learning_rate, float weight_decay, int epochs)
{   
    #undef f 
    #define f(x) (pow(x,2) + 10.0 * sin(x))
    char* title = "f(x) = x**2 + 10* sin(x)";
    printf("\n\nTrains a Multi Layer Perceptron to predict "
           "the values of the function \n    %s\n\n",title);

    const int L = layers_cnt + 1;
    const int M = (int) ((range[1] - range[0]) / range[2] + 0.5);
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 2;  /* Input vector dimension (including bias)    */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values, X[][1] is bias == 1.0  */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range[0];
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        X[i][1] = 1.0;
        yt[i][0] = f(x);
        x += range[2];
    }
    /* Create Model (single batch of all samples) */
    MODEL* m = model_create(L,M,D,0,1); /* don't add bias, normalize */
    model_add(m,dense_create(layers[0],"relu"),"dense"); 
    for (int i = 1; i < L - 1; i++)
        model_add(m,dense_create(layers[i],"relu"),"dense");
    model_add(m,dense_create(N,"none"),"dense"); 

    model_compile(m,"mean-square-error",optimizer);

    /* Train model */
    float losses[epochs];
    float accuracies[epochs];

    model_fit(m,X,yt,NULL,M,
              NULL,NULL,NULL,0,
              epochs,learning_rate,weight_decay,
              losses,accuracies,NULL,NULL,
              "shuffle=0 final=1 verbose=1");

    model_predict(m,X,y,M);
    
    printf("\n");

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float x[M];
        for (int i = 0; i < M; i++)
            x[i] = X[i][0];
        plot_graph(x,(float*)y,(float*)yt,M,
                   epochs,losses,accuracies,NULL,NULL,title);
    }
#else
    (void) accuracies;
    (void) losses;
    (void) title;
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
#endif
    model_free(m);
    return 0;
}

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
int test_lstm_regression(const float range[3], 
                         const int layers[], int layers_cnt, char* optimizer,
                         float learning_rate, float weight_decay, int epochs)
{    
    #undef f 
    #define f(x) (0.6 * (sin(x) + 0.4 * sin(1.6 + 1.5 * x)))
    char* title = "f(x) = 0.6 * (sin(x) + 0.4 * sin(1.6 + 1.5 * x))";
    printf("\n\nTrains a Multi Layer LSTM to predict "
           "the values of the function \n    %s\n\n",title);
    
    const int L = layers_cnt + 1;
    const int M = (int) ((range[1] - range[0]) / range[2] + 0.5);
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 2;   /* Input vector dimension (including bias)    */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values, X[][1] is bias == 1.0  */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range[0];
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        X[i][1] = 1.0;
        yt[i][0] = f(x);
        x += range[2];
    }
    /* Create Model (single batch of all samples) */
    MODEL* m = model_create(L,M,D,0,0); /* don't add bias, don't normalize */
    model_add(m,lstm_create(layers[0],1),"lstm");
    for (int i = 1; i < L - 1; i++)
        model_add(m,lstm_create(layers[i],1),"lstm");
    model_add(m,lstm_create(N,1),"lstm"); 

    model_compile(m,"mean-square-error",optimizer);

    /* Train model */
    float losses[epochs];
    float accuracies[epochs];

    model_fit(m,X,yt,NULL,M,
              NULL,NULL,NULL,0,
              epochs,learning_rate,weight_decay,
              losses,accuracies,NULL,NULL,
              "shuffle=0 final=1 verbose=1");

    model_predict(m,X,y,M);

    printf("\n");

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float x[M];
        for (int i = 0; i < M; i++)
            x[i] = X[i][0];
        plot_graph(x,(float*)y,(float*)yt,M,
                   epochs,losses,accuracies,NULL,NULL,title);
    }
#else
    (void) accuracies;
    (void) losses;
    (void) title;
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
#endif
    model_free(m);
    return 0;
}

/* Trains a multi layer LSTM followed by Dense layer to predict 
 * the values of f(x) = sin(x) + 0.4 * sin(1.6 + 1.5 * x)
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied. So for example if layers contains two elements, 32 and 16,
 * two LSTM layers will be created with size of 32 and 16, followed by 
 * a Dense layer of size 1.
 *
 * range[0] - lowest input (x) value
 * range[1] - highest input (x) value (exclusive)
 * range[2] - increment between x values
 */
int test_lstm_dense_regression(const float range[3], 
                           const int layers[], int layers_cnt, char* optimizer,
                           float learning_rate, float weight_decay, int epochs)
{    
    #undef f 
    #define f(x) (sin(x) + 0.4 * sin(1.6 + 1.5 * x))
    char* title = "f(x) = sin(x) + 0.4 * sin(1.6 + 1.5 * x)";
    printf("\n\nTrains LSTM + final dense layer to predict "
           "the values of the function \n    %s\n\n",title);
    const float range_start = range[0]; 
    const float range_end = range[1];
    const float range_step = range[2]; 
    const int L = layers_cnt + 1;
    const int M = (int) ((range_end - range_start) / range_step + 0.5);
    const int B = M;
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = 2;  /* Input vector dimension (including bias)   */
    const int N = 1;  /* Output vector dimension                    */
    float X[M][D];    /* X[][0] is x values, X[][1] is bias == 1.0  */
    float yt[M][N];   /* True labels vector yt = f(X)               */
    float y[M][N];    /* Output prediction (single dimension)       */
    float x = range_start;
    /* Initialize data */
    for (int i = 0; i < M ; i++) {
        X[i][0] = x;
        X[i][1] = 1.0;
        yt[i][0] = f(x);
        x += range_step;
    }
    /* Create Model */
    MODEL* m = model_create(L,B,D,0,1); /* don't add bias, normalize */
    model_add(m,lstm_create(layers[0],1),"lstm"); 
    for (int i = 1; i < L - 1; i++)
        model_add(m,lstm_create(layers[i],1),"lstm");
    model_add(m,dense_create(N,"none"),"dense"); 

    model_compile(m,"mean-square-error",optimizer);

    /* Train model */
    float losses[epochs];
    float accuracies[epochs];

    model_fit(m,X,yt,NULL,M,
              NULL,NULL,NULL,0,
              epochs,learning_rate,weight_decay,
              losses,accuracies,NULL,NULL,
              "shuffle=0 final=1 verbose=1");

    model_predict(m,X,y,M);

    printf("\n");

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float x[M];
        for (int i = 0; i < M; i++)
            x[i] = X[i][0];
        plot_graph(x,(float*)y,(float*)yt,M,
                   epochs,losses,accuracies,NULL,NULL,title);
    }
#else
    (void) accuracies;
    (void) losses;
    (void) title;
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
#endif
    model_free(m);
    return 0;
}

/* Trains a Multi Layer Perceptron to predict the class of an 
 * Iris plant out of three classes, based on four features.  
 * Uses the R.A. Fisher Iris Plants Database 
 *
 * https://archive.ics.uci.edu/dataset/53/iris
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16,
 * three layers will be created with size of 32, 16, and 3.
 */
int test_dense_classification(
                        const int layers[], int layers_cnt, 
                        char* optimizer, int batch_size,
                        float learning_rate, float weight_decay, int epochs)
{    
    const char* irisfile = "data/iris/iris.csv";
    char* title = "Confusion Matrix";
    printf("\n\nTrains a Multi Layer Perceptron to predict the \n");
    printf("classes of samples from the Iris dataset\n\n");
    const int L = layers_cnt + 1;   /* Number of layers  */
    const int M = IRIS_SAMPLE_CNT;  /* Number of samples */
    const int B = batch_size;       /* Batch size        */
    printf("%d layers (including output layer), %d input samples\n",L,M);
    const int D = IRIS_FEAT_CNT;  /* Input vector dimension (excluding bias) */
    const int N = IRIS_CLASS_CNT; /* Output vector dimension                 */
    float X[M][D];    /* Iris Dataset                               */
    int   yc[M];      /* True labels (values 0,1,2)                 */
    float yt[M][N];   /* True labels vectors (values 001, 010, 100) */

    /* Read data */
    int ok = read_iris_file(irisfile,M,X,yc);
    if (!ok)
        return -1;
    /* Shuffle samples to have even mix for train, validate, test sets*/
    for (int i = M - 1; i > 0; i--) {
        int j = (int) urand(0.0,1.0 + i);
        /* Swap i sequence with j sequence */
        float x_t[D]; fltcpy(x_t,X[i],D);
        int yc_t = yc[i];
        yc[i] = yc[j];
        fltcpy(X[i],X[j],D);
        yc[j] = yc_t;
        fltcpy(X[j],x_t,D);
    }
    /* Encode yc as one-hot vectors */    
    fltclr(yt,M * N);
    for (int i = 0; i < M; i++)
        yt[i][yc[i]] = 1.0;
        
    /* Create Model (multiple batches of B samples each) */
    MODEL* m = model_create(L,B,D,1,0); /* add bias, don't normalize */
    model_add(m,dense_create(layers[0],"relu"),"dense"); 
    for (int i = 1; i < L - 1; i++)
        model_add(m,dense_create(layers[i],"relu"),"dense");
    model_add(m,dense_create(N,"softmax"),"dense"); 

    model_compile(m,"cross-entropy",optimizer);

    /* Traininig set */
    int trCnt = 8 * M / 10;
    fArr2D xTr = X;
    fArr2D yTr = yt;

    /* Validation set */
    int vdCnt = M / 10;
    fArr2D xVd = X + trCnt;
    fArr2D yVd = yt + trCnt;

    /* Test set */
    int teCnt = M - vdCnt - trCnt;
    fArr2D xTe = X + trCnt + vdCnt;
    
    float losses[epochs];
    float accuracies[epochs];
    float v_losses[epochs];
    float v_accuracies[epochs];

    model_fit(m,xTr,yTr,NULL,trCnt,
                xVd,yVd,NULL,vdCnt,
                epochs,learning_rate,weight_decay,
                losses,accuracies,v_losses,v_accuracies,
                "final=1 verbose=1");
    printf("\n");

    /* Test */
    float yp[teCnt][N]; /* Output predictions */
    model_predict(m,xTe,yp,teCnt);

    int cm[N][N];
    memset(cm,0,N * N * sizeof(int));

    int* ytc = yc + trCnt + vdCnt; /* Test true class indcies */
    int  ypc[teCnt]; /* Test predicted class indecies */
    /* Convert output predictions to predicted class indcies */
    int cnt = 0;
    for (int i = 0; i < teCnt; i++) {
        int pi = 0;
        float pv = yp[i][pi];
        for (int j = 1; j < N; j++) {
            if (yp[i][j] > pv) {
                pv = yp[i][j];
                pi = j;
            }
        }
        ypc[i] = pi;
        cm[ytc[i]][ypc[i]]++;
        if (ytc[i] == ypc[i])
            cnt++;
    }
    printf("Test accuracy %5.3f\n",((float) cnt) / teCnt);
    
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        plot_cm(cm,3,iris_class_names,
                epochs,losses,accuracies,v_losses,v_accuracies,title,"numbers"); 
    }
#else
    (void) accuracies;
    (void) losses;
    (void) v_accuracies;
    (void) v_losses;
    (void) title;
    printf("\n");
#endif
    model_free(m);
    return 0;
}

#define JVOWELS_TR_SEQUENCE_CNT 270
#define JVOWELS_TR_SAMPLE_CNT  4274
#define JVOWELS_TE_SEQUENCE_CNT 370
#define JVOWELS_TE_SAMPLE_CNT  5687
#define JVOWELS_FEAT_CNT         12
#define JVOWELS_CLASS_CNT         9
#define JVOWELS_SUBJECT_CNT       9

const char* jvowels_class_names[JVOWELS_CLASS_CNT] = {
    "P1","P2","P3","P4","P5","P6","P7","P8","P9"
};

int read_jvowels_file(const char* input_path, const char* type, 
                      int n_sequences, int *seq_len,
                      int n_samples, float x[][JVOWELS_FEAT_CNT], int yc[])
{
    char fn_s[256];
    snprintf(fn_s,sizeof(fn_s),"%s/size_ae.%s",input_path,type);
    FILE *fp_s = fopen(fn_s,"rb");
    if (fp_s == NULL) {
        fprintf(stderr,"%s: failed to open file for read\n",fn_s);
        return 0;
    }
    char fn_x[256];
    snprintf(fn_x,sizeof(fn_x),"%s/ae.%s",input_path,type);
    FILE *fp_x = fopen(fn_x,"rb");
    if (fp_x == NULL) {
        fprintf(stderr,"%s: failed to open file for read\n",fn_x);
        return 0;
    }

    int Pn = 0;  /* Person # 1 - 9 */
    int Psc = 0; /* Number of sequences for person Pn */
    int i = 0;   /* Total sample count                */
    int j = 0;   /* Count sequences per person        */
    int k = 0;   /* Counts samples within a sequence  */
    int l = 0;   /* Total sequence count              */
    for (;;) {
        char buffer[256];
        char *line;
        int cnt;
        if (Psc == 0) { /* Fetch number of sequences for next person */
            Pn++; /* Person # */
            if (Pn > JVOWELS_SUBJECT_CNT) /* Note Pn is 1-based */
                break;
            j = 0;
            int cnt = fscanf(fp_s,"%d",&Psc);
            if (cnt < 1 || Psc == 0) {
                fprintf(stderr,"%s: at line %d: "
                        "failed to parse value from file\n",fn_s,Pn);
                fclose(fp_x);
                fclose(fp_s);
                return 0;
            }
        }
        line = fgets(buffer,sizeof(buffer),fp_x);
        if (line == NULL) {
            fprintf(stderr,"%s: at line %d: "
                           "failed to read from file\n",fn_x,i + 1);
            fclose(fp_x);
            fclose(fp_s);
            return 0;
        }
        if (strspn(line, " \t\n\v\f\r") == strlen(line)) { /* Blank line */
            if (l >= n_sequences)
                break;
            seq_len[l] = k;
            k = 0;
            l++; /* Total sequences              */
            j++; /* Sequences for current person */
            if (j >= Psc) /* End of sequences for current person */
                Psc = 0;
            continue;
        }
        if (i >= n_samples)
            break;
        cnt = sscanf(line,FMTF " " FMTF " " FMTF " " FMTF " " 
                          FMTF " " FMTF " " FMTF " " FMTF " " 
                          FMTF " " FMTF " " FMTF " " FMTF,
                          &x[i][0],&x[i][1],&x[i][2],&x[i][3],
                          &x[i][4],&x[i][5],&x[i][6],&x[i][7],
                          &x[i][8],&x[i][9],&x[i][10],&x[i][11]);
        if (cnt < JVOWELS_FEAT_CNT) {
            fprintf(stderr,"%s: at line %d: "
                    "failed to parse 12 values from file\n",fn_x,i + 1);
            fclose(fp_x);
            fclose(fp_s);
            return 0;
        }
        yc[i] = Pn;
        i++;
        k++;
    }
    fclose(fp_x);
    fclose(fp_s);
    return l;
}

/* Trains an LSTM layer followed by Dense layer to recognize
 * speakers of japanese vowels.
 *
 * https://archive.ics.uci.edu/dataset/128/japanese+vowels
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16, 
 * two LSTM layers will be created with size of 32 and 16, followed by 
 * a Dense layer of size 4.
 */
int test_lstm_dense_classification(
                        const int layers[], int layers_cnt, 
                        char* optimizer, int batch_size,
                        float learning_rate, float weight_decay, int epochs)
{
    const char* jvowels_path = "data/jvowels";
    char* title = "Confusion Matrix";
    printf("\n\nTrains LSTM + final dense layer to predict the \n");
    printf("speaker of samples from the Japanese Vowels dataset\n\n");
    const int L = layers_cnt + 1;       /* Number of layers                  */
    const int B = batch_size;           /* Batch size                        */
    const int D = JVOWELS_FEAT_CNT;     /* Raw data dimension                */
    const int N = JVOWELS_CLASS_CNT;    /* Output vector dimension           */

    printf("%d layers (including output layer) ",L);
    for (int i = 0; i < layers_cnt; i++)
        printf("%d,",layers[i]);
    printf("%d. Batch size %d. ",N,B);
    printf("Input dimension %d.\n\n",D);
    printf("%d epochs, learning rate %f, weight decay %f \n",
                                           epochs,learning_rate,weight_decay);

    typedef float (*ArrMD)[D];
    typedef float (*ArrMN)[N];
    typedef int (*VecM);
    typedef int (*VecS);

    const int STr = JVOWELS_TR_SEQUENCE_CNT;
    const int MTr = JVOWELS_TR_SAMPLE_CNT;
    ArrMD xTr = allocmem(MTr,D,float);  /* Vowels training Dataset           */
    VecS  sTr = allocmem(STr,1,int);    /* Vowels training sequence lengths  */
    VecM  yTrc = allocmem(MTr,1,int);   /* True labels (values 0,1,2,3,4,5)  */
    ArrMN yTrt = allocmem(MTr,N,float); /* True labels vectors (000001 ... 100000) */
    const int STe = JVOWELS_TE_SEQUENCE_CNT;
    const int MTe = JVOWELS_TE_SAMPLE_CNT;
    ArrMD xTe = allocmem(MTe,D,float);  /* Vowels test Dataset               */
    VecS  sTe = allocmem(STe,1,int);    /* Vowels test sequence lengths      */
    VecM  yTec = allocmem(MTe,1,int);   /* True labels (values 0,1,2,3,4,5)  */
    ArrMN yTet = allocmem(MTe,N,float); /* True labels vectors (000001 ... 100000) */

    int cnt;

    /* Read data */
    cnt = read_jvowels_file(jvowels_path,"train",STr,sTr,MTr,xTr,yTrc);
    if (cnt != STr)
        return -1;
    cnt = read_jvowels_file(jvowels_path,"test",STe,sTe,MTe,xTe,yTec);
    if (cnt != STe)
        return -1;
        
    /* Encode yc as one-hot vectors */
    fltclr(yTrt,MTr * N);
    for (int i = 0; i < MTr; i++)
        yTrt[i][yTrc[i] - 1] = 1.0; /* Note 1-based class numbers */
    fltclr(yTet,MTe * N);
    for (int i = 0; i < MTe; i++)
        yTet[i][yTec[i] - 1] = 1.0;
        
    /* Create Model (multiple batches of B samples each) */
    MODEL* m = model_create(L,B,D,1,1); /* add bias, normalize */
    model_add(m,lstm_create(layers[0],1),"lstm"); 
    for (int i = 1; i < L - 1; i++)
        model_add(m,lstm_create(layers[i],1),"lstm");
    model_add(m,dense_create(N,"softmax"),"dense"); 

    model_compile(m,"cross-entropy",optimizer);
    
    float losses[epochs];
    float accuracies[epochs];
    float v_losses[epochs];
    float v_accuracies[epochs];

    model_fit(m,xTr,yTrt,sTr,STr,
                xTe,yTet,sTe,STe,
                epochs,learning_rate,weight_decay,
                losses,accuracies,v_losses,v_accuracies,
                "final=1 verbose=2");
    printf("\n");

    /* Test */
    ArrMN yp = allocmem(MTe,N,float); /* Predicted label vectors (000001 ... 100000) */
    VecS ytc = allocmem(STe,1,int);   /* True *sequence* label      */
    VecS ypc = allocmem(STe,1,int);   /* Predicted *sequence* label */

    int cm[N][N]; /* Confusion matrix */
    memset(cm,0,N * N * sizeof(int));

    /* Iterate over sequences, off is offset of sequence i first data point */
    cnt = 0;
    for (int i = 0, off = 0; i < STe; off += sTe[i++]) {
        /* All sequence data point have same true label, using the first one */
        ytc[i] = yTec[off] - 1; /* Note 1-based class numbers adjusted */
        model_predict(m,xTe + off,yp,sTe[i]);

        /* Calculate combined predictons for this sequence by class */
        float ay[N];
        fltclr(ay,N);
        for (int j = 0; j < sTe[i]; j++)
            for (int k = 0; k < N; k++) 
                ay[k] += yp[j][k];
        /* Find class with highest value */
        ypc[i] = 0; /* Assume first one */
        for (int k = 1; k < N; k++)
            if (ay[k] > ay[ypc[i]])
                ypc[i] = k;
        cm[ytc[i]][ypc[i]]++;
        if (ytc[i] == ypc[i])
            cnt++;
    }
    printf("Test accuracy %5.3f\n",((float) cnt) / STe);
    
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        plot_cm(cm,N,jvowels_class_names,
                epochs,losses,accuracies,v_losses,v_accuracies,title,"numbers"); 
    }
#else
    (void) accuracies;
    (void) losses;
    (void) v_accuracies;
    (void) v_losses;
    (void) title;
    printf("\n");
#endif
    freemem(yp);
    freemem(ytc);
    freemem(ypc);
    freemem(xTr);
    freemem(sTr);
    freemem(yTrc);
    freemem(yTrt);
    freemem(xTe);
    freemem(sTe);
    freemem(yTec);
    freemem(yTet);
    model_free(m);
    return 0;
}

/* Trains dense(projection) -> transformer stack -> dense(output) on a
 * selective-retrieval task ("last-pulse hold").
 *
 * The input is small random noise with occasional large pulses. The target
 * at each position is the value of the most recent pulse, held until the next
 * pulse. To solve the task, the transformer must identify pulse tokens by
 * content, use RoPE positional information to select the latest one, and
 * retrieve its value while ignoring the intervening noise. Causal attention
 * is used because the relevant pulse is always in the past.
 */

/* Fills one length-T sequence: X[T][2] (value,bias), yt[T][1] (held pulse). */
static void gen_sequence(float* X, float* yt, int T)
{
    float last = 0;
    int next = 3 + (int) urand(0,8);
    for (int t = 0; t < T; t++) {
        float v;
        if (t == next) {         /* pulse */
            float mag = 1 + urand(0,2);
            float sgn = (urand(0,1) < 0.5) ? -1 : 1;
            v = sgn * mag;
            last = v;            /* +/- 1.0 .. 3.0 */
            next = t + 8 + (int) urand(0,11); /* gap 8 .. 18 */
        }
        else {                   /* noise */
            v = urand(-0.6,0.6); /* +/- 0 .. 0.6 */
        }
        X[t * 2 + 0] = v;
        X[t * 2 + 1] = 1;        /* bias  */
        yt[t] = last;
    }
}

int test_transformer_retrieval(int heads, int model_dim, int ffn_dim,
                               int n_layers, const char* optimizer,
                               float learning_rate, float weight_decay,
                               int epochs)
{
    char* title = "last-pulse hold (selective retrieval, held-out sequence)";
    printf("\n\nTrains dense + transformer + dense on\n%s task\n\n",title);

    const int seq_len = 120;   /* sequence length T              */
    const int num_train = 128;   /* training sequences             */
    const int num_val = 32;    /* held-out validation sequences  */

    const int T = seq_len;      /* sequence length; model batch size = T    */
    const int L = n_layers + 2; /* proj dense + n transformers + out dense  */
    const int D = 2;            /* input dim: value + bias                  */
    const int N = 1;            /* output dim                               */

    /* num_* sequences of length T, laid out sequence by sequence, with
     * per-sequence lengths all equal to T. Constructed via flat views. */
    fArr2D xTr = allocmem(num_train * T, D, float);
    fArr2D yTr = allocmem(num_train * T, N, float);
    int*   sTr = allocmem(num_train, 1, int);
    fArr2D xVd = allocmem(num_val * T, D, float);
    fArr2D yVd = allocmem(num_val * T, N, float);
    int*   sVd = allocmem(num_val, 1, int);

    float* xTrf = (float*) xTr;
    float* yTrf = (float*) yTr;
    float* xVdf = (float*) xVd;
    float* yVdf = (float*) yVd;

    for (int s = 0; s < num_train; s++) {
        gen_sequence(xTrf + (size_t) s * T * D,yTrf + (size_t) s * T * N,T);
        sTr[s] = T;
    }
    for (int s = 0; s < num_val; s++) {
        gen_sequence(xVdf + (size_t) s * T * D,yVdf + (size_t) s * T * N,T);
        sVd[s] = T;
    }

    printf("%d layers (projection + %d transformer + output), seq length %d\n",
           L,n_layers,T);
    printf("model_dim %d, heads %d, ffn_dim %d\n",model_dim,heads,ffn_dim);

    /* One sequence per batch: batch size == T, so B = T/T = 1. */
    MODEL* m = model_create(L,T,D,0,0); /* don't add bias, don't normalize */
    model_add(m,dense_create(model_dim,"none"),"dense");
    for (int i = 0; i < n_layers; i++)
        model_add(m,transformer_create(heads,T,model_dim,ffn_dim,0),
                                                              "transformer");
    model_add(m,dense_create(N,"none"),"dense");
    model_compile(m,"mean-square-error",optimizer);

    /* Train, with validation on the held-out sequences */
    float losses[epochs];
    float accuracies[epochs];
    float v_losses[epochs];
    float v_accuracies[epochs];

    model_fit(m,xTr,yTr,sTr,num_train,
              xVd,yVd,sVd,num_val,
              epochs,learning_rate,weight_decay,
              losses,accuracies,v_losses,v_accuracies,
              "final=1 verbose=2");

    float y[T][N];
    model_predict(m,xVd,(fArr2D) y,T);

    printf("\n");

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        float xv[T];
        for (int t = 0; t < T; t++)
            xv[t] = (float) t;
        plot_graph(xv,(float*)y,yVdf,T,
                   epochs,losses,accuracies,v_losses,v_accuracies,title);
    }
#else
    (void) accuracies; (void) losses;
    (void) v_accuracies; (void) v_losses; (void) title;
    {
        int step = (T > 24) ? T / 24 : 1;
        printf("held-out sequence:\n");
        printf("  t    input     target     pred    %s\n","(* = pulse)");
        for (int t = 0; t < T; t += step) {
            float in = xVdf[t * D + 0];
            int is_pulse = (fabsf(in) >= 0.8);
            printf("%4d  %7.3f  %8.3f  %8.3f   %s\n",
                   t,in,yVdf[t * N + 0],y[t][0],is_pulse ? "*" : "");
        }
    }
    printf("\n");
#endif
    model_free(m);
    freemem(xTr); freemem(yTr); freemem(sTr);
    freemem(xVd); freemem(yVd); freemem(sVd);
    return 0;
}

int main(int argc, char** argv)
{
    const char* usage = 
        "Usage: testmodel [-h | <test number>...]           \n"
        "for example 'testmodel 1 3' will runs tests 1 and 3\n"
        "test numbers are 1..6 and are seperated by spaces  \n"
        "runs all tests if none specified                   \n";
    int tests[6] = {0};
    
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            int n = atoi(argv[i]);
            if (n <= 0 || n > (int) (sizeof(tests)/sizeof(int))) {
                fprintf(stderr,usage);
                exit(-1);
            }
            tests[n - 1] = 1;
        }
    }
    else {
        for (int i = 0; i < (int) (sizeof(tests)/sizeof(int)); i++)
            tests[i] = 1;
        printf("Running all tests\n");
        printf("Run 'testmodel -h' to list program options\n\n");
    }


    if (tests[0]) {
        init_lrng(42);
        const int layers[3] = {32,128,32};
        const float range[3] = {0.0,5.0,0.1};
        test_dense_regression(range,layers,3,"linear",0.0008,0.008,10000);
    }
    if (tests[1]) {
        init_lrng(42);
        const int layers[4] = {32,16,32,16};
        const float range[3] = {-10.0,10.0,0.1};
        test_lstm_regression(range,layers,4,"adamw",0.0002,0.02,1000);
    }
    if (tests[2]) {
        init_lrng(42);
        const int layers[1] = {35};
        const float range[3] = {-10.0,10.0,0.1};
        test_lstm_dense_regression(range,layers,1,"adamw",0.0003,0.09,1100);
    }
    if (tests[3]) {
        init_lrng(42);
        const int layers[2] = {12,12};
        test_dense_classification(layers,2,"linear",1,0.001,0.01,100);
    }
    if (tests[4]) {
        init_lrng(42);
        const int layers[1] = {80};
        test_lstm_dense_classification(layers,1,"adamw",6,0.001,0.01,10);
    }
    if (tests[5]) {
        init_lrng(42 * 2);
        const int heads = 4;
        const int model_dim = 32; /* must be a multiple of heads    */
        const int ffn_dim = 128;  /* 4 * model_dim                  */
        const int n_layers = 2;   /* transformer layers             */
        const char* optimizer = "adamw";
        const float lr = 0.0005;
        const float wd = 0.01;
        const int epochs = 20;

        test_transformer_retrieval(heads,model_dim,ffn_dim,
                                   n_layers,optimizer,lr,wd,epochs);
    }
    printf("\nAll tests completed\n\n");
    return 0;
}
