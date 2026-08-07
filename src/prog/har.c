/* Copyright (c) 2023-2024 Gilad Odinak */
/* Trains a multi layer LSTM followed by Dense layer to recognize Human
 * Activity Reocrdings (HAR).
 *
 * https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
 * https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include "mem.h"
#include "float.h"
#include "random.h"
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "accuracy.h"
#include "model.h"
#include "modelio.h"
#include "delta.h"
#include "onehot.h"

/* Directory of raw sensor data and labels (no trailing slash) */
const char* har_raw_data_dir = "data/har/RawData";

#define HAR_FEAT_CNT           6 /* Number of raw sensor signals         */
#define EXPENDED_FEAT_CNT     18 /* including additional delta features  */
#define HAR_CLASS_CNT         12 /* Number of activities and transitions */
#define REDUCED_CLASS_CNT      6 /* Number of distinct activities        */
#define HAR_SUBJECT_CNT       30 /* Number of people sampled (not used)  */

const char* har_class_names[HAR_CLASS_CNT] = {
    "walking","upstairs","downstairs","sitting","standing","laying",
    "stand2sit","sit2stand","sit2lie","lie2sit","stand2lie","lie2stand"
};

static const int har2reduced[HAR_CLASS_CNT] = {
    0,/* 0 walking*/
    1,/* 1 upstairs*/
    2,/* 2 downstairs*/
    3,/* 3 sitting*/
    4,/* 4 standing*/
    5,/* 5 laying*/
    3,/* 6 stand2sit*/
    4,/* 7 sit2stand*/
    5,/* 8 sit2lie*/
    3,/* 9 lie2sit*/
    5,/*10 stand2lie*/
    4 /*11 lie2stand*/
};

/* Reads activity samples from raw HAR data files.
 * 61 data files contain 3-axis accelometer or gyroscope mesaurments
 * of 30 subjects pefromming 6 activities.  The file names have this format:
 * <sgyro|acc>_exp<eid>_user<uid>.txt where eid is a 2 digit experiment
 * number, and uid is a two digits subjectr number.
 * A labels.txt file contains meta data about the raw mesaurments. Each
 * line describe a section of measurments in one of the raw files and has
 * this format: <eid> <uid> <aid> <start> <end>
 * eid - experiment id (1 - 61)
 * uid - subject id (1 - 30)
 * aid - activity id (1-12) (6 activities, and 6 transitions)
 * start - sample number in file where activity starts
 * end   - sample number in file past activity's end
 * Transitions and unlabeled data are mapped to "unknown" activity.
 *
 * Parameters:
 *
 *   input_dir     - a directory containing sample files, and labels.txt file
 *   max_sequences - maximum number of sequences (size of seq_length[])
 *   seq_length    - pointer to array the receives sequence lengths
 *   max_samples   - maximum number of samples (number of rows in x, and
 *                   number of elements in yc)
 *   x             - pointer to array that receives measurment vectors
 *                   measurment data is expended to 18 dimensions
 *   yc            - true labels of input vectors
 *   sid           - only return data for these subjects
 *   sid_len       - number of entries in sid array. 
 *
 * Returns:   
 * An array of input vectors in x, and their corresponding labels in yc,
 * and the length of each sequence of input vectors in seq_length.
 * 
 * This function returns the actual number of sequences, which is also the 
 * actual number of elements in seq_length. It returns 0 if an error occurred.
 */
int read_har_files(const char* input_dir,
                   int max_sequences, int *seq_length,
                   int max_samples, fArr2D x_/*[][]*/, int* yc/*[]*/,
                   const int* sid/*[]*/, int sid_len)
{
    typedef float (*ArrMN)[EXPENDED_FEAT_CNT];
    ArrMN x = (ArrMN) x_;
    char fnl[256];
    snprintf(fnl,sizeof(fnl),"%s/labels.txt",input_dir);
    FILE* fpl = fopen(fnl,"rb");
    if (fpl == NULL) {
        fprintf(stderr,"%s: failed to open file for read\n",fnl);
        return 0;
    }
    char fna[256];      /* Accelerator data current file name */
    char fng[256];      /* Accelerator data current file name */
    FILE* fpa = NULL;
    FILE* fpg = NULL;
    int filesample = 0; /* Current data file(s) sample */
    int fileline = 1;   /* Current date file(s) line   */
    int seqcnt = -1; 
    int lasteid = -1;
    int samplecnt = 0;  /* Total number of samples */
    for (int lineno = 1;; lineno++) {
        char buf[256];
        char* line = fgets(buf,sizeof(buf),fpl);
        if (line == NULL) { /* End of data */
            if (seqcnt >= 0) {  /* Finalize last sequence */
                int M = seq_length[seqcnt];
                int a = samplecnt - M;
                int N = EXPENDED_FEAT_CNT;
                calculate_deltas(x[a],M,N,0,6,6,5);  /* deltas       */
                calculate_deltas(x[a],M,N,6,12,6,5); /* delta-deltas */
            }
            seqcnt++; /* Count last sequence */
            break;
        }
        int eid;   /* Experiment ID */
        int uid;   /* User ID       */
        int aid;   /* Activity ID   */
        int start; /* Sample start  */
        int end;   /* Sample end    */
        int cnt = sscanf(line," %d %d %d %d %d",&eid,&uid,&aid,&start,&end);
        if (cnt != 5) {
            fprintf(stderr,
                "File %s, at line %d: failed to read 5 values\n",fnl,lineno);
            break;
        }
        aid--; /* File classes are 1-based */
        aid = har2reduced[aid];
        /* Check if requested subject */
        int sid_inx;
        for (sid_inx = 0; sid_inx < sid_len; sid_inx++)
            if (sid[sid_inx] == uid)
                break;
        if (sid_inx >= sid_len)
            continue; /* Not one of requested subjects */

        if (eid != lasteid) {   /* New expriment file */
            if (seqcnt >= 0) {  /* Finalize previous sequence */
                int M = seq_length[seqcnt];
                int a = samplecnt - M;
                int N = EXPENDED_FEAT_CNT;
                calculate_deltas(x[a],M,N,0,6,6,5);  /* deltas       */
                calculate_deltas(x[a],M,N,6,12,6,5); /* delta-deltas */
                fclose(fpa);
                fclose(fpg);
            }
            seqcnt++; /* New sequence */
            if (seqcnt >= max_sequences) {
                fprintf(stderr,
                    "Reached max number of sequences (%d)\n",max_sequences);
                break;
            }
            seq_length[seqcnt] = 0;
            snprintf(fna,sizeof(fna),
                     "%s/acc_exp%02d_user%02d.txt",input_dir,eid,uid);
            fpa = fopen(fna,"rb");
            if (fpa == NULL) {
                fprintf(stderr,"%s: failed to open file for read\n",fna);
                break;
            }
            snprintf(fng,sizeof(fng),
                     "%s/gyro_exp%02d_user%02d.txt",input_dir,eid,uid);
            fpg = fopen(fng,"rb");
            if (fpg == NULL) {
                fprintf(stderr,"%s: failed to open file for read\n",fng);
                break;
            }
            filesample = 0;
            fileline = 1;
            lasteid = eid;
        }
        while (filesample < end) {
            if (samplecnt >= max_samples) {
                fprintf(stderr,
                    "Reached max number of samples (%d)\n",max_samples);
                break;
            }
            int i = samplecnt;
            line = fgets(buf,sizeof(buf),fpa);
            if (line == NULL) {
                fprintf(stderr,
                    "%s: unexpected end of file at line %d\n",fna,fileline);
                break;
            }
            cnt = sscanf(line," "FMTF" "FMTF" "FMTF,&x[i][0],&x[i][1],&x[i][2]);
            if (cnt != 3) {
                fprintf(stderr,
                    "%s, at line %d: failed to read 3 values\n",fna,fileline);
                break;
            }
            line = fgets(buf,sizeof(buf),fpg);
            if (line == NULL) {
                fprintf(stderr,
                    "%s: unexpected end of file at line %d\n",fng,fileline);
                break;
            }
            cnt = sscanf(line," "FMTF" "FMTF" "FMTF,&x[i][3],&x[i][4],&x[i][5]);
            if (cnt != 3) {
                fprintf(stderr,
                    "%s, at line %d: failed to read 3 values\n",fng,fileline);
                break;
            }
            yc[i] = aid;
            fileline++;
            filesample++;
            samplecnt++;  
            seq_length[seqcnt]++;
        }
        if (filesample < end) /* Failed to read data */
            break;
    }
    if (fpl != NULL)
        fclose(fpl);
    if (fpa != NULL)
        fclose(fpa);
    if (fpg != NULL)
        fclose(fpg);
    return seqcnt;
}

int dataset_size(int seqcnt, int* seq_lengths)
{
    int total = 0;
    for (int i = 0; i < seqcnt; i++)
        total += seq_lengths[i];
    return total;
}

/* Trains a multi layer LSTM followed by Dense layer to recognize 
 * Human Activity Reocrdings (HAR).
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 32 and 16, 
 * two LSTM layers will be created with size of 32 and 16, followed by 
 * a Dense layer of size 4.
 *
 * The remaining parameters are passed to model_compile() and model_fit()
 */
int har_lstm_dense_classification(const char* loadmodel, const char* storemodel,
                        const int layers[], int layers_cnt, char* optimizer,
                        int batch_size, int test_batch_size, int stateful,
                        float learning_rate, float weight_decay, int epochs)
{
    printf("\nTrains a multi layer LSTM followed by Dense layer to predict the\n");
    printf("classes of samples from the Human Activity Recordings dataset\n\n");
    printf("Run 'har -h' to list program options\n\n");
    printf("Training with default parameters may take  a few minutes\n\n");
    const int L = layers_cnt + 1;    /* Number of layers        */
    const int B = batch_size;        /* Train batch size        */
    const int Dr = HAR_FEAT_CNT;     /* Raw data dimension      */
    const int D = EXPENDED_FEAT_CNT; /* Expended data dimension */
    const int N = REDUCED_CLASS_CNT; /* Output vector dimension */

    printf("%d layers (including output layer) ",L);
    for (int i = 0; i < layers_cnt; i++)
        printf("%d,",layers[i]);
    printf("%d.\n",N);
    printf("Input dimension %d. Expended input dimension %d.\n",Dr,D);
    printf("Train batch size %d. Test batch size %d\n",B,test_batch_size);
    printf("%d epochs, ",epochs); 
    printf("learning rate %g, weight decay %g \n",learning_rate,weight_decay);

    typedef float (*ArrMD)[D];
    typedef float (*ArrMN)[N];
    typedef int (*VecM);
    typedef int (*VecS);

    const int uTr[20] = {1,2,4,5,6,8,10,11,13,14,15,16,17,20,21,22,25,26,28,30};
    const int uVd[5] = {7,12,18,23,27};    
    const int uTe[5] = {3,9,19,24,29};    

    int STr = 41;
    int MTr = 700000;
    ArrMD xTr = allocmem(MTr,D,float);  /* HAR training Dataset             */
    VecS  sTr = allocmem(STr,1,int);    /* HAR training sequence lengths    */
    VecM  yTrc = allocmem(MTr,1,int);   /* True labels (values 0,1,2,3,4,5) */
    ArrMN yTrv = allocmem(MTr,N,float); /* True labels one-hot vectors      */

    int SVd = 10;
    int MVd = 200000;
    ArrMD xVd = allocmem(MVd,D,float);  /* HAR validation Dataset           */
    VecS  sVd = allocmem(SVd,1,int);    /* HAR validation sequence lengths  */
    VecM  yVdc = allocmem(MVd,1,int);   /* True labels (values 0,1,2,3,4,5) */
    ArrMN yVdv = allocmem(MVd,N,float); /* True labels one-hot vectors      */

    int STe = 10;
    int MTe = 200000;
    ArrMD xTe = allocmem(MTe,D,float);  /* HAR training Dataset             */
    VecS  sTe = allocmem(STe,1,int);    /* HAR training sequence lengths    */
    VecM  yTec = allocmem(MTe,1,int);   /* True labels (values 0,1,2,3,4,5) */
    ArrMN yTev = allocmem(MTe,N,float); /* True labels one-hot vectors      */

    /* Read data */
    printf("Loading data...\n");
    STr = read_har_files(har_raw_data_dir,STr,sTr,MTr,xTr,yTrc,uTr,20);
    SVd = read_har_files(har_raw_data_dir,SVd,sVd,MVd,xVd,yVdc,uVd,5);
    STe = read_har_files(har_raw_data_dir,STe,sTe,MTe,xTe,yTec,uTe,5);

    /* Calculate actual dataset sizes */
    MTr = dataset_size(STr,sTr);
    MVd = dataset_size(SVd,sVd);
    MTe = dataset_size(STe,sTe);

    printf("%d training sequences (%d samples)\n",STr,MTr);
    printf("%d validation sequences (%d samples)\n",SVd,MVd);
    printf("%d test sequences (%d samples)\n",STe,MTe);

    /* Encode yc as one-hot vectors */    
    onehot_encode(yTrc,yTrv,MTr,N);
    onehot_encode(yVdc,yVdv,MVd,N);
    onehot_encode(yTec,yTev,MTe,N);
        
    MODEL* m;
    if (loadmodel != NULL) 
        m = load_model(loadmodel);
    else {
        /* Create Model (to process multiple batches of B samples each) */
        m = model_create(L,B,D,1,1); /* Notice adding bias to input */
        model_add(m,lstm_create(layers[0],stateful),"lstm"); 
        for (int i = 1; i < L - 1; i++)
            model_add(m,lstm_create(layers[i],stateful),"lstm");
        model_add(m,dense_create(N,"softmax"),"dense");
        model_compile(m,"cross-entropy",optimizer);
    }
    float losses[epochs];
    float accuracies[epochs];
    float v_losses[epochs];
    float v_accuracies[epochs];

    if (epochs > 0) {
        printf("Training...");
        model_fit(m,xTr,yTrv,sTr,STr,
                    xTe,yTev,sTe,STe,
                    epochs,learning_rate,weight_decay,
                    losses,accuracies,v_losses,v_accuracies,
                    "verbose=2");
    }

    if (storemodel != NULL) 
        store_model(m,storemodel);

    printf("Testing...\n");
    ArrMN yp = allocmem(MTe,N,float);
    VecS ytc = yTec;                  /* True *sequence* label      */
    VecS ypc = allocmem(MTe,1,int);   /* Predicted *sequence* label */
    int i, j;
    int off;    /* current sequence offset within test data     */
    float mcnt; /* Number of matching labels                    */

    int cm[N][N]; /* Confusion matrix */
    memset(cm,0,N * N * sizeof(int));
    
    model_set_batch_size(m,test_batch_size);
    mcnt = 0;
    for (i = 0, off = 0; i < STe; off += sTe[i++]) {
        model_predict(m,xTe + off,yp + off,sTe[i]);
        onehot_decode(yp + off,ypc + off,sTe[i],N);
        for (j = 0; j < sTe[i]; j++) {
            cm[ytc[off + j]][ypc[off + j]]++;
            if (ytc[off + j] == ypc[off + j])
                mcnt++;
        }
    }
    printf("Test accuracy %5.3f\n", mcnt / MTe);

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        char* title = "Confusion Matrix";
        plot_cm(cm,N,har_class_names,
                epochs,losses,accuracies,v_losses,v_accuracies,title,"both"); 
    }
#else
    (void) accuracies;
    (void) losses;
    (void) v_accuracies;
    (void) v_losses;
    printf("\n");
#endif
    freemem(yp);
    freemem(ypc);
    freemem(xTr);
    freemem(sTr);
    freemem(yTrc);
    freemem(yTrv);
    freemem(xVd);
    freemem(sVd);
    freemem(yVdc);
    freemem(yVdv);
    freemem(xTe);
    freemem(sTe);
    freemem(yTec);
    freemem(yTev);
    model_free(m);
    return 1;
}

int main(int argc, char** argv)
{
    const char* usage = 
        "Usage: har [-h] [-e <epochs>]                                  \n"
        "           [-r <learning rate>] [-w weight decay]              \n"
        "           [-b <train batch size>[:<test batch size]]          \n"
        "           [-S <stateful|stateless] [-L 's1 s2 ...']           \n"
        "           [-l <model file>] [-s <model file>]                 \n"
        "                                                               \n"
        " -L: LSTM layer specification. One additional output layer     \n"
        "     is implied. So for example -L '126 64' specifies two      \n"
        "     LSTM layers followed by a Dense output layer.             \n"
        "     Notice that the specification is quoted.                  \n"
        " -l: Load model from file and continue to train for number     \n"
        "     of epochs specified by -e; ignore -L -b and -t options.   \n"
        " -s: Store model in file at the end of training                \n"
        "\n";

    int epochs = 7, bsize = 64, tbsize = 64;
    float lr = 0.0001, wd = 0.1;
    char *loadfile = NULL, *storefile = NULL;
    int lyrcnt = 2;
    int layers[5] = {64,64}; /* Up to 4 layers */
    int stateful = 1;
    int opt;
    while ((opt = getopt(argc, argv, "e:r:w:b:l:s:S:L:h")) != -1) {
        switch (opt) {
            case 'h': printf(usage); exit(0);
            case 'e': epochs = atoi(optarg); break;
            case 'l': loadfile = optarg; break;
            case 's': storefile = optarg; break;
            case 'r': lr = atof(optarg); break;
            case 'w': wd = atof(optarg); break;
            case 'b': 
                bsize = atoi(optarg); 
                char *t = index(optarg,':');
                if (t != NULL)
                    tbsize = atoi(t + 1);
                else
                    tbsize = bsize;
            break;
            case 'S': 
                if (strcmp(optarg,"stateful")==0)
                    stateful = 1;
                else
                if (strcmp(optarg,"stateless")==0)
                    stateful = 0;
                else {
                    fprintf(stderr,"timit: invalid value to -S option\n");
                    printf(usage);
                }
            break;
            case 'L':
                lyrcnt = sscanf(optarg,"%d %d %d %d %d",&layers[0],
                                &layers[1],&layers[2],&layers[3],&layers[4]);
                if (lyrcnt >= 5) {
                    fprintf(stderr,"timit: too many LSTM layers\n");
                    printf(usage);
                }
            break;
            case '?': 
            default: 
                fprintf(stderr,"har: syntax error"); 
                printf(usage);
                exit(-1);
        }
    }
    int ok;
    init_lrng(42);
    ok = har_lstm_dense_classification(loadfile,storefile,layers,lyrcnt,
                                   "adamw",bsize,tbsize,stateful,lr,wd,epochs);
    return (ok) ? 0 : -1;
}
