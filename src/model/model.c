/* Copyright (c) 2023-2024 Gilad Odinak */
/* Multi layer neural network model and functions */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "mem.h"
#include "random.h"
#include "etime.h"
#include "array.h"
#include "loss.h"
#include "ctc.h"
#include "adamw.h"
#include "dense.h"
#include "lstm.h"
#include "transformer.h"
#include "negsample.h"
#include "layer.h"
#include "activation.h"
#include "clip.h"
#include "batch.h"
#include "normalize.h"
#include "accuracy.h"
#include "model.h"
#include "modelio.h"

static void model_batch_forward(MODEL* m, fArr2D x, int training, fArr2D* yp);
static void model_batch_backward(MODEL* m, fArr2D x, fArr2D* dy, fArr2D* yp);
static void model_update(MODEL* m, float learning_rate, float weight_decay);
static void print_status(int epoch, int nepochs, int progress, float etime,
                             float loss, float acc, float v_loss, float v_acc);

static const char* find_kwarg(const char* kwargs, const char* key);
static void get_kw_int(const char* kwargs, const char* key, int* val);
static void get_epoch_params(const char* sch, int epoch, float* lr, float* wd);

static inline void reset_state(MODEL* m)
{
    for (int i = 0; i < m->num_layers; i++)
        layer_reset(&m->layer[i]);
}

/* Creates a container for multi layer neural network.
 * num_layers specifies the number of layers
 * batch_size is the number of vector inputs processed together
 * between model updates.
 * input_dim is the dimension of the model first layer input.
 *
 * If add_bias is zero, input_dim includes a bias dimension
 * whose value is 1.0; otherwise, a bias dimension is added internally.
 *
 * If normalize is not zero, normalizes input feature vectors by feature,
 * to have mean of zero and standard deviation of one.
 *
 * Use model_add() to add layers to the model up to numlayers.
 *
 * Note that the output dimension is determined by the size of the last layer.
 */
MODEL* model_create(int num_layers,
                    int batch_size, int input_dim, int add_bias, int normalize)
{
    MODEL* m = allocmem(1,1,MODEL);
    m->num_layers = num_layers;
    m->layer = allocmem(1,m->num_layers,LAYER);
    m->batch_size = batch_size;
    m->input_dim = input_dim;
    m->add_bias = (add_bias) ? 1 : 0;
    m->normalize = (normalize) ? 1 : 0;
    m->compiled = 0;
    m->final = 0;
    return m;
}

/* Frees the memory allocated by model_create() and all added layers */
void model_free(MODEL* m)
{
    for (int i = 0; i < m->num_layers; i++) {
        layer_free(&m->layer[i]);
        if (m->layer[i].grads) {
            for (int j = 0; j < m->layer[i].num_grads; j++)
                freemem(m->layer[i].grads[j]);
            freemem(m->layer[i].grads);
        }
    }
    freemem(m->ctc);
    freemem(m->mean);
    freemem(m->sdev);
    freemem(m->layer);
    freemem(m);
}

/* Adds a layer to a model
 * m points to a model
 * layer points to a neural network (e.g. DENSE) to be added as a layer
 * type is the type of the layer: "dense", "lstm", "transformer" or "negsample"
 *
 * The layer is added after all other layers in the model
 */
void model_add(MODEL* m, void* layer, const char* type)
{
    int i;
    for (i = 0; i < m->num_layers; i++)
        if (m->layer[i].type == 0)
            break;
    if (i >= m->num_layers) {
        fflush(stdout);
        fprintf(stderr,"model_create: all layers already added\n");
        exit(-1);
    }
    if (!strcasecmp("dense",type)) {
        m->layer[i].type = 'd';
        m->layer[i].dense = layer;
    }
    if (!strcasecmp("lstm",type)) {
        m->layer[i].type = 'l';
        m->layer[i].lstm = layer;
    }
    if (!strcasecmp("transformer",type)) {
        m->layer[i].type = 't';
        m->layer[i].transformer = layer;
    }
    if (!strcasecmp("negsample",type)) {
        m->layer[i].type = 'n';
        m->layer[i].negsample = layer;
    }
    if (m->layer[i].type == 0) {
        fflush(stdout);
        fprintf(stderr,"model_create: invalid layer type '%s'\n",type);
        exit(-1);
    }
}

/* Prepares model for training.
 *
 * loss_func can be one of mean-square-error cross-entropy ctc negative-sampling
 *
 * optimizer can be one of linear, adamw
 *
 * both optimizers incorporate weight decay; to disable, set it to 0 when
 * invoking model_fit().
 */
void model_compile(MODEL* m, const char* loss_func, const char* optimizer)
{
    if (m->compiled) {
        fflush(stdout);
        fprintf(stderr,"model_compile: model already compiled\n");
        exit(-1);
        return;
    }
    if (!strcasecmp("mean-square-error",loss_func)) m->loss_func = 'm';
    if (!strcasecmp("cross-entropy",loss_func)) m->loss_func = 'c';
    if (!strcasecmp("ctc",loss_func)) m->loss_func = 'C';
    if (!strcasecmp("negative-sampling",loss_func)) m->loss_func = 'N';
    if (m->loss_func == 0) {
        fflush(stdout);
        fprintf(stderr,"model_compile: invalid loss function '%s'\n",loss_func);
        exit(-1);
    }
    if (!strcasecmp("linear",optimizer)) m->optimizer = 'l';
    if (!strcasecmp("adamw",optimizer)) m->optimizer = 'a';
    if (m->optimizer == 0) {
        fflush(stdout);
        fprintf(stderr,"model_compile: invalid optimizer '%s'\n",optimizer);
        exit(-1);
    }
    if (m->num_layers < 1) {
        fflush(stdout);
        fprintf(stderr,"model_compile: model does not have any layers\n");
        exit(-1);
    }
    m->compiled = 1;
    if (m->normalize) {
        int D = m->input_dim;           /* Input dimension: may include bias */
        int Dx = D - (1 - m->add_bias); /* Input dimension excluding bias    */
        m->mean = allocmem(1,Dx,float);
        m->sdev = allocmem(1,Dx,float);
    }

    int D = m->input_dim + m->add_bias;
    int B = m->batch_size;
    int L = m->num_layers;
    for (int i = 0; i < L; i++) {
        /* layer_init returns the layer's output size,
         * which is the input size of the next layer.
         */
        D = layer_init(&m->layer[i],D,B);
    }
    m->output_dim = layer_output_dim(&m->layer[L - 1]);
    /* Negative-sampling targets are a single word index per row;
     * all other losses use dense targets of the model's output width.
     */
    m->target_dim = (m->loss_func == 'N') ? 1 : m->output_dim;
    if (m->loss_func == 'C')
        m->ctc = ctc_create(B,m->output_dim,0);

    /* Allocate gradient arrays */
    for (int i = 0; i < m->num_layers; i++)
        layer_alloc_grads(&m->layer[i],m->optimizer);
}

/* Sets a new batch size.
 *
 * This function changes the batch size of an existing, possibly trained,
 * model. Smaller batch size reduces memory requirements and decoding latency
 * but may also reduce accuracy.
 *
 * Parameters:
 *   batch_size - Number of input vectors processed simultaneously
 */
void model_set_batch_size(MODEL* m, int batch_size)
{
    if (m->batch_size == batch_size)
        return;
    m->batch_size = batch_size;
    for (int i = 0; i < m->num_layers; i++)
        layer_set_batch_size(&m->layer[i],batch_size);
    if (m->ctc != NULL) {
        ctc_free(m->ctc);
        m->ctc = ctc_create(m->batch_size,m->output_dim,0);
    }
}

/* Trains model on data xTr and true outputs yTr. The data is organized as
 * a list of data sample sequences of varying lengths and corresponding
 * true outputs. The dimension of the vectors in x sequences is
 * the input dimension of the first layer, and the dimension of the
 * vectors in y sequences is the output dimension of the last layer.
 *
 * This function may be called more than once to train a pre-trained
 * model.
 *
 * xTr is an array of vectors of one or more training data sequences.
 *
 * yTr is an array of vectors of corresponding true outputs.
 *
 * lenTr is an array of integers denoting the lengths of training sequences.
 * if the data consist of one sequence (or is not sequential) it can be NULL.
 *
 * numTr is the number of vectors in xTr, yTr if lenTr is NULL. Otherwise,
 * it is the number of sequences; the total number of vectors is the sum
 * of the values in lenTr.
 *
 * xVd is an array of vectors of one or more training data sequences.
 *
 * yVd is an array of vectors of corresponding true outputs.
 *
 * lenVd is an array of integers denoting the lengths of training sequences.
 * if the data consist of one sequence (or is not sequential) it can be NULL.
 *
 * numVd is the number of vectors in xVd, yVd if lenVd is NULL. Otherwise,
 * it is the number of sequences; in that case, the total number of vectors
 * is the sum of the values in lenVd. Set numVd to 0 to indicate that no
 * validation data is provided. In that case, xVd, yVd and lenVd should
 * be set to NULL.
 *
 * num_epochs is the number of iterations through the entire dataset.
 *
 * learning_rate is a gradient multiplier controling the rate of descent.
 *
 * weight_decay is a multiplier that suppresses weights magnitude.
 * 
 * If losees is not NULL, it points to an array of num_epoch elements
 * that are updated with the loss value of each epoch.
 *
 * If accuracies is not NULL, it points to an array of num_epoch elements
 * that are updated with the accuracy at the end of each epoch.
 * For regression, accuracy is the R-squared coefficient
 * For classification, accuracy is the fraction of predicted labels that
 * matched the true labels.
 *
 * Similarily, if validation data is present and v_losses, v_accuracies 
 * are not NULL, they are updated with the validation loss and accuracy
 * at the end of each epoch.
 *
 * kwargs points to a string that specifies additional optional 
 * parameters. The parameters are in key=value format, separated by 
 * spaces:
 *
 * shuffle applies only when data consists of one sequence; if not zero,
 * samples within the sequence are shuffled. Otherwise, sequences are 
 * always shuffled, and samples within sequences are never shuffled.
 * Default value is 1.
 *
 * If final is not zero, frees gradients memory at the end of training.
 * Otherwise, memory is retained, allowing further training of the model.
 * Default value is 0.
 *
 * If verbose is not zero, prints the loss and accuracy values at the 
 * end of each epoch to standard output; if it is greater then 1, prints
 * each epoch's loss and accuracy on a separate line.
 * Default value is 0.
 *
 * Schedule specified a training schedule with variable learning rate 
 * and  * weight decay. The format of this parameter is <e>:<l>:<w>,...
 * where <e> is number of epochs, <l> and <w> are the learning rate and 
 * weight decay values for these epochs.
 */ 
void model_fit(MODEL* m, 
    const fArr2D xTr, const fArr2D yTr, const int *lenTr, int numTr, 
    const fArr2D xVd, const fArr2D yVd, const int *lenVd, int numVd, 
    int num_epochs, float learning_rate, float weight_decay,
    float* losses, float* accuracies, 
    float* v_losses, float* v_accuracies,
    const char* kwargs)
{
    if (m->final) {
        fflush(stdout);
        fprintf(stderr,"model_fit: final model cannot be trained\n");
        exit(-1);
        return;
    }
    int verbose = 0; get_kw_int(kwargs,"verbose",&verbose);
    int shuffle = 1; get_kw_int(kwargs,"shuffle",&shuffle);
    int final = 0;   get_kw_int(kwargs,"final",&final);
    const char* sch = find_kwarg(kwargs,"schedule");
    int L = m->num_layers;
    int N = m->output_dim;          /* Dimension of model output vectors */
    int Nt = m->target_dim;         /* Target width (1 for negsample)    */
    int B = m->batch_size;          /* Batch size (all layers)           */
    int D = m->input_dim;           /* Input dimension: may include bias */
    int Dx = D - (1 - m->add_bias); /* Input dimension excluding bias    */
    int Db = D + m->add_bias;       /* Input dimension including bias    */
    
    int MTr = 0;           /* Total number of training samples        */
    if (lenTr != NULL) {
    for (int i = 0; i < numTr; i++)
        MTr += lenTr[i];
    }
    else
        MTr = numTr;
         
    int MVd = 0;           /* Total number of validation samples      */
    if (lenVd != NULL) {
        for (int i = 0; i < numVd; i++)
            MVd += lenVd[i];
    }
    else
        MVd = numVd; 

    typedef float (*VecDx);
    VecDx mean = (VecDx) m->mean;
    VecDx sdev = (VecDx) m->sdev;
    if (m->normalize)
        calculate_mean_sdev(xTr,MTr,D,mean,sdev,D - Dx);

    BATCH* bTr = batch_create(xTr,D,yTr,Nt,B,lenTr,numTr,shuffle,m->add_bias);
    BATCH* bVd = NULL;
    if (MVd > 0) /* Notice validation data not shuffled */
        bVd = batch_create(xVd,D,yVd,Nt,B,lenVd,numVd,0,m->add_bias);
        
    fArr2D dy[L];  /* Gradients with respect to the inputs          */
    for (int i = 0; i < L; i++)
        dy[i] = allocmem(layer_batch_size(&m->layer[i]),
                         layer_output_dim(&m->layer[i]),float);
    
    /* Allocate memory for one batch */
    typedef float (*ArrBDb)[Db];
    typedef float (*ArrBN)[Nt];
    ArrBDb x = (ArrBDb) allocmem(B,Db,float); /* Array of samples      */
    ArrBN yt = (ArrBN) allocmem(B,Nt,float);  /* Array of true outputs */

    /* Track training loss, accuracy and model improvement across epochs */
    float loss = 0;
    float accuracy = 0;
    float match_cnt;
    int sample_cnt;

    /* Track validation loss, accuracy */
    float v_loss = 0;
    float v_accuracy = 0;
    float v_match_cnt;
    int v_sample_cnt;

    if (verbose)
        printf("\n");

    float start_time = current_time();
    int epoch;
    for (epoch = 0; epoch < num_epochs; epoch++) {
        loss = 0;
        match_cnt = 0;
        sample_cnt = 0;

        if (sch != NULL)
            get_epoch_params(sch,epoch,&learning_rate,&weight_decay);

        batch_shuffle(bTr);
        reset_state(m);
        for (;;) {
            fArr2D yp[L]; /* Pointers to layers' prediction arrays */
            int cnt = batch_copy(bTr,x,yt);
            if (cnt == 0)
                break;
            if (m->normalize)
                normalize(x,B,Db,mean,sdev,1);
            model_batch_forward(m,x,1,yp);
            sample_cnt += cnt;

            /* Note that gradient calculation below is additive.
             * If the actual number of samples in the last batch 
             * is less than batch size (cnt < B), only that number
             * of samples is used to calculate the gradients.
             */
            switch(m->loss_func) {
                case 'm':
                    loss += mean_square_error(yp[L - 1],yt,cnt,N) * 100;
                    match_cnt += R2_sum(yp[L - 1],yt,cnt,N);
                    dLdy_mean_square_error(yp[L - 1],yt,dy[L - 1],cnt,N);
                break;
                case 'c':
                    loss += cross_entropy_loss(yp[L - 1],yt,cnt,N);
                    match_cnt += match_sum(yp[L - 1],yt,cnt,N);
                    dLdy_cross_entropy_loss(yp[L - 1],yt,dy[L - 1],cnt,N);
                break;
                case 'C':
                    loss += ctc_loss(m->ctc,yp[L - 1],yt,cnt,N);
                    match_cnt += ctc_accuracy(m->ctc,yp[L - 1],yt,cnt,N);
                    dLdy_ctc_loss(m->ctc,yp[L - 1],yt,dy[L - 1],cnt,N);
                break;
                case 'N': {
                    /* Negative-sampling : loss and grad w.r.t. h are
                     * computed here (h = yp[L-1], identity forward); 
                     * the gradient into the stack is written to dy[L-1]
                     * and the output-weight grads accumulate into the
                     * layer's grads.
                     */
                    NEGSAMPLE* head = m->layer[L - 1].negsample;
                    int correct = 0;
                    loss += negsample_loss(head,yp[L - 1],yt,
                                           m->layer[L - 1].grads[0],
                                           dy[L - 1],cnt,&correct);
                    match_cnt += correct;
                }
                break;
            }
            model_batch_backward(m,x,dy,yp);
            if (verbose) {
                print_status(epoch + 1,num_epochs,
                            (B < MTr) ? sample_cnt * 100 / MTr : -1,
                            elapsed_time(start_time),
                            loss / sample_cnt, match_cnt / sample_cnt,-1,-1);
            }
            model_update(m,learning_rate,weight_decay); /* Update weights */
            if (batch_eos(bTr))
                reset_state(m);
        }
        loss /= sample_cnt;
        accuracy = match_cnt / sample_cnt;
        if (verbose) {
            print_status(epoch + 1,num_epochs,
                         (B < MTr) ? 100 : -1,
                         elapsed_time(start_time),
                         loss,accuracy,-1,-1);
        }
        if (losses != NULL) 
            losses[epoch] = loss;
        if (accuracies != NULL) 
            accuracies[epoch] = accuracy;
        if (MVd > 0) { /* Validation data present */
            v_loss = 0;
            v_match_cnt = 0;
            v_sample_cnt = 0;
            
            batch_shuffle(bVd); /* Only resets, doesn't actually shuffle */
            reset_state(m);
            for (;;) {
                fArr2D yp[L]; /* Pointers to layers' prediction arrays */
                int cnt = batch_copy(bVd,x,yt);  
                if (cnt == 0)
                    break;
                if (m->normalize)
                    normalize(x,B,Db,mean,sdev,1); 
                model_batch_forward(m,x,0,yp);
                v_sample_cnt += cnt;

                switch(m->loss_func) {
                    case 'm':
                        v_loss += mean_square_error(yp[L - 1],yt,cnt,N) * 100;
                        v_match_cnt += R2_sum(yp[L - 1],yt,cnt,N);
                    break;
                    case 'c':
                        v_loss += cross_entropy_loss(yp[L - 1],yt,cnt,N);
                        v_match_cnt += match_sum(yp[L - 1],yt,cnt,N);
                    break;
                    case 'C':
                        v_loss += ctc_loss(m->ctc,yp[L - 1],yt,cnt,N);
                        v_match_cnt += ctc_accuracy(m->ctc,yp[L - 1],yt,cnt,N);
                    break;
                    case 'N': {
                        /* Eval only: compute loss/accuracy. The grad buffers
                         * (dy[L-1], head grads) are written but never applied,
                         * since no model_update runs during validation.
                         */
                        NEGSAMPLE* head = m->layer[L - 1].negsample;
                        int correct = 0;
                        v_loss += negsample_loss(head,yp[L - 1],yt,
                                                 m->layer[L - 1].grads[0],
                                                 dy[L - 1],cnt,&correct);
                        v_match_cnt += correct;
                    }
                    break;
                }
                if (verbose) {
                    print_status(epoch + 1,num_epochs,
                            (B < MVd) ? v_sample_cnt * 100 / MVd : -1,
                            elapsed_time(start_time),
                            loss,accuracy,
                            v_loss / v_sample_cnt, v_match_cnt / v_sample_cnt);
                }
                if (batch_eos(bVd))
                    reset_state(m);
            }
            v_loss /= v_sample_cnt;
            v_accuracy = v_match_cnt / v_sample_cnt;
            if (verbose) {
                print_status(epoch + 1,num_epochs,
                             (B < MVd)? 100 : -1,
                             elapsed_time(start_time),
                             loss,accuracy,v_loss,v_accuracy);
            }
            if (v_losses != NULL) 
                v_losses[epoch] = v_loss;
            if (v_accuracies != NULL) 
                v_accuracies[epoch] = v_accuracy;
        }
        if (verbose > 1)
            printf("\n");
    }
    for (int i = 0; i < L; i++) {
        freemem(dy[i]);
    }
    freemem(x);
    freemem(yt);
    batch_free(bTr);
    if (bVd != NULL)
        batch_free(bVd);
    if (final) {
        m->final = 1;
        for (int i = 0; i < m->num_layers; i++) {
            if (m->layer[i].grads) {
                for (int j = 0; j < m->layer[i].num_grads; j++)
                    freemem(m->layer[i].grads[j]);
                free(m->layer[i].grads);
                m->layer[i].num_grads = 0;
                m->layer[i].grads = NULL;
            }
        }
    }
    if (verbose)
        printf("\n");
}
         
/* Predicts the outputs of the inputs samples in x, and returns them in yp.
 *
 * x is an array of input samples.
 * y is an array to be updated with output predictions.
 * len is number of smaples.
 */
void model_predict(MODEL* m, const fArr2D x_, fArr2D y_, int len)
{
    int L = m->num_layers;
    int N = m->output_dim;    /* Dimension of model output vectors */
    /* Negative-sampling head outputs a full-vocab distribution [len][K],
     * matching dense + softmax; all other heads output [len][output_dim].
     */
    int No = (m->loss_func == 'N') ? m->layer[L - 1].negsample->K : N;
    int B = m->batch_size;    /* Batch size (all layers)           */
    int D = m->input_dim;     /* Input dimension: may include bias */
    int Dx = D - (1 - m->add_bias); (void) Dx; /* Input dim excluding bias */
    int Db = D + m->add_bias; /* Input dimension including bias    */
    
    typedef float (*ArrMD)[D];
    typedef float (*ArrMNo)[No];
    ArrMD x = (ArrMD) x_;
    ArrMNo y = (ArrMNo) y_;
    
    typedef float (*VecDx);
    VecDx mean = (VecDx) m->mean;
    VecDx sdev = (VecDx) m->sdev;
    /* Allocate memory for one batch */
    typedef float (*ArrBDb)[Db];
    ArrBDb xb = (ArrBDb) allocmem(B,Db,float); /* Array of samples      */

    BATCH* b = batch_create(x,D,NULL,0,B,NULL,len,0,m->add_bias);
    reset_state(m);
    for (;;) {
        fArr2D yp[L]; /* Pointers to layers' prediction arrays */
        int cnt = batch_copy(b,xb,NULL);
        if (cnt == 0)
            break;
        if (m->normalize)
            normalize(xb,B,Db,mean,sdev,1); 
        model_batch_forward(m,xb,0,yp);
        if (m->loss_func == 'N') {
            /* Full-vocabulary pass, then normalize to a distribution. */
            negsample_logits(m->layer[L - 1].negsample,yp[L - 1],
                             (fArr2D) y,cnt);
            softmax((fArr2D) y,cnt,No);
        }
        else
            fltcpy((fArr2D) y,yp[L - 1],cnt * No);
        y += cnt;
    }
    freemem(xb);
    batch_free(b);
}

static void model_batch_forward(MODEL* m, fArr2D x, int training, fArr2D* yp)
{
    int L = m->num_layers;
    yp[0] = layer_forward(&m->layer[0],x,training,0);
    for (int j = 1; j < L; j++)
        yp[j] = layer_forward(&m->layer[j],yp[j - 1],training,j);
}

static void model_batch_backward(MODEL* m, fArr2D x, fArr2D* dy, fArr2D* yp)
{
    int L = m->num_layers;
    for (int j = L - 1; j > 0; j--)
        layer_backward(&m->layer[j],dy[j],yp[j - 1],dy[j - 1],j);
    layer_backward(&m->layer[0],dy[0],x,NULL,0);
}

/* Updates model weights */
static void model_update(MODEL* m, float learning_rate, float weight_decay)
{
    int uc = ++m->update_cnt;
    for (int j = 0; j < m->num_layers; j++)
        layer_update(&m->layer[j],m->optimizer,
                     learning_rate,weight_decay,uc);
}

/* Prints a text line with model training progress information. 
 * - epoch is a number between 1 and 99999
 * - nepochs is the highest value of epoch
 * - progress is a number between 0 and 100. Set to -1 to exclude.
 * - etime is elapsed time since start of training, in seconds. Set to 
 *   0 to exclude.
 * - loss, v_loss are floating point numbers, formatted with 5 digits 
 *   precision. Set to -1 to exclude.
 * - acc, v_acc are floating point numbers, formatted with 4 digits
 *   precision. Set to -1 to exclude.
 */
static void print_status(int epoch, int nepochs, int progress, float etime,
                         float loss, float acc, float v_loss, float v_acc)
{
    char status[78];
    int pos = 0;
    if (nepochs > 0) {
        int l = snprintf(NULL,0,"%d",nepochs);
        l = (l > 5) ? 5 : l;
        snprintf(status + pos,sizeof(status) - pos,"Epoch %*d ",l,epoch);
        pos = strlen(status);
    }
    if (loss != -1) {
        int l = 5;
        int f = l - (((int) log10f(loss + 0.999)) + 1);
        snprintf(status + pos,sizeof(status) - pos,"Tr loss %*.*f ",l,f,loss);
        pos = strlen(status);
    }
    if (acc != -1) {
        int l = 4;
        int f = l - (((int) log10f(acc + 0.999)) + 1);
        snprintf(status + pos,sizeof(status) - pos,"acc %*.*f ",l,f,acc);
        pos = strlen(status);
    }
    if (v_loss != -1) {
        int l = 5;
        int f = l - (((int) log10f(v_loss + 0.999)) + 1);
        snprintf(status + pos,sizeof(status) - pos,"Vd loss %*.*f ",l,f,v_loss);
        pos = strlen(status);
    }
    if (v_acc != -1) {
        int l = 4;
        int f = l - (((int) log10f(v_acc + 0.999)) + 1);
        snprintf(status + pos,sizeof(status) - pos,"acc %*.*f ",l,f,v_acc);
        pos = strlen(status);
    }
    if (progress >= 0 && progress < 100) {
        snprintf(status + pos,sizeof(status) - pos,"%3d%% ",progress);
        pos = strlen(status);
    }
    if (etime > 0) {
        snprintf(status + pos, sizeof(status) - pos,"%.f seconds",etime);
        pos = strlen(status);
    }
    while (pos < (int)(sizeof(status) - 1))
        status[pos++] = ' ';
    status[pos] = '\0';
    printf("\r%s",status);
    fflush(stdout);
}

static const char* find_kwarg(const char* kwargs, const char* key)
{
    if (kwargs == NULL || key == NULL)
        return NULL;
    char* p = strstr(kwargs,key);
    if (p == NULL || (p != kwargs && p[-1] != ' '))
        return NULL;
    p += strlen(key);
    while (*p == ' ') p++;
    if (*(p++) != '=')
        return NULL;
    while (*p == ' ') p++;
    return p;        
}

static void get_kw_int(const char* kwargs, const char* key, int* val)
{
    const char* sval = find_kwarg(kwargs,key);
    if (sval != NULL)
        *val = atoi(sval);
}

static void get_epoch_params(const char* sch, int epoch, float* lr, float* wd)
{
    
    int te = 0;
    while(sch != NULL) {
        int e; float l, w;
        int c = sscanf(sch,"%d" ":" FMTF ":" FMTF,&e,&l,&w);
        if (c == 0 || c == EOF)
            break;
        te += e;
        if (c >= 3) *wd = w;
        if (c >= 2) *lr = l;
        if (epoch < te)
            break;
        sch = index(sch,',');
        if (sch != NULL)
            if (*(++sch) == '\0')
                sch = NULL;
    }
}


