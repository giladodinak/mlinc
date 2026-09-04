/* Copyright (c) 2026 Gilad Odinak */

/* This program implements standalone decoder-only language-model trainer.
 *
 * The input and output layers are structured similarly to those in word2vec.c.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "mem.h"
#include "float.h"
#include "etime.h"
#include "blascpu.h"
#include "array.h"
#include "random.h"
#include "textfile.h"
#include "vocab.h"
#include "activation.h"
#include "lmemb.h"
#include "transformer.h"
#include "smsftmax.h"
#include "layer.h"
#include "lm.h"
#include "lmio.h"

static const char* usage =
"Usage: lmtrain [options]\n"
"Options:\n"
"  -h                   Show this help message, then exit\n"
"  -b <batch_size>      Sequences (windows) per batch (default 16)\n"
"  -T <seq_len>         Sequence length / context window T (default 128)\n"
"  -d <model_dim>       Model dimension D (default 128)\n"
"  -H <heads>           Attention heads (D must divide by heads) (default 8)\n"
"  -L <layers>          Number of transformer layers (default 4)\n"
"  -f <ffn_dim>         FFN hidden dim (default 4*D)\n"
"  -e <num_epochs>      Number of epochs (default 8)\n"
"  -F <sample_frac>     Fraction of files to train on each epoch (def. 0.2)\n"
"  -n <neg_samples>     Negative samples per position (default 40)\n"
"  -r <learning_rate>   Starting learning rate (default 3e-4)\n"
"  -i <train_file>      File list (default data/news/selected_files.lst)\n"
"  -o <output_file>     Output model file (default lmtrain.model)\n"
"  -l <model_file>      Load a saved model and continue training\n"
"  -s <base>            Base name for per-epoch saves <base>.<epoch>.model\n"
"                       (default lmtrain)\n"
"  --rate-decay=<f>     Learning rate decay per epoch (default 0.9)\n"
"  --weight-decay=<f>   Weight decay (default 0.01)\n"
"  --dropout-rate=<f>   Sub-layer dropout rate (default 0.1)\n"
"  --vocab-size=<n>     Limit vocabulary size (including PAD)\n"
"  --vocab-coverage=<f> Vocabulary coverage fraction (default 0.99)\n"
"  --data-dir=<dir>     Location of training data (default data/news/data)\n"
"  --prompt=\"<prompt>\"  Seed prompt for text generation\n"
"                       (default 'according to the')\n"
"  --validation-frac=<f>  Fraction of files held out for validation\n"
"                         (default 0.01)\n"
"  --print-vocab        Print vocabulary and exit\n"
"  --cores=<list>       Specify cores for openblas use (def. 0,2,4,6)\n"
;

static void shuffle_list(char** list, int cnt)
{
    if (cnt <= 1) return;
    for (int i = cnt - 1; i > 0; i--) {
        int j = (int) urand(0,i + 1);
        char* tmp = list[i]; list[i] = list[j]; list[j] = tmp;
    }
}

static LM* load_checkpoint(int argc, char** argv, char** load_file,
                           VOCAB** pvocab, LMPARAM* st,
                           char* optimizer, int* update_cnt,
                           float* learning_rate, float* lr_decay,
                           float* weight_decay, int* num_epochs,
                           int* start_epoch, float* sample_frac,
                           int* vocab_size, int* model_dim, int* heads,
                           int* seq_len, int* batch_size, int* layers,
                           int* ffn_dim, int* neg_samples, float* dropout)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"-l")) {
            if (i + 1 < argc) 
                *load_file = argv[i + 1];
            break;
        }
        if (!strncmp(argv[i],"-l",2) && argv[i][2] != '\0') {
            *load_file = argv[i] + 2;
            break;
        }
    }
    if (*load_file == NULL)
        return NULL;

    printf("Loading model from '%s' to continue training\n",*load_file);
    fflush(stdout);
    LM* m = load_lm(*load_file,pvocab,st);
    if (m == NULL) {
        fprintf(stderr,"Failed to load model from '%s'\n",*load_file);
        return NULL;
    }
    if (st->final) {
        fprintf(stderr,"Cannot continue from a final model '%s'\n",*load_file);
        lm_free(m);
        vocab_free(*pvocab);
        return NULL;
    }
    if ((*pvocab)->dist == NULL || (*pvocab)->dist_size <= 0) {
        fprintf(stderr,"Checkpoint '%s' has no sampling table\n",*load_file);
        lm_free(m);
        vocab_free(*pvocab);
        return NULL;
    }

    init_lrng(st->lrng_seed);
    *optimizer     = st->optimizer;
    *update_cnt    = st->update_cnt;
    *learning_rate = st->learning_rate;
    *lr_decay      = st->lr_decay;
    *weight_decay  = st->weight_decay;
    *num_epochs    = st->num_epochs;
    *start_epoch   = st->epoch + 1;
    *sample_frac   = st->sample_frac;

    *vocab_size = m->V;
    *model_dim = m->E;
    *seq_len = m->T;
    *batch_size = m->B;
    *layers = m->N;
    *neg_samples = m->n_neg;
    if (m->N > 0) {
        *heads = m->tr[0].transformer->mha->H;
        *ffn_dim = m->tr[0].transformer->Dff;
        *dropout = m->tr[0].transformer->dropout_rate;
    }
    printf("Resuming at epoch %d of %d, lr %g, seed %d\n",
           *start_epoch,*num_epochs,*learning_rate,st->lrng_seed);
    fflush(stdout);
    return m;
}

/* Prints one validation progress line.
 *
 * Parameters:
 *   correct   - running count of correct top-1 predictions so far
 *   positions - running count of scored positions so far
 *   nll       - running summed negative log-likelihood so far
 *   fi        - index of the file just processed (0-based)
 *   nfiles    - total number of validation files
 *   val_start - wall-clock time when validation began
 */
static void print_validation_status(long long correct, long long positions,
                                    double nll, int fi, int nfiles,
                                    double val_start)
{
    int sec = (int) (current_time() - val_start);
    double t1 = positions ?
                100.0 * (double) correct / (double) positions : 0.0;
    double pp = positions ? exp(nll / (double) positions) : 0.0;
    char buf[128]; /* Larger than needed, to pacify gcc */
    snprintf(buf,sizeof(buf),
        "Validating: top-1 %4.1f%% perplexity %8.2f (%d/%d files) %d:%02d:%02d",
        t1,pp,fi + 1,nfiles,sec/3600,(sec/60)%60,sec%60);
    printf("\r%-79s\r",buf);
    fflush(stdout);
}

/* Runs a full-vocabulary validation pass over the given files.
 * For each valid position it computes logits over the whole vocabulary,
 * takes the argmax for true top-1 accuracy, and accumulates cross-entropy
 * (via log-softmax at the true target) for perplexity. Forward pass only;
 * no gradients, no weight updates. Returns via out_top1 and out_ppl.
 */
void validate(LM* m, const VOCAB* vocab, char** files, int nfiles,
              const char* data_dir, int vocab_size,
              int* file_words, int max_file_words,
              double* out_top1, double* out_ppl)
{
    const int BT = m->BT;
    const int seq_len = m->T;
    const int batch_size = m->B;
    const int K = m->V;

    fArr2D logits = allocmem(BT,K,float); /* Full-vocab scores */
    long long correct = 0;
    long long positions = 0;
    double nll = 0.0; /* Summed negative log-likelihood */

    double val_start = current_time();
    double last_report = val_start;

    for (int fi = 0; fi < nfiles; fi++) {
        int fwcnt = process_text_file(files[fi],data_dir,
                  vocab->hmap,0,vocab_size,NULL,file_words,max_file_words);
        if (fwcnt <= 1)
            continue;

        int stride = seq_len;
        int pos = 0;
        while (pos < fwcnt - 1) {
            for (int i = 0; i < BT; i++) {
                m->ids[i] = 0;
                m->pad_mask[i] = 0;
                ((fVec) m->labels)[i] = 0.0;
            }
            int filled = 0;
            for (int b = 0; b < batch_size; b++) {
                int base = pos + b * stride;
                if (base >= fwcnt - 1) break;
                for (int t = 0; t < seq_len; t++) {
                    int src = base + t;
                    if (src >= fwcnt - 1)
                        break;
                    int row = b * seq_len + t;
                    if (file_words[src] > 0) {
                        m->ids[row] = file_words[src];
                        m->pad_mask[row] = 1;
                        ((fVec) m->labels)[row] = (float) file_words[src + 1];
                        filled++;
                    }
                }
            }
            if (filled == 0)
                break;
            pos += batch_size * stride;

            fArr2D hhead = lm_forward(m,0);

            /* Full-vocabulary logits for every row in the batch */
            smsftmax_logits(m->head,hhead,logits,BT);

            typedef float (*ArrBK)[K];
            ArrBK lg = (ArrBK) logits;
            const float* labels = (const float*) m->labels;

            for (int row = 0; row < BT; row++) {
                int target = (int) labels[row];
                if (target <= 0 || target >= K)
                    continue; /* PAD / OOV target: skip */

                /* argmax and log-sum-exp over the full vocabulary */
                int best = 1;
                float maxl = lg[row][1];
                for (int c = 2; c < K; c++) {
                    if (lg[row][c] > maxl) {
                        maxl = lg[row][c];
                        best = c;
                    }
                }
                float sum = 0.0;
                for (int c = 1; c < K; c++)
                    sum += expf(lg[row][c] - maxl);
                float logp = (lg[row][target] - maxl) - logf(sum);

                nll -= (double) logp;
                if (best == target)
                    correct++;
                positions++;
            }
            if (current_time() - last_report >= 1.0) {
                last_report = current_time();
                print_validation_status(correct,positions,nll,fi,nfiles,val_start);
            }
        }
    }
    print_validation_status(correct,positions,nll,nfiles - 1,nfiles,val_start);
    printf("\n");
    freemem(logits);

    *out_top1 = positions ? 100.0 * (double) correct / (double) positions : 0.0;
    *out_ppl  = positions ? exp(nll / (double) positions) : 0.0;
}



/* Prints one training progress line.
 *
 * Parameters:
 *   epoch         - current epoch number
 *   learning_rate - current learning rate
 *   ep_loss       - running summed loss this epoch
 *   ep_positions  - running count of scored positions this epoch
 *   fi            - index of the file just processed (0-based)
 *   act_num_files - total training files this epoch
 *   start_time    - wall-clock time when the epoch began
 */
static void print_training_status(int epoch, float learning_rate,
                                  double ep_loss, long long ep_positions,
                                  int fi, int act_num_files,
                                  double start_time)
{
    int sec = (int) elapsed_time(start_time);
    char buf[128]; /* Larger than needed, to pacify gcc */
    snprintf(buf,sizeof(buf),
        "Epoch %2d lr %8.6f loss %7.4f (file %d/%d) %d:%02d:%02d",
        epoch,learning_rate,
        ep_positions ? ep_loss/ep_positions : 0.0,
        fi + 1,act_num_files,sec/3600,(sec/60)%60,sec%60);
    printf("\r%-79s\r",buf);
    fflush(stdout);
}

int main(int argc, char** argv)
{
    int   batch_size     = 16;
    int   seq_len        = 128;
    int   model_dim      = 128;
    int   heads          = 8;
    int   layers         = 4;
    int   ffn_dim        = 0; /* 0 -> 4*model_dim */
    int   num_epochs     = 8;
    float sample_frac    = 0.2;
    float validation_frac = 0.01;
    int   neg_samples    = 40;
    float learning_rate  = 3e-4;

    char* tr_file        = "data/news/selected_files.lst";
    char* output_file    = "lmtrain.model";
    char* load_file      = NULL;
    char* save_base      = "lmtrain";

    float lr_decay       = 0.9;
    float weight_decay   = 0.01;
    float dropout        = 0.1;
    int   vocab_size     = 0; /* 0 -> from coverage */
    float vocab_coverage = 0.99;
    char* data_dir       = "data/news/data";
    char* prompt         = "according to the";
    int   print_vocab    = 0;
    char* blas_cores     = "0,2,4,6";

    const int max_vocab  =    10000000; /* Can be represented in a float */
    const int max_file_words = 1000000;
    char optimizer       = 'a'; /* AdamW for the transformer stack */

    VOCAB* vocab = NULL;
    LM* m = NULL;
    LMPARAM st;
    int update_cnt = 0;
    int start_epoch = 1;

    /* If '-l' option specified load saved training checkpoint.
     * Note that load_checkpoint handles the -l command line argument
     * and updates load_file with a pointer to the file name if specified.
     */
    m = load_checkpoint(argc,argv,&load_file,&vocab,&st,
                        &optimizer,&update_cnt,&learning_rate,&lr_decay,
                        &weight_decay,&num_epochs,&start_epoch,&sample_frac,
                        &vocab_size,&model_dim,&heads,&seq_len,&batch_size,
                        &layers,&ffn_dim,&neg_samples,&dropout);
    if (load_file != NULL && m == NULL)
        return -1;

    int opt;
    while ((opt = getopt(argc,argv,"b:T:d:H:L:f:e:F:n:r:i:o:l:s:-:h")) != -1) {
        switch (opt) {
            case 'b': if (m == NULL) batch_size = atoi(optarg); break;
            case 'T': if (m == NULL) seq_len = atoi(optarg); break;
            case 'd': if (m == NULL) model_dim = atoi(optarg); break;
            case 'H': if (m == NULL) heads = atoi(optarg); break;
            case 'L': if (m == NULL) layers = atoi(optarg); break;
            case 'f': if (m == NULL) ffn_dim = atoi(optarg); break;
            case 'e': num_epochs  = atoi(optarg); break;
            case 'F': sample_frac = atof(optarg); break;
            case 'n': if (m == NULL) neg_samples = atoi(optarg); break;
            case 'r': learning_rate = atof(optarg); break;
            case 'i': tr_file     = optarg; break;
            case 'o': output_file = optarg; break;
            case 'l': load_file   = optarg; break;
            case 's': save_base   = optarg; break;
            case '-':
                if      (!strncmp(optarg,"rate-decay=",11))      lr_decay = atof(optarg + 11);
                else if (!strncmp(optarg,"weight-decay=",13))    weight_decay = atof(optarg + 13);
                else if (!strncmp(optarg,"dropout-rate=",13))    { if (m == NULL) dropout = atof(optarg + 13); }
                else if (!strncmp(optarg,"vocab-size=",11))      { if (m == NULL) vocab_size = atoi(optarg + 11); }
                else if (!strncmp(optarg,"vocab-coverage=",15))  { if (m == NULL) vocab_coverage = atof(optarg + 15); }
                else if (!strncmp(optarg,"data-dir=",9))         data_dir = optarg + 9;
                else if (!strncmp(optarg,"prompt=",7))           prompt = optarg + 7;
                else if (!strncmp(optarg,"validation-frac=",16)) validation_frac = atof(optarg + 16);
                else if (!strncmp(optarg,"print-vocab",11))      print_vocab = 1;
                else if (!strncmp(optarg,"cores=",6))            blas_cores = (optarg + 6);
                else goto opterr;
            break;
            case 'h': printf("%s",usage); exit(0);
            default:
            opterr:
                if (opt != '-')
                    fprintf(stderr,"lmtrain: Invalid option: -- '%s'\n",optarg);
                fprintf(stderr,"%s",usage);
                return -1;
        }
    }

    if (model_dim % heads != 0) {
        fprintf(stderr,"lmtrain: model_dim %d must be divisible by heads %d\n",
                model_dim,heads);
        return -1;
    }
    if (ffn_dim <= 0)
        ffn_dim = 4 * model_dim;

    if (sample_frac <= 0 || sample_frac > 1.0) {
        fprintf(stderr,"lmtrain: invalid sample_frac %g\n",sample_frac);
        return -1;
    }

    if (validation_frac < 0 || validation_frac >= 1.0) {
        fprintf(stderr,"lmtrain: invalid validation_frac %g\n",validation_frac);
        return -1;
    }

    if (neg_samples < 1) {
        fprintf(stderr,"lmtrain: invalid neg_samples %d\n",neg_samples);
        return -1;
    }

    if (m != NULL && start_epoch > num_epochs)
        printf("Model already trained %d epochs; nothing to do.\n",num_epochs);

    char *p = blas_cores;
    int cores[256];
    int core_cnt = 0;
    while (*p != '\0' && core_cnt < (int)(sizeof(cores)/sizeof(cores[0]))) {
        cores[core_cnt++] = strtol(p, &p, 10);
        if (*p == ',') p++;
    }
    openblas_use_cpus(cores,core_cnt);
    
    float initial_lr = learning_rate;

    printf("\nDecoder-only LM trainer\n");
    printf("D=%d heads=%d layers=%d ffn=%d T=%d batch=%d\n",
           model_dim,heads,layers,ffn_dim,seq_len,batch_size);
    printf("epochs=%d lr=%g rd=%g wd=%g dropout=%g neg=%d\n",
           num_epochs,initial_lr,lr_decay,weight_decay,dropout,neg_samples);
    fflush(stdout);
    
    int num_files = 0;
    char** file_list = read_text_file_list(tr_file,data_dir,&num_files);
    if (file_list == NULL || num_files == 0) {
        fprintf(stderr,"Failed to read data files list from '%s'\n",tr_file);
        return -1;
    }

    shuffle_list(file_list,num_files);
    
    int num_valid = (int) (num_files * validation_frac);
    int num_train = num_files - num_valid;
    char** valid_list = file_list + num_train;
    if (num_train < 1) {
        fprintf(stderr,"lmtrain: validation_frac %g leaves no training files\n",
                validation_frac);
        return -1;
    }
    printf("Split: %d training files, %d validation files\n",
           num_train,num_valid);
    fflush(stdout);

    if (load_file == NULL) {
        printf("Creating vocabulary from dataset\n");
        fflush(stdout);
        vocab = vocab_build(file_list,num_files,data_dir,
                            vocab_size,vocab_coverage,max_vocab,1,1,1);
        if (vocab == NULL) {
            fprintf(stderr,"Failed to create vocabulary\n");
            free_text_file_list(file_list,num_files);
            return -1;
        }
        vocab_size = vocab->size;
        vocab_coverage = vocab->coverage;

        if (print_vocab) {
            printf("index frequency word\n");
            for (int i = 0; i < vocab_size; i++)
                printf("%5d %9.7f %-16s\n",
                       i,vocab_freq(vocab,i),vocab_word(vocab,i));
            vocab_free(vocab);
            free_text_file_list(file_list,num_files);
            return 0;
        }

        m = lm_create(vocab_size,model_dim,heads,seq_len,batch_size,
                      layers,ffn_dim,neg_samples,dropout,optimizer);
        smsftmax_set_dist(m->head,vocab->dist,vocab->dist_size);

        st.optimizer    = optimizer;
        st.lr_decay     = lr_decay;
        st.weight_decay = weight_decay;
        st.num_epochs   = num_epochs;
    }

    int* file_words = allocmem(1,max_file_words,int);
    const int BT = m->BT;

    char prompt_buf[256];
    int max_prompt_tokens = 8;
    int prompt_tokens[max_prompt_tokens];
    int prompt_token_count = 0;
    snprintf(prompt_buf,sizeof(prompt_buf),"%s",prompt);
    char *token = strtok(prompt_buf," ");
    while (token != NULL && prompt_token_count < max_prompt_tokens) {
        char *p = token;
        for (; *p != '\0'; p++)
            *p = tolower((unsigned char) *p);
        int id = vocab_lookup(vocab,token);
        if (id > 0)
            prompt_tokens[prompt_token_count++] = id;
        else {
            fprintf(stderr,"'%s' is not in the vocabulary - exiting\n",token);
            lm_free(m);
            vocab_free(vocab);
            free_text_file_list(file_list,num_files);
            return 1;
        }            
        token = strtok(NULL," ");
    }
    printf("Training (sampling %g of %d training files each epoch)\n",
           sample_frac,num_train);
    printf("\n\n");
    fflush(stdout);
    double start_time = current_time();

    int epoch;
    for (epoch = start_epoch; epoch <= num_epochs; epoch++) {
        shuffle_list(file_list,num_train);
        float ep_loss = 0;
        long long ep_positions = 0;
        long long ep_correct = 0;
        /* Use a fraction of the dataset each epoch. Inspired by
         * RS2: Okanovic et al. https://arxiv.org/pdf/2305.18424
         */
        int act_num_files = (int) (num_train * sample_frac);
        if (act_num_files < 1) 
            act_num_files = 1;
        double last_report = current_time();
        for (int fi = 0; fi < act_num_files; fi++) {
            int fwcnt = process_text_file(file_list[fi],data_dir,
                     vocab->hmap,0,vocab_size,NULL,file_words,max_file_words);
            if (fwcnt <= 1)
                continue;

            /* Slide over the token stream in windows of T, B windows/batch.
             * Each batch fills ids[BT], labels[BT] (next token), pad_mask[BT].
             * Windows advance by T (non-overlapping) so every token is a
             * target once per epoch.
             */
            int stride = seq_len;
            int pos = 0;
            while (pos < fwcnt - 1) {
                for (int i = 0; i < BT; i++) {
                    m->ids[i] = 0;
                    m->pad_mask[i] = 0;
                    ((float*) m->labels)[i] = 0.0;
                }
                int filled = 0;
                for (int b = 0; b < batch_size; b++) {
                    int base = pos + b * stride;
                    if (base >= fwcnt - 1) break;
                    for (int t = 0; t < seq_len; t++) {
                        int src = base + t;
                        if (src >= fwcnt - 1)
                            break; /* Need src+1 target */
                        int row = b * seq_len + t;
                        /* Skip OOV source positions and count only valid
                         * training targets. No <unk> token exists; OOV
                         * is represented as PAD.
                         */
                        if (file_words[src] > 0) {
                            m->ids[row] = file_words[src];
                            m->pad_mask[row] = 1;
                            ((float*) m->labels)[row] = (float) file_words[src+1];
                            filled++;
                        }
                    }
                }
                if (filled == 0)
                    break;
                pos += batch_size * stride;

                fArr2D hhead = lm_forward(m,1);

                /* Loss + dh into m->dtop, sparse gWo into gHead[0]. */
                int correct = 0;
                float loss = smsftmax_loss(m->head,hhead,m->labels,
                                           m->gHead[0],m->dtop,BT,&correct);

                lm_backward(m);

                update_cnt++;
                lm_update(m,optimizer,learning_rate,weight_decay,update_cnt);

                ep_loss += loss; ep_positions += filled; ep_correct += correct;

                if (current_time() - last_report >= 1.0) {
                    last_report = current_time();
                    print_training_status(epoch,learning_rate,
                                          ep_loss,ep_positions,
                                          fi,act_num_files,start_time);
                }
            }
        }
        print_training_status(epoch,learning_rate,
                              ep_loss,ep_positions,
                              act_num_files - 1,act_num_files,start_time);
        printf("\n");
        if (num_valid > 0) {
            printf("Validating...");
            fflush(stdout);
            double val_top1 = 0.0, val_ppl = 0.0;
            validate(m,vocab,valid_list,num_valid,data_dir,vocab_size,
                     file_words,max_file_words,&val_top1,&val_ppl);
        }

        char output[1024];
        lm_generate(m,vocab->hmap,prompt_tokens,prompt_token_count,
                    20,output,sizeof(output),0.8,40,32,3.7);
        printf("%s\n",output);
        fflush(stdout);

        learning_rate *= lr_decay;
        {
            st.optimizer     = optimizer;
            st.update_cnt    = update_cnt;
            st.epoch         = epoch; /* Number of epochs completed */
            st.num_epochs    = num_epochs;
            st.sample_frac   = sample_frac;
            st.learning_rate = learning_rate;
            st.lr_decay      = lr_decay;
            st.weight_decay  = weight_decay;
            st.lrng_seed     = get_lrng_seed();
            char fname[1024];
            snprintf(fname,sizeof fname,"%s.%d.model",save_base,epoch);
            if (store_lm(fname,m,0,vocab,&st))
                printf("Saved checkpoint '%s'\n",fname);
            else
                fprintf(stderr,"Failed to save checkpoint '%s'\n",fname);
            fflush(stdout);
        }
    }
    {
        st.optimizer     = optimizer;
        st.update_cnt    = update_cnt;
        st.epoch         = epoch - 1; /* Number of epochs completed */
        st.num_epochs    = num_epochs;
        st.sample_frac   = sample_frac;
        st.learning_rate = learning_rate;
        st.lr_decay      = lr_decay;
        st.weight_decay  = weight_decay;
        st.lrng_seed     = get_lrng_seed();
        if (store_lm(output_file,m,0,vocab,&st))
            printf("Saved final model '%s'\n",output_file);
        else
            fprintf(stderr,"Failed to save final model '%s'\n",output_file);
    }
    printf("\nTraining complete\n");

    lm_free(m);
    vocab_free(vocab);
    freemem(file_words);
    free_text_file_list(file_list,num_files);
    return 0;
}
