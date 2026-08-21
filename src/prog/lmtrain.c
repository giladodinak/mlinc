/* Copyright (c) 2026 */
/* lmtrain - standalone decoder-only language-model trainer.
 *
 * Architecture:
 *
 *   ids[B*T] --> lmemb --> [ transformer x N (causal) ] --> negsample_loss
 *                  ^                                              |
 *                  |                                              | dh
 *                  +--------- lmemb_backward <-- transformer_backward x N
 *
 * The output head is negative sampling (Mikolov et al., 2013), chosen because
 * the vocabulary is large (~200K): a full softmax would cost O(V) per token
 * for logits and a dense O(V*E) weight update every step, whereas negative
 * sampling costs O((1+n_neg)*E) and updates only the sampled rows.
 *
 * The text front-end (vocabulary construction, frequency sorting, PAD at
 * index 0, and the unigram distribution table) mirrors word2vec.c, reusing
 * process_news_file() / read_news_file_list() from newsfile.c.
 *
 * This trainer drives the layer forward/backward inline functions directly
 * rather than going through MODEL/model_fit(), because the token-embedding
 * layer (lmemb) is not a MODEL layer type and because the embedding backward
 * needs the input-gradient produced by the transformer stack.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "mem.h"
#include "float.h"
#include "etime.h"
#include "blascpu.h"
#include "array.h"
#include "random.h"
#include "hash.h"
#include "newsfile.h"
#include "activation.h"
#include "lmemb.h"
#include "transformer.h"
#include "negsample.h"
#include "layer.h"
#include "lm.h"
#include "lmio.h"

static const char* usage =
"Usage: lmtrain [options]\n"
"Options:\n"
"  -h                   Show this help message, then exit\n"
"  -b <batch_size>      Sequences (windows) per batch (default 16)\n"
"  -T <seq_len>         Sequence length / context window T (default 64)\n"
"  -d <model_dim>       Model dimension D (default 256)\n"
"  -H <heads>           Attention heads (D must divide by heads) (default 8)\n"
"  -L <layers>          Number of transformer layers (default 4)\n"
"  -f <ffn_dim>         FFN hidden dim (default 4*D)\n"
"  -e <num_epochs>      Number of epochs (default 5)\n"
"  -n <neg_samples>     Negative samples per position (default 10)\n"
"  -r <learning_rate>   Starting learning rate (default 3e-4)\n"
"  -i <train_file>      File list (default data/news/all_files.lst)\n"
"  -o <output_file>     Output model file (default lmtrain.model)\n"
"  -s <base>            Base name for per-epoch saves <base>.<epoch>.model\n"
"                       (default lmtrain)\n"
"  -l <model_file>      Load a saved model and continue training\n"
"  -p <dropout>         Sub-layer dropout rate (default 0.1)\n"
"  --rate-decay=<f>     Learning rate decay per epoch (default 0.9)\n"
"  --weight-decay=<f>   Weight decay (default 0.01)\n"
"  --vocab-size=<n>     Limit vocabulary size (including PAD)\n"
"  --vocab-coverage=<f> Vocabulary coverage fraction (default 0.99)\n"
"  --data-dir=<dir>     Location of training data (default data/news/data)\n"
"  --print-vocab        Print vocabulary and exit\n"
"  --gen-every=<n>      Sample a generation every n epochs (0=off, def. 1)\n"
"  --cores=<list>       Specify cores for openblas use (def. 0,2,4,6)\n"
;

/* ------------------------------------------------------------------ */
/* Vocabulary + distribution table (mirrors word2vec.c)               */
/* ------------------------------------------------------------------ */

static int qsort_compare_word_freq(const void* a, const void* b)
{   /* WRDFRQ declared in newsfile.h */
    if (((WRDFRQ*)b)->cnt > ((WRDFRQ*)a)->cnt) return 1;
    if (((WRDFRQ*)b)->cnt < ((WRDFRQ*)a)->cnt) return -1;
    return 0;
}

static void shuffle_list(char** list, int cnt)
{
    if (cnt <= 1) return;
    for (int i = cnt - 1; i > 0; i--) {
        int j = (int) urand(0,i + 1);
        char* tmp = list[i]; list[i] = list[j]; list[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* The model: an embedding, a stack of transformer LAYERs, a head.    */
/* ------------------------------------------------------------------ */

/* The LM struct now lives in lm.h so the serializer (lmio.c) can see it. */

static LM* lm_create(int vocab, int model_dim, int heads, int seq_len,
                     int batch, int layers, int ffn_dim, int n_neg,
                     float dropout, char optimizer)
{
    LM* m = allocmem(1,1,LM);
    m->V = vocab; m->E = model_dim; m->T = seq_len; m->B = batch;
    m->N = layers; m->BT = batch * seq_len; m->n_neg = n_neg;

    /* Embedding: owns its weights (untied), training mode. */
    m->emb = lmemb_create(model_dim,seq_len,/*padinx=*/0);
    lmemb_init(m->emb,vocab,batch,/*training=*/1,/*tied=*/0);

    /* Transformer stack, each wrapped in a LAYER for grad/optim mgmt. */
    m->tr = allocmem(layers,1,LAYER);
    for (int i = 0; i < layers; i++) {
        TRANSFORMER* t = transformer_create(heads,seq_len,model_dim,
                                            ffn_dim,/*lookahead=*/0);
        transformer_init(t,batch,/*training=*/1,dropout);
        m->tr[i].type = 't';
        m->tr[i].transformer = t;
        layer_alloc_grads(&m->tr[i],optimizer);
    }

    /* Head. */
    m->head = negsample_create(vocab,n_neg);
    negsample_init(m->head,model_dim,m->BT);   /* head batch = B*T rows */
    m->gHead = allocmem(1,1,fArr2D*);
    m->gHead[0] = allocmem(vocab,model_dim,float);

    /* Buffers. */
    m->acts = allocmem(layers + 1,1,fArr2D*);
    for (int i = 0; i <= layers; i++)
        m->acts[i] = allocmem(m->BT,model_dim,float);
    m->dtop  = allocmem(m->BT,model_dim,float);
    m->dcur  = allocmem(m->BT,model_dim,float);
    m->dnext = allocmem(m->BT,model_dim,float);

    m->pad_mask = allocmem(m->BT,1,int);
    m->labels   = allocmem(m->BT,1,float);
    m->ids      = allocmem(m->BT,1,int);
    return m;
}

static void lm_free(LM* m)
{
    lmemb_free(m->emb);
    for (int i = 0; i < m->N; i++) {
        transformer_free(m->tr[i].transformer);
        for (int j = 0; j < m->tr[i].num_grads; j++)
            freemem(m->tr[i].grads[j]);
        freemem(m->tr[i].grads);
    }
    freemem(m->tr);
    negsample_free(m->head);
    freemem(m->gHead[0]);
    freemem(m->gHead);
    for (int i = 0; i <= m->N; i++)
        freemem(m->acts[i]);
    freemem(m->acts);
    freemem(m->dtop);
    freemem(m->dcur);
    freemem(m->dnext);
    freemem(m->pad_mask);
    freemem(m->labels);
    freemem(m->ids);
    freemem(m);
}

/* Forward the whole stack. Returns head-input activations [BT][E]. */
static fArr2D lm_forward(LM* m)
{
    /* Embedding gather: acts[0] = emb(ids). lmemb_forward writes into its
     * own l->h; copy into acts[0] so the stack owns a stable buffer. */
    fArr2D h = lmemb_forward(m->emb,m->ids,0);
    fltcpy(m->acts[0],h,m->BT * m->E);

    for (int i = 0; i < m->N; i++)
        transformer_forward(m->tr[i].transformer,
                            m->acts[i],m->pad_mask,m->acts[i+1],i);
    return m->acts[m->N];
}

/* Backward the whole stack given dh at the head input (in m->dtop). */
static void lm_backward(LM* m)
{
    /* Walk transformers in reverse. dcur holds grad w.r.t. layer i+1's
     * output; dnext receives grad w.r.t. layer i's input. */
    fltcpy(m->dcur,m->dtop,m->BT * m->E);
    for (int i = m->N - 1; i >= 0; i--) {
        transformer_backward(m->tr[i].transformer,
                             m->dcur,m->acts[i],m->dnext,i);
        /* swap dcur <-> dnext for next (lower) layer */
        fArr2D tmp = m->dcur; m->dcur = m->dnext; m->dnext = tmp;
    }
    /* dcur now holds grad w.r.t. embedding output: scatter into gWx. */
    lmemb_backward(m->emb,m->dcur,0);
}

/* One optimizer step across all trainable layers. */
static void lm_update(LM* m, char optimizer, float lr, float wd,
                      int update_cnt)
{
    /* Transformers via the LAYER machinery. */
    for (int i = 0; i < m->N; i++)
        layer_update(&m->tr[i],optimizer,lr,wd,update_cnt);

    /* Head: sparse SGD over touched rows (its own scheme). */
    negsample_update(m->head,m->gHead[0],lr,wd);

    /* Embedding: sparse row update over rows touched this batch. Weight
     * decay omitted for embeddings (standard practice). */
    {
        typedef float (*ArrVE)[m->E];
        ArrVE Wx  = (ArrVE) m->emb->Wx;
        ArrVE gWx = (ArrVE) m->emb->gWx;
        for (int r = 0; r < m->emb->ntouched; r++) {
            int row = m->emb->touched[r];
            for (int j = 0; j < m->E; j++)
                Wx[row][j] -= lr * gWx[row][j];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Generation (greedy / sampled) for a quick qualitative check.       */
/* Runs a single sequence (B-effective = 1 row of the batch buffers). */
/* ------------------------------------------------------------------ */

static int sample_from_logits(float* logits, int K, float temperature)
{
    if (temperature <= 0) { /* greedy */
        int best = 1; float bv = logits[1];
        for (int k = 2; k < K; k++) if (logits[k] > bv) { bv = logits[k]; best = k; }
        return best;
    }
    for (int k = 0; k < K; k++) logits[k] /= temperature;
    softmax((fArr2D) logits,1,K);
    float r = (float) urand(0,1.0), c = 0;
    for (int k = 1; k < K; k++) { c += logits[k]; if (r <= c) return k; }
    return K - 1;
}

/* Generate `steps` tokens continuing from a seed of ids (length seedlen),
 * printing words via the hashmap. Uses the model at its training batch
 * shape but only fills row 0..T-1 of one sequence (b=0). */
static void lm_generate(LM* m, HASHMAP* hmap, const int* seed, int seedlen,
                        int steps, float temperature)
{
    int T = m->T, K = m->V, E = m->E;
    int* ctx = allocmem(T,1,int);
    float* logits = allocmem(1,K,float);

    int n = seedlen < T ? seedlen : T;
    for (int i = 0; i < n; i++) ctx[i] = seed[i];

    printf("  gen: ");
    for (int i = 0; i < n; i++)
        printf("%s ",hashmap_inx2str(hmap,ctx[i]));

    for (int s = 0; s < steps; s++) {
        /* Build a full batch input but we only read position (n-1) of b=0. */
        for (int i = 0; i < m->BT; i++) m->ids[i] = 0;
        for (int i = 0; i < m->BT; i++) m->pad_mask[i] = 0;
        for (int i = 0; i < n; i++) { m->ids[i] = ctx[i]; m->pad_mask[i] = 1; }

        fArr2D hb = lm_forward(m);           /* [BT][E] */
        typedef float (*ArrBTE)[E];
        ArrBTE H = (ArrBTE) hb;

        negsample_logits(m->head,(fArr2D) H[n-1],(fArr2D) logits,1);
        int nxt = sample_from_logits(logits,K,temperature);

        printf("%s ",hashmap_inx2str(hmap,nxt));
        if (n < T) ctx[n++] = nxt;
        else { for (int i = 1; i < T; i++) ctx[i-1] = ctx[i]; ctx[T-1] = nxt; }
    }
    printf("\n");
    freemem(ctx);
    freemem(logits);
}

int main(int argc, char** argv)
{
    char* data_dir       = "data/news/data";
    char* tr_file        = "data/news/all_files.lst";
    char* output_file    = "lmtrain.model";
    char* save_base      = "lmtrain";
    char* load_file      = NULL;
    int   batch_size     = 16;
    int   seq_len        = 64;
    int   model_dim      = 256;
    int   heads          = 8;
    int   layers         = 4;
    int   ffn_dim        = 0;      /* 0 -> 4*model_dim */
    int   num_epochs     = 5;
    int   neg_samples    = 10;
    float learning_rate  = 3e-4;
    float lr_decay       = 0.9;
    float weight_decay   = 0.01;
    float dropout        = 0.1;
    int   vocab_size     = 0;      /* 0 -> from coverage */
    float vocab_coverage = 0.99;
    int   print_vocab    = 0;
    int   gen_every      = 1;
    char* blas_cores     = "0,2,4,6";

    const int max_vocab      = 30000000;
    const int hash_mem       = 100000000;
    const int max_file_words = 10000000;
    char optimizer           = 'a'; /* AdamW for the transformer stack */

    int opt;
    while ((opt = getopt(argc,argv,"b:T:d:H:L:f:e:n:r:i:o:p:s:l:-:h")) != -1) {
        switch (opt) {
            case 'h': printf("%s",usage); exit(0);
            case 'b': batch_size  = atoi(optarg); break;
            case 'T': seq_len     = atoi(optarg); break;
            case 'd': model_dim   = atoi(optarg); break;
            case 'H': heads       = atoi(optarg); break;
            case 'L': layers      = atoi(optarg); break;
            case 'f': ffn_dim     = atoi(optarg); break;
            case 'e': num_epochs  = atoi(optarg); break;
            case 'n': neg_samples = atoi(optarg); break;
            case 'r': learning_rate = atof(optarg); break;
            case 'i': tr_file     = optarg; break;
            case 'o': output_file = optarg; break;
            case 's': save_base   = optarg; break;
            case 'l': load_file   = optarg; break;
            case 'p': dropout     = atof(optarg); break;
            case '-':
                if      (!strncmp(optarg,"rate-decay=",3))     lr_decay = atof(optarg+3);
                else if (!strncmp(optarg,"weight-decay=",3))   weight_decay = atof(optarg+3);
                else if (!strncmp(optarg,"vocab-size=",11))    vocab_size = atoi(optarg+11);
                else if (!strncmp(optarg,"vocab-coverage=",15))vocab_coverage = atof(optarg+15);
                else if (!strncmp(optarg,"data-dir=",9))       data_dir = optarg+9;
                else if (!strncmp(optarg,"print-vocab",11))    print_vocab = 1;
                else if (!strncmp(optarg,"gen-every=",10))     gen_every = atoi(optarg+10);
                else if (!strncmp(optarg,"cores=",6)) blas_cores = (optarg+6);
                else goto opterr;
            break;
            default: opterr:
                fprintf(stderr,"lmtrain: syntax error\n");
                printf("%s",usage);
                exit(-1);
        }
    }

    if (model_dim % heads != 0) {
        fprintf(stderr,"lmtrain: model_dim %d must be divisible by heads %d\n",
                model_dim,heads);
        exit(-1);
    }
    if (ffn_dim <= 0)
        ffn_dim = 4 * model_dim;

    char *p = blas_cores;
    int cores[256];
    int core_cnt = 0;
    while (*p != '\0' && core_cnt < (int)(sizeof(cores)/sizeof(cores[0]))) {
        cores[core_cnt++] = strtol(p, &p, 10);
        if (*p == ',') p++;
    }
    openblas_use_cpus(cores,core_cnt);
    
    float initial_lr = learning_rate;

    printf("\nDecoder-only LM trainer (negative-sampling head)\n");
    printf("D=%d heads=%d layers=%d ffn=%d T=%d batch=%d\n",
           model_dim,heads,layers,ffn_dim,seq_len,batch_size);
    printf("epochs=%d lr=%g rd=%g wd=%g dropout=%g neg=%d\n",
           num_epochs,initial_lr,lr_decay,weight_decay,dropout,neg_samples);
    fflush(stdout);

    /* State shared by the fresh-build and resume paths. */
    HASHMAP* hmap = NULL;
    LM* m = NULL;
    int* dist = NULL;
    int dist_size = 0;
    WRDFRQ* word_freq = NULL;   /* fresh-build path only */
    LMTRAIN st;
    int update_cnt = 0;
    int start_epoch = 1;

    /* The training file list is needed whether building fresh or resuming. */
    int num_files = 0;
    char** file_list = read_news_file_list(tr_file,data_dir,&num_files);
    if (file_list == NULL || num_files == 0) {
        fprintf(stderr,"Failed to read data files list from '%s'\n",tr_file);
        return -1;
    }

    if (load_file != NULL) {
        /* Resume from a saved model */
        printf("Loading model from '%s' to continue training\n",load_file);
        fflush(stdout);
        m = load_lm(load_file,&hmap,&st);
        if (m == NULL) {
            fprintf(stderr,"Failed to load model from '%s'\n",load_file);
            free_news_file_list(file_list,num_files);
            return -1;
        }
        /* Restore the exact schedule and generator state,
         * so the run continues as if it had never been stopped
         */
        init_lrng(st.lrng_seed);
        optimizer     = st.optimizer;
        update_cnt    = st.update_cnt;
        learning_rate = st.learning_rate;
        lr_decay      = st.lr_decay;
        weight_decay  = st.weight_decay;
        num_epochs    = st.num_epochs;
        start_epoch   = st.epoch + 1;
        /* Architecture is fixed by the model; the sampling table (owned by
         * the caller) is attached to the head by load_lm(). */
        vocab_size = m->V; model_dim = m->E; seq_len = m->T; batch_size = m->B;
        dist = m->head->dist; dist_size = m->head->dist_size;
        printf("Resuming at epoch %d of %d, lr %g, seed %d\n",
               start_epoch,num_epochs,learning_rate,st.lrng_seed);
        if (start_epoch > num_epochs)
            printf("Model already trained %d epochs; nothing to do.\n",
                   num_epochs);
        fflush(stdout);
    } else {
    printf("Creating vocabulary from dataset\n"); fflush(stdout);
    hmap = hashmap_create(max_vocab,hash_mem);
    hashmap_str2inx(hmap,"",1);           /* index 0 = PAD */
    int tot_file_cnt = 0;       /* Total number of files    */
    long long tot_word_cnt = 1; /* Total number of words    */
    word_freq = allocmem(max_vocab,1,WRDFRQ);

    for (int i = 0; i < num_files; i++) {
        tot_file_cnt++;
        tot_word_cnt += process_news_file(file_list[i],data_dir,
                                          hmap,1,max_vocab,word_freq,NULL,0);
        printf("Processed file %d of %d, %lld words\r",
                                          i + 1,num_files,tot_word_cnt);
        fflush(stdout);
    }

    printf("\nDataset: %d files, %lld words, %d unique\n",
                                   tot_file_cnt,tot_word_cnt,hmap->map_used);
    fflush(stdout);

    qsort(word_freq,hmap->map_used,sizeof(WRDFRQ),qsort_compare_word_freq);

    /* Shift so index 0 is PAD, index i describes vocab word i. */
    for (int i = hmap->map_used - 1; i > 0; i--)
        word_freq[i] = word_freq[i-1];
    word_freq[0].inx = 0; word_freq[0].cnt = 0; word_freq[0].frq = 0.0f;

    long long word_cnt = 0;
    if (vocab_size == 0) {
        long long target = (long long)(vocab_coverage * (float) tot_word_cnt);
        vocab_size = 1;
        for (int i = 1; i < hmap->map_used; i++) {
            word_cnt += word_freq[i].cnt;
            vocab_size = i + 1;
            if (word_cnt >= target) break;
        }
    } else {
        if (vocab_size > hmap->map_used) vocab_size = hmap->map_used;
        for (int i = 1; i < vocab_size; i++) word_cnt += word_freq[i].cnt;
    }
    vocab_coverage = (float) word_cnt / (float) tot_word_cnt;
    printf("Vocabulary limited to %d words, covering %2.0f%% of corpus\n",
           vocab_size,100 * vocab_coverage);

    /* Re-index retained words into a compact hashmap. */
    HASHMAP* hmap2 = hashmap_create(vocab_size * 3,hmap->mem_used);
    hashmap_str2inx(hmap2,"",1);          /* PAD at 0 */
    for (int i = 1; i < vocab_size; i++) {
        const char* w = hashmap_inx2str(hmap,word_freq[i].inx);
        if (strlen(w) == 0) continue;
        word_freq[i].inx = hashmap_str2inx(hmap2,w,1);
    }
    hashmap_free(hmap); hmap = hmap2;

    for (int i = 1; i < vocab_size; i++)
        word_freq[i].frq = (float) word_freq[i].cnt / (float) word_cnt;

    /* Unigram distribution table (freq^0.75), PAD excluded. */
    const float dist_pow = 0.75f; const int dist_scale = 10;
    dist_size = 0;
    for (int i = 1; i < vocab_size; i++)
        dist_size += (int)(powf(word_freq[i].cnt,dist_pow)/dist_scale) + 1;
    dist = allocmem(dist_size,1,int);
    for (int i = 1, j = 0; i < vocab_size; i++) {
        int rpt = (int)(powf(word_freq[i].cnt,dist_pow)/dist_scale) + 1;
        for (int k = 0; j < dist_size && k < rpt; k++)
            dist[j++] = word_freq[i].inx;
    }
    printf("Distribution table size %d\n",dist_size); fflush(stdout);

    if (print_vocab) {
        printf("ord   index count word\n");
        for (int i = 0; i < vocab_size; i++)
            printf("%5d %5d %6d %-16s\n",i,word_freq[i].inx,word_freq[i].cnt,
                   hashmap_inx2str(hmap,word_freq[i].inx));
        return 0;
    }

    /* ---------------- Build model ---------------- */
    m = lm_create(vocab_size,model_dim,heads,seq_len,batch_size,
                  layers,ffn_dim,neg_samples,dropout,optimizer);
    negsample_set_dist(m->head,dist,dist_size);

    /* Fresh training schedule (the rest is filled at each checkpoint). */
    st.optimizer    = optimizer;
    st.lr_decay     = lr_decay;
    st.weight_decay = weight_decay;
    st.num_epochs   = num_epochs;
    } /* ---------------- end fresh-build path ---------------- */

    int* file_words = allocmem(1,max_file_words,int);
    const int BT = m->BT;

    /* Seed for periodic generation: the most frequent words are indices
     * 1.. (index 0 is PAD) in both the fresh and the resumed vocabulary
     */
    int seed[8];
    int seedlen = 0;
    for (int i = 1; i < vocab_size && seedlen < 4; i++)
        seed[seedlen++] = i;

    printf("\nTraining\n"); fflush(stdout);
    double start_time = current_time();

    int epoch = 0;
    for (epoch = start_epoch; epoch <= num_epochs; epoch++) {
        shuffle_list(file_list,num_files);
        float ep_loss = 0;
        long long ep_positions = 0;
        long long ep_correct = 0;

        for (int fi = 0; fi < num_files; fi++) {
            int fwcnt = process_news_file(file_list[fi],data_dir,
                            hmap,0,vocab_size,NULL,file_words,max_file_words);
            if (fwcnt <= 1) continue;

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
                        m->ids[row] = file_words[src];
                        m->pad_mask[row] = (file_words[src] > 0) ? 1 : 0;
                        ((float*) m->labels)[row] = (float) file_words[src+1];
                        filled++;
                    }
                }
                if (filled == 0) break;
                pos += batch_size * stride;

                /* Forward */
                fArr2D hhead = lm_forward(m);

                /* Loss + dh into m->dtop, sparse gWo into gHead[0]. */
                int correct = 0;
                float loss = negsample_loss(m->head,hhead,m->labels,
                                            m->gHead[0],m->dtop,BT,&correct);

                /* Backward through stack + embedding scatter. */
                lm_backward(m);

                /* Optimizer step. */
                update_cnt++;
                lm_update(m,optimizer,learning_rate,weight_decay,update_cnt);

                ep_loss += loss; ep_positions += filled; ep_correct += correct;

                if (update_cnt) {
                    int sec = (int) elapsed_time(start_time);
                    printf("epoch %2d lr %7.5f loss %7.4f acc %4.1f%% "
                           "(file %d/%d) %d:%02d:%02d\r",
                           epoch,learning_rate,
                           ep_positions ? ep_loss/ep_positions : 0.0,
                           ep_positions ? 100.0*ep_correct/ep_positions : 0.0,
                           fi+1,num_files,
                           sec/3600,(sec/60)%60,sec%60);
                    fflush(stdout);
                }
            }
        }
        printf("\nepoch %2d done: loss %7.4f  acc %4.1f%%\n",
               epoch, ep_positions ? ep_loss/ep_positions : 0.0,
               ep_positions ? 100.0*ep_correct/ep_positions : 0.0);

        if (gen_every > 0 && epoch % gen_every == 0)
            lm_generate(m,hmap,seed,seedlen,20,0.8f);

        learning_rate *= lr_decay;
        fflush(stdout);

        {
            st.optimizer     = optimizer;
            st.update_cnt    = update_cnt;
            st.epoch         = epoch;
            st.num_epochs    = num_epochs;
            st.learning_rate = learning_rate;
            st.lr_decay      = lr_decay;
            st.weight_decay  = weight_decay;
            st.lrng_seed     = get_lrng_seed();
            char fname[1024];
            snprintf(fname,sizeof fname,"%s.%d.model",save_base,epoch);
            if (store_lm(fname,m,0,hmap,dist,dist_size,&st))
                printf("Saved checkpoint '%s'\n",fname);
            else
                fprintf(stderr,"Failed to save checkpoint '%s'\n",fname);
            fflush(stdout);
        }
    }
    {
        st.optimizer     = optimizer;
        st.update_cnt    = update_cnt;
        st.epoch         = epoch;
        st.num_epochs    = num_epochs;
        st.learning_rate = learning_rate;
        st.lr_decay      = lr_decay;
        st.weight_decay  = weight_decay;
        st.lrng_seed     = get_lrng_seed();
        if (store_lm(output_file,m,0,hmap,dist,dist_size,&st))
            printf("Saved final model '%s'\n",output_file);
        else
            fprintf(stderr,"Failed to save final model '%s'\n",output_file);
    }
    printf("\nTraining complete\n");

    lm_free(m);
    hashmap_free(hmap);
    freemem(dist);
    freemem(word_freq);
    freemem(file_words);
    free_news_file_list(file_list,num_files);
    return 0;
}
