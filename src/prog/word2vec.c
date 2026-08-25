/* Copyright (c) 2023-2026 Gilad Odinak */

/* This program implements word2vec based on mikolov's papers.
 * The input is a file containing a list of files. Each of these files
 * contains some text.
 *
 * References:
 * - Efficient Estimation of Word Representations in Vector Space
 *   https://arxiv.org/pdf/1301.3781
 * - Distributed Representations of Words and Phrases and their Compositionality
 *   https://arxiv.org/pdf/1310.4546
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <getopt.h>
#include "mem.h"
#include "float.h"
#include "etime.h"
#include "random.h"
#include "array.h"
#include "hash.h"
#include "textfile.h"
#include "activation.h"
#include "embedding.h"
#include "negsample.h"
#include "wembio.h"

const char* usage =
"Usage: word2vec [options]\n"
"Options:\n"
"  -h                 Show this help message, then exit\n"
"  -b <batch_size>    Set batch size (default 16)\n"
"  -c <context_size>  Set context size (must be even, default 8)\n"
"  -d <embedding_dim> Set embedding dimension (default 100)\n"
"  -e <num_epochs>    Set number of epochs (default 10)\n"
"  -i <train_file>    Set training files list (def. data/news/all_files.lst)\n"
"  -n <neg_samples>   Num of negative samples per positive target (def 10)\n"
"  -o <output_file>   Set output embedding file (default word2vec.model)\n"
"  -r <learning_rate> Set starting learning rate (default 0.005)\n"
"  --rd=<f>           Learning rate decay (default 0.8)\n"
"  --vocab-size=<n>   Limit vocabulary size\n"
"  --vocab-coverage=<f> Limit vocabulary coverage (default 0.99)\n"
"  --print-vocab      Print vocabulary and exit\n"
"  --data-dir         Location of training data (default data/news)\n"
;

/* Creates a batch of contexts for words in a larger text sequence.
 *
 * Parameters:
 *   sw     - array of text word indices, in the order of the text
 *   swc    - number of entries in sw
 *   start  - offset of the first target word for this batch
 *   B      - number of rows allocated in cxt and labels
 *   cxt    - array that receives contexts [B][cs]
 *   labels - array that receives target word indices [B][1]
 *   cs     - context size; must be even
 *
 * Note: This function indexes into the full word sequence so
 * contexts can include words just outside the current batch.
 */
static void text2cxt(int* sw, int swc, int start, int B,
                     fArr2D cxt_, fArr2D labels_, int cs)
{
    typedef float (*ArrBC)[cs];
    typedef float (*ArrB1)[1];
    ArrBC cxt = (ArrBC) cxt_;
    ArrB1 labels = (ArrB1) labels_;

    fltclr(cxt,B * cs);
    fltclr(labels,B);

    int m = cs / 2;
    for (int b = 0; b < B; b++) {
        int i = start + b;
        if (i >= swc)
            break;

        labels[b][0] = sw[i];

        for (int j = 0; j < m; j++) {
            int k = i - m + j;
            if (k >= 0)
                cxt[b][j] = sw[k];
        }
        for (int j = 0; j < m; j++) {
            int k = i + 1 + j;
            if (k < swc)
                cxt[b][m + j] = sw[k];
        }
    }
}

/* Update layer's weights for selected rows.
 *
 * Applies weight -= lr * gradient only to the rows listed in rows[nrows].
 * Used with negative sampling and the embedding backward pass, where eachi
 * step touches few rows of a large weight matrix.
 */
static void update(fArr2D Wx_, fArr2D gWx_,
                   const int* rows, int nrows, int N, float lr)
{
    typedef float (*ArrDN)[N];
    ArrDN Wx = (ArrDN) Wx_;
    ArrDN gWx = (ArrDN) gWx_;
    for (int r = 0; r < nrows; r++) {
        int i = rows[r];
        for (int j = 0; j < N; j++)
            Wx[i][j] -= lr * gWx[i][j];
    }
}

/* Returns the embedding vector of a word
 * embd - embedding layer
 * wrdinx - the index of the word
 */
float* word_embedding(EMBEDDING* embd, int wrdinx)
{
    if (wrdinx < 0 || wrdinx >= embd->D)
        wrdinx = 0;
    typedef float (*ArrDE)[embd->E];
    ArrDE Wx = (ArrDE)  embd->Wx;
    return Wx[wrdinx];
}



/* Compare two word frequency values - used with qsort to order
 * words based on their frequency in descending order,
 * that is, most frequent word first.
 */
int qsort_compare_word_freq(const void *a, const void *b)
{   /* WRDFRQ declared in textfile.h */
    if (((WRDFRQ *)b)->cnt > ((WRDFRQ *)a)->cnt) return 1;
    if (((WRDFRQ *)b)->cnt < ((WRDFRQ *)a)->cnt) return -1;
    return 0;
}

void shuffle_list(char** list, int cnt)
{
    if (cnt <= 1) return;
    for (int i = cnt - 1; i > 0; i--) {
        int j = (int) urand(0,i + 1);
        char* temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}

int main(int argc, char** argv)
{
    char* data_dir = "data/news/data";         /* Input  */
    char* tr_file = "data/news/all_files.lst"; /* Input  */
    char* embedding_file = "word2vec.model";   /* Output */
    float vocab_coverage = 0.99; /* 99% */
    int vocab_size = 0; /* Default: size derived from vocab_coverage */
    int embedding_dim = 100;
    int batch_size = 16;
    int cxt_size = 8;
    int num_epochs = 10;
    float learning_rate = 0.005;
    float learning_rate_decay = 0.8;
    int print_vocab = 0;
    int max_vocab = 10000000;  /* Set to 3 x expected number of unique words */
    int hash_mem = 100000000;  /* hashmap will increase this value as needed */
    int max_file_words = 1000000; /* Maximum number of words per file        */
    int neg_samples = 10;      /* Negative samples per positive target       */

    int opt;
    while ((opt = getopt(argc,argv,"b:c:d:e:i:n:o:r:-:h")) != -1) {
        switch (opt) {
            case 'h': printf(usage); exit(0);
            case 'b': batch_size = atoi(optarg); break;
            case 'c': cxt_size = atoi(optarg); break;
            case 'd': embedding_dim = atoi(optarg); break;
            case 'e': num_epochs = atoi(optarg); break;
            case 'i': tr_file = optarg; break;
            case 'n': neg_samples = atoi(optarg); break;
            case 'o': embedding_file = optarg; break;
            case 'r': learning_rate = atof(optarg); break;
            case '-':
                if (strncmp(optarg,"rd=",3) == 0)
                    learning_rate_decay = atof(optarg+3);
                else
                if (strncmp(optarg,"vocab-size=",11) == 0)
                    vocab_size = atoi(optarg+11);
                else
                if (strncmp(optarg,"vocab-coverage=",15) == 0)
                    vocab_coverage = atof(optarg+15);
                else
                if (strncmp(optarg,"print-vocab",11) == 0)
                    print_vocab = 1;
                else
                if (strncmp(optarg,"data-dir=",9) == 0)
                    data_dir = optarg+9;
                else
                    goto opterr;
            break;
            case '?':
            default:
            opterr:
                fprintf(stderr,"word2vec: syntax error\n");
                printf(usage);
                exit(-1);
        }
    }

    if (cxt_size % 2) {
        fprintf(stderr,"word2vec: context size must be an even number. got -c %d\n",cxt_size);
        exit(-1);
    }
    float initial_learning_rate = learning_rate;

    printf("\n");
    printf("Trains an embedding layer to create word embeddings using\n");
    printf("Continuous Bag of Words (CBOW) method\n");
    printf("context size = %d, embedding dim = %d, batch size = %d\n"
           "%d epochs, learning_rate = %g learning_rate_decay = %g\n",
           cxt_size,embedding_dim,batch_size,
           num_epochs,initial_learning_rate,learning_rate_decay);
    fflush(stdout);

    int tot_file_cnt = 0;       /* Total number of files    */
    long long tot_word_cnt = 0; /* Total number of words    */

    printf("Creating vocabulary from dataset\n");
    fflush(stdout);
    HASHMAP* hmap = hashmap_create(max_vocab,hash_mem);
    hashmap_str2inx(hmap,"",1);
    tot_word_cnt++;
    WRDFRQ* word_freq = allocmem(max_vocab,1,WRDFRQ);

    int num_files = 0;
    char** file_list = read_text_file_list(tr_file,data_dir,&num_files);
    if (file_list == NULL || num_files == 0) {
        fprintf(stderr,"Failed to read data files list from '%s'\n",tr_file);
        return -1;
    }
    for (int i = 0; i < num_files; i++) {
        tot_file_cnt++;
        tot_word_cnt += process_text_file(file_list[i],data_dir,
                                          hmap,1,max_vocab,word_freq,NULL,0);
        printf("Processed file %d of %d, %lld words\r",
                                          i + 1,num_files,tot_word_cnt);
        fflush(stdout);
    }

    printf("\nDataset: %d files, %lld words, ",tot_file_cnt,tot_word_cnt);
    printf("%d unique words\n",hmap->map_used);
    printf("%d bytes of word storage memory used\n",hmap->mem_used);
    fflush(stdout);

    /* Sort vocabulary words by frequency, descending */
    qsort(word_freq,hmap->map_used,sizeof(WRDFRQ),qsort_compare_word_freq);

    /* Reserve word_freq[0] for PAD, matching hashmap index 0.
     * After this, word_freq[i] describes vocabulary index i.
     * The last sorted entry is dropped if the table was completely full.
     */
    for (int i = hmap->map_used - 1; i > 0; i--)
        word_freq[i] = word_freq[i - 1];
    word_freq[0].inx = 0;
    word_freq[0].cnt = 0;
    word_freq[0].frq = 0.0f;

    long long word_cnt = 0;
    if (vocab_size == 0) {
        /* Calculate how many most frequent vocabulary words are needed
         * to represent vocab_coverage percent of all corpus words.
         * vocab_size includes PAD at index 0.
         */
        long long target_word_cnt = (long long)(vocab_coverage * ((float)tot_word_cnt));
        vocab_size = 1; /* Include PAD */
        for (int vocab_inx = 1; vocab_inx < hmap->map_used; vocab_inx++) {
            word_cnt += word_freq[vocab_inx].cnt;
            vocab_size = vocab_inx + 1;
            if (word_cnt >= target_word_cnt)
                break;
        }
    }
    else {
        /* Calculate what percentage of all corpus words can be
         * represented by the vocab_size most frequent vocabulary entries.
         * vocab_size includes PAD at index 0, which is not counted.
         */
        if (vocab_size > hmap->map_used)
            vocab_size = hmap->map_used;
        for (int vocab_inx = 1; vocab_inx < vocab_size; vocab_inx++)
            word_cnt += word_freq[vocab_inx].cnt;
    }
    vocab_coverage = ((float) word_cnt) / tot_word_cnt;
    printf("Limit vocabulary to %d most frequent words\n",vocab_size);
    printf("The vocabulary covers %lld (%2.0f%%) of dataset words\n",
                                       word_cnt,100 * vocab_coverage);

    /* Create new vocabulary index of only the retained words,
     * in order of their frequency in the dataset.
     */
    printf("Creating vocabulary of %d words\n",vocab_size);
    fflush(stdout);
    HASHMAP* hmap2 = hashmap_create(vocab_size * 3,hmap->mem_used);
    hashmap_str2inx(hmap2,"",1); /* Reserve first entry (index 0) for pad */

    /* Add words up to vocabulary size. PAD already occupies index 0. */
    for (int i = 1; i < vocab_size; i++) {
        const char* wrd = hashmap_inx2str(hmap,word_freq[i].inx);
        if (strlen(wrd) == 0)
            continue;
        int inx = hashmap_str2inx(hmap2,wrd,1);
        word_freq[i].inx = inx;
    }
    hashmap_free(hmap);
    hmap = hmap2;
    hmap2 = NULL;

    printf("Calculating word frequencies\n");
    word_freq[0].frq = 0.0;
    for (int i = 1; i < vocab_size; i++)
        word_freq[i].frq = ((float) word_freq[i].cnt) / word_cnt;

    printf("Creating vocabulary distribution table\n");
    fflush(stdout);
    float dist_compress = 0.75;
    int dist_scale = 10;
    int dist_table_size = 0;
    /* Calculate requried size for all tokens (excluding pad) */
    for (int i = 1; i < vocab_size; i++) {
        int freq = word_freq[i].cnt;
        dist_table_size += (int)(pow(freq,dist_compress)/dist_scale) + 1;
    }
    int* dist_table = allocmem(dist_table_size,1,int);
    printf("Distribution table size %d\n",dist_table_size);

    /* Populate table, note pad (i == 0) is excluded */
    for (int i = 1, j = 0; i < vocab_size; i++) {
        int inx = word_freq[i].inx;
        int freq = word_freq[i].cnt;
        int rpt = (int)(pow(freq,dist_compress)/dist_scale) + 1;
        for (int k = 0; j < dist_table_size && k < rpt; k++)
            dist_table[j++] = inx;
    }

    if (print_vocab) {
        printf("ord   index count word\n");
        for (int i = 0; i < vocab_size; i++) {
            const char* wrdstr = hashmap_inx2str(hmap,word_freq[i].inx);
            printf("%5d %5d %5d %-16s\n",i,word_freq[i].inx,word_freq[i].cnt,wrdstr);
        }
        printf("ord   index word\n");
        for (int i = 0; i < dist_table_size; i++) {
            const char* wrdstr = hashmap_inx2str(hmap,dist_table[i]);
            printf("%5d %5d %-16s\n",i,dist_table[i],wrdstr);
        }
        return 0;
    }

    printf("\n");
    float start_time = current_time();
    printf("Creating word embeddings\n");
    fflush(stdout);

    /* Create and initialize layers */
    EMBEDDING* embedding = embedding_create(embedding_dim,cxt_size,0);
    embedding_init(embedding,vocab_size,batch_size,1);
    NEGSAMPLE* output = negsample_create(vocab_size,neg_samples);
    negsample_init(output,embedding_dim,batch_size);
    negsample_set_dist(output,dist_table,dist_table_size);

    /* Allocate memory for gradients */
    fArr2D dy;  /* Gradients with respect to the input  */
    dy = allocmem(embedding->B,embedding->E,float);
    fArr2D gWx[2]; /* Gradients: [0] input embeddings, [1] output weights */
    gWx[0] = allocmem(embedding->D,embedding->E,float);
    gWx[1] = allocmem(vocab_size,embedding_dim,float);

    /* Rows of Wx (input embeddings) touched per batch: at most B context words */
    int* touched_in = allocmem(batch_size * cxt_size,1,int);

    int* file_words = allocmem(1,max_file_words,int);
    int file_cnt;

    for (int epoch = 1; epoch <= num_epochs; epoch++ ) {
        word_cnt = 0;
        float loss = 0;
        file_cnt = 0;
        shuffle_list(file_list,num_files);
        for (int i = 0; i < num_files; i++) {
            file_cnt++;
            int fwcnt = process_text_file(file_list[i],data_dir,
                              hmap,0,max_vocab,NULL,file_words,max_file_words);

            /* Sub sample frequent words by removing some */
            int cnt = 0;
            for (int i = 0; i < fwcnt; i++) {
                int wrdinx = file_words[i];
                if (wrdinx <= 0 || wrdinx >= vocab_size)
                    continue;
                float r = urand(0,1.0);
                float t = 1e-5;
                float f = word_freq[wrdinx].frq;
                if (f <= 0.0f)
                    continue;
                float p = (sqrt(f / t) + 1.0) * (t / f);
                if (p > 1.0) p = 1.0;
                if (r < p)
                    file_words[cnt++] = wrdinx;
            }

            /* Process each word in current file */
            for (int i = 0; i < cnt; i += batch_size) {

                /* Create word contexts, which consist of cxt_size words
                 * that are adjacent to the current word.
                 */
                float contexts[batch_size][cxt_size];
                float labels[batch_size][1];
                int wcnt = batch_size;
                if (i + wcnt >= cnt)
                    wcnt = cnt - i;
                text2cxt(file_words,cnt,i,batch_size,contexts,labels,cxt_size);

                for (int j = 0; j < wcnt; j++) {
                    if (labels[j][0] >= vocab_size) {
                        printf("labels[%d] %g file_word[%d]\n",j,labels[j][0],i + j);
                        exit(1);
                    }
                }

                if (wcnt < batch_size) {
                    for (int j = wcnt; j < batch_size; j++) {
                        fltclr(contexts[j],cxt_size);
                        labels[j][0] = 0;
                    }
                }

                /* Forward pass */
                fArr2D yp = embedding_forward(embedding,contexts,0);

                loss += negsample_loss(output,yp,labels,gWx[1],dy,
                                       batch_size,NULL);

                /* backward pass */
                int ntouched_in = 0;
                embedding_backward(embedding,dy,contexts,gWx[0],
                                   touched_in,&ntouched_in,0);

                /* Update weights. Both matrices are updated only on the rows
                 * their sparse gradients actually touched this batch.
                 */
                update(embedding->Wx,gWx[0],touched_in,ntouched_in,
                                                   embedding->E,learning_rate);
                negsample_update(output,gWx[1],learning_rate,0.0);

                word_cnt += wcnt;
                int pct = (tot_file_cnt >= 1) ?
                                (((float)file_cnt / tot_file_cnt) * 100) : 100;
                int seconds = (int) elapsed_time(start_time);
                int sec = seconds % 60;
                int min = (seconds / 60) % 60;
                int hours = seconds / 3600;
                printf("epoch %2d lr %6.4f loss %6.4f %3d%% "
                       "(file %d of %d, %lld words) %d:%02d:%02d\r",
                       epoch,learning_rate,loss / word_cnt,pct,
                       file_cnt,tot_file_cnt,word_cnt,hours,min,sec);
                fflush(stdout);
            }
        }
        printf("\n");
        learning_rate = learning_rate_decay * learning_rate;
    }

    printf("\n");
    printf("Saving word embeddings to %s\n",embedding_file);
    if (!store_word_embeddings(embedding_file,vocab_size,embedding_dim,
                               initial_learning_rate,learning_rate_decay,
                               num_epochs,hmap,embedding->Wx))
        fprintf(stderr,"Failed to save word embeddings to '%s'\n",
                embedding_file);
    printf("\n");
    hashmap_free(hmap);
    embedding_free(embedding);
    negsample_free(output);
    freemem(dy);
    freemem(gWx[0]);
    freemem(gWx[1]);
    freemem(touched_in);
    freemem(dist_table);
    freemem(word_freq);
    freemem(file_words);
    free_text_file_list(file_list,num_files);
    return 0;
}
