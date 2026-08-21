/* Copyright (c) 2023-2024 Gilad Odinak */
/* 
 * This file includes some basic tests for the embedding layer,
 * as well as toy demo program for the Embedding layer, inspired by
 * "Linguistic Regularities in Continuous Space Word Representations"
 * https://aclanthology.org/N13-1090.pdf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "hash.h"
#include "loss.h"
#include "embedding.h"
#include "dense.h"
#include "clip.h"
#include "cossim.h"
#include "normalize.h"
#include "pca.h"

#define EPS 1e-3
#define TOL 1e-2

/* Test dimensions */
#define VOCAB 7   /* D: vocabulary size (index 0 is the pad row) */
#define EMBD  4   /* E: embedding dimension                      */
#define CTX   4   /* M: context length (must be even)            */
#define BATCH 5   /* B: number of contexts in a batch            */
#define PAD   0   /* pad index                                   */

int failures = 0;

/* Builds and initializes a fresh embedding layer with the test dimensions. */
static EMBEDDING* make_embedding(void)
{
    EMBEDDING* l = embedding_create(EMBD,CTX,PAD);
    embedding_init(l,VOCAB,BATCH,1);
    return l;
}

/* Fills an [B][M] index array with random token indices in [0, D).
 * Includes the pad index, so pad handling is exercised.
 */
static void fill_random_indices(fArr2D X_, int B, int M, int D)
{
    typedef float (*ArrBM)[M];
    ArrBM X = (ArrBM) X_;
    for (int i = 0; i < B; i++)
        for (int j = 0; j < M; j++)
            X[i][j] = (float)(int) urand(0.0,(float) D);
}

/* Scalar loss L = sum_{i,k} dy[i][k] * h[i][k], where h is the forward output.
 * Using a random dy (rather than all ones) makes the gradient check sensitive
 * to per-row/column mistakes, not just to row sums.
 */
static float embed_loss(EMBEDDING* l, fArr2D X, fArr2D dy_)
{
    typedef float (*ArrBE)[l->E];
    ArrBE h  = (ArrBE) embedding_forward(l, X, 0);
    ArrBE dy = (ArrBE) dy_;
    float L = 0;
    for (int i = 0; i < l->B; i++)
        for (int k = 0; k < l->E; k++)
            L += dy[i][k] * h[i][k];
    return L;
}

/* With all weights zero, the forward output must be zero. */
static void test_embedding_zero_forward(EMBEDDING* l)
{
    printf("Test: embedding zero forward\n");

    float X[BATCH][CTX];
    fill_random_indices((fArr2D) X, l->B, l->M, l->D);
    fltclr(l->Wx, l->D * l->E);

    typedef float (*ArrBE)[l->E];
    ArrBE h = (ArrBE) embedding_forward(l, (fArr2D) X, 0);

    for (int i = 0; i < l->B; i++)
        for (int k = 0; k < l->E; k++)
            if (fabsf(h[i][k]) > 1e-9f) {
                printf("FAIL: h[%d][%d]=%g, expected 0\n", i, k, h[i][k]);
                failures++;
                return;
            }
    printf("PASS\n");
}

/* With Wx[w][k] = w (each row equals its own index), the output for a context
 * must equal the sum of that context's indices. Also confirms the pad row
 * (index 0, value 0) contributes nothing.
 */
static void test_embedding_forward_sum(EMBEDDING* l)
{
    printf("Test: embedding forward sum\n");

    typedef float (*ArrDE)[l->E];
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBE)[l->E];

    ArrDE Wx = (ArrDE) l->Wx;
    for (int w = 0; w < l->D; w++)
        for (int k = 0; k < l->E; k++)
            Wx[w][k] = (float) w;

    float X[BATCH][CTX];
    ArrBM Xr = (ArrBM) X;
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->M; j++)
            Xr[i][j] = (float)((i + j) % l->D);   /* deterministic indices */

    ArrBE h = (ArrBE) embedding_forward(l, (fArr2D) X, 0);

    for (int i = 0; i < l->B; i++) {
        float expect = 0;
        for (int j = 0; j < l->M; j++)
            expect += Xr[i][j];        /* row value == index, so h == sum idx */
        for (int k = 0; k < l->E; k++)
            if (fabsf(h[i][k] - expect) > TOL) {
                printf("FAIL: h[%d][%d]=%g, expected %g\n", i, k, h[i][k], expect);
                failures++;
                return;
            }
    }
    printf("PASS\n");
}

/* Compares the analytic weight gradient gWx against a central finite
 * difference for every non-pad weight, and checks that the pad row receives
 * exactly zero gradient (it must never be trained).
 */
static void test_embedding_finite_diff(EMBEDDING* l)
{
    printf("Test: embedding finite-difference gradients (gWx)\n");

    typedef float (*ArrDE)[l->E];

    float X[BATCH][CTX];
    float dy[BATCH][EMBD];

    ArrDE Wx   = (ArrDE) l->Wx;
    fArr2D gWx = allocmem(l->D, l->E, float);
    ArrDE g    = (ArrDE) gWx;

    fill_random_indices((fArr2D) X, l->B, l->M, l->D);
    for (int i = 0; i < l->B; i++)
        for (int k = 0; k < l->E; k++)
            dy[i][k] = urand(-1.0, 1.0);

    /* Analytic gradient (dense path). backward does not read Wx, so later
     * perturbing Wx for the finite difference leaves g untouched. */
    embedding_forward(l, (fArr2D) X, 0);
    embedding_backward(l, (fArr2D) dy, (fArr2D) X, gWx, NULL, NULL, 0);

    /* Pad row must have zero gradient. */
    for (int k = 0; k < l->E; k++)
        if (fabsf(g[PAD][k]) > 1e-9f) {
            printf("FAIL: gWx[pad=%d][%d]=%g, expected 0\n", PAD, k, g[PAD][k]);
            failures++;
            return;
        }

    /* Finite-difference check for every non-pad weight. */
    for (int w = 0; w < l->D; w++) {
        if (w == l->padinx)
            continue;
        for (int k = 0; k < l->E; k++) {
            float old = Wx[w][k];
            Wx[w][k] = old + EPS;
            float Lp = embed_loss(l, (fArr2D) X, (fArr2D) dy);
            Wx[w][k] = old - EPS;
            float Ln = embed_loss(l, (fArr2D) X, (fArr2D) dy);
            Wx[w][k] = old;

            float num = (Lp - Ln) / (2 * EPS);
            float ana = g[w][k];
            if (fabsf(num - ana) > TOL) {
                printf("FAIL gWx[%d][%d]: numeric=%g analytic=%g\n",
                       w, k, num, ana);
                failures++;
                return;
            }
        }
    }
    freemem(gWx);
    printf("PASS\n");
}

/* The sparse path (grows/grcnt) must produce the same gWx as the dense path,
 * and grows must list exactly the distinct non-pad context rows.
 */
static void test_embedding_sparse_dense_equiv(EMBEDDING* l)
{
    printf("Test: embedding sparse vs dense gradient\n");

    typedef float (*ArrDE)[l->E];
    typedef float (*ArrBM)[l->M];

    float X[BATCH][CTX];
    float dy[BATCH][EMBD];

    fill_random_indices((fArr2D) X, l->B, l->M, l->D);
    for (int i = 0; i < l->B; i++)
        for (int k = 0; k < l->E; k++)
            dy[i][k] = urand(-1.0, 1.0);

    fArr2D gd = allocmem(l->D, l->E, float);
    fArr2D gs = allocmem(l->D, l->E, float);
    fltclr(gd, l->D * l->E);
    fltclr(gs, l->D * l->E);  /* sparse leaves untouched rows as-is: pre-zero */

    embedding_forward(l, (fArr2D) X, 0);
    embedding_backward(l, (fArr2D) dy, (fArr2D) X, gd, NULL, NULL, 0);

    int grows[BATCH * CTX];
    int grcnt = 0;
    embedding_backward(l, (fArr2D) dy, (fArr2D) X, gs, grows, &grcnt, 0);

    ArrDE A = (ArrDE) gd;
    ArrDE Bm = (ArrDE) gs;
    for (int w = 0; w < l->D; w++)
        for (int k = 0; k < l->E; k++)
            if (fabsf(A[w][k] - Bm[w][k]) > 1e-6f) {
                printf("FAIL: gWx[%d][%d] dense=%g sparse=%g\n",
                       w, k, A[w][k], Bm[w][k]);
                failures++;
                return;
            }

    /* Expected distinct non-pad rows from X. */
    int seen[l->D];
    for (int w = 0; w < l->D; w++)
        seen[w] = 0;
    ArrBM Xr = (ArrBM) X;
    int expect = 0;
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->M; j++) {
            int w = (int) Xr[i][j];
            if (w != l->padinx && !seen[w]) {
                seen[w] = 1;
                expect++;
            }
        }
    if (grcnt != expect) {
        printf("FAIL: grcnt=%d, expected %d distinct non-pad rows\n",
               grcnt, expect);
        failures++;
        return;
    }
    /* grows must contain exactly those rows, with no duplicates. */
    for (int r = 0; r < grcnt; r++) {
        int w = grows[r];
        if (w == l->padinx || w < 0 || w >= l->D || !seen[w]) {
            printf("FAIL: grows[%d]=%d is not a distinct non-pad row\n", r, w);
            failures++;
            return;
        }
        seen[w] = 0;  /* clear so a duplicate would be caught next time */
    }
    freemem(gd);
    freemem(gs);
    printf("PASS\n");
}

int run_tests(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned int seed = (unsigned int)(ts.tv_sec ^ ts.tv_nsec);
    printf("seed %u\n", seed);
    init_lrng(seed);

    printf("EPS %g TOL %g\n", EPS, TOL);
    printf("vocab %d embd %d ctx %d batch %d pad %d\n",
           VOCAB, EMBD, CTX, BATCH, PAD);

    EMBEDDING* l;

    l = make_embedding(); test_embedding_zero_forward(l);        embedding_free(l);
    l = make_embedding(); test_embedding_forward_sum(l);         embedding_free(l);
    l = make_embedding(); test_embedding_finite_diff(l);         embedding_free(l);
    l = make_embedding(); test_embedding_sparse_dense_equiv(l);  embedding_free(l);

    if (failures == 0)
        printf("\nALL TESTS PASSED\n");
    return 0;
}

char *sentences[] = {
    "At dawn, the skilled carpenter began crafting a beautiful wooden table for the village square.",
    "During the grand feast, the jovial king entertained the guests with his witty stories.",
    "After a long journey through the forest, the king discovered a hidden treasure chest.",
    "Under the shade of the old oak tree, the apprentice potter practiced molding clay into elegant forms.",
    "At the dawn of the new era, the visionary queen proposed radical changes to the council.",
    "In the quiet village square, an old queen recited tales of ancient heroes.",
    "Amidst the bustling marketplace, the merchant queen haggled over the price of silk.",
    "In the quiet village square, an old man recited tales of ancient heroes.",
    "At the dawn of the new era, the visionary man proposed radical changes to the council.",
    "After a long journey through the forest, the queen discovered a hidden treasure chest.",
    "Amidst the bustling marketplace, the merchant man haggled over the price of silk.",
    "At the dawn of the new era, the visionary woman proposed radical changes to the council.",
    "In the heat of the battle, the warrior king fought valiantly to defend his people.",
    "After a long journey through the forest, the woman discovered a hidden treasure chest.",
    "During the grand feast, the jovial queen entertained the guests with her witty stories.",
    "After a long journey through the forest, the man discovered a hidden treasure chest.",
    "With unwavering determination, the young woman scaled the treacherous mountain peak.",
    "In the quiet village square, an old woman recited tales of ancient heroes.",
    "With unwavering determination, the young king scaled the treacherous mountain peak.",
    "In the heat of the battle, the warrior man fought valiantly to defend his people.",
    "During the grand feast, the jovial woman entertained the guests with her witty stories.",
    "In the bustling market, the experienced carpenter sold intricately designed chairs to eager buyers.",
    "Amidst the bustling marketplace, the merchant woman haggled over the price of silk.",
    "In the bustling market, the experienced potter displayed his vibrant ceramic bowls, attracting a crowd.",
    "At the dawn of the new era, the visionary king proposed radical changes to the council.",
    "Under the shade of the old oak tree, the apprentice carpenter learned the art of building sturdy furniture.",
    "In the quiet village square, an old king recited tales of ancient heroes.",
    "Amidst the bustling marketplace, the merchant king haggled over the price of silk.",
    "At dawn, the skilled potter began shaping a delicate vase on the wheel in his workshop.",
    "In the heat of the battle, the warrior woman fought valiantly to defend her people.",
    "With unwavering determination, the young man scaled the treacherous mountain peak.",
    "In the heat of the battle, the warrior queen fought valiantly to defend her people.",
    "During the grand feast, the jovial man entertained the guests with his witty stories.",
    "With unwavering determination, the young queen scaled the treacherous mountain peak."
};

/* Returns a pointer to the first letter (a-z A-Z) in a string, or NULL */
static inline const char* first_letter(const char* s)
{
    while (*s != '\0' && !isalpha(*s)) s++;
    return (*s != '\0') ? s : NULL;
}

/* Returns a pointer to the first non letter in a string */
static inline const char* first_nonletter(const char* s)
{
    while (isalpha(*s)) s++;
    return s;
}

/* Creates contexts for a sentence's words. one context per word.
 *
 * sw  - array of sentence word indices, in the order of the sentence words
 * swc - number of entries in sw
 * cxt - array that receives the contexts (output); it has swc rows
 * cs  - context size; number of word indices in a context, columns in cxt.
 *       Must be an even number.
 *
 * Each context is stored in a row of cxt array. Assumes cs is an even number.
 * The context consist of the indices of cs/2 words preceding the context's
 * target word, followed by the indices of cs/2 words that come after it,
 * in order,
 */
static void sent2cxt(int* sw, int swc, fArr2D cxt_, int cs)
{
    typedef float (*ArrCS)[cs];
    ArrCS cxt = (ArrCS) cxt_;

    fltclr(cxt,swc * cs); /* Set all elements to the pad (index 0) value */

    int m = cs / 2;
    for (int i = 0; i < swc; i++) {
        for (int j = m, k = i + 1; j < cs && k < swc; j++, k++)
            cxt[i][j] = sw[k];
        for (int j = m - 1, k = i - 1; j >= 0 && k >= 0; j--, k--)
            cxt[i][j] = sw[k];
    }
}

/* Swaps row i with row j of array a[M][N]
 */
static inline void swap_rows(fArr2D a_, int M, int N, int i, int j)
{
    typedef float (*ArrMN)[N];
    ArrMN a = (ArrMN) a_;
    (void) M;
    float t;
    for (int k = 0; k < N; k++) {
        t = a[i][k];
        a[i][k] = a[j][k];
        a[j][k] = t;
    }
}

/* Shuffles the rows of arrays a[M][N], and l[M][1]
 */
static void shuffle_samples(fArr2D a, fArr2D l, int M, int N)
{
    for (int i = M - 1; i > 0; i--) {
        int j = (int) urand(0.0,1.0 + i);
        swap_rows(a,M,N,i,j);
        swap_rows(l,M,1,i,j);
    }
}

/* Updates layer's weights in a linear way:
 * weight = weight - learning_rate * weight_gradient
 * Wx:  weight matrix [D][N]
 * gWx: gradient matrix [D][N]
 * lr:  learning_rate
 */
static void update(fArr2D Wx_, fArr2D gWx_, int D, int N, float lr)
{
    typedef float (*ArrDN)[N];
    ArrDN Wx = (ArrDN) Wx_;
    ArrDN gWx = (ArrDN) gWx_;
    clip_gradients(gWx,D,N,1e-12,10);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < N; j++)
            Wx[i][j] -= lr * gWx[i][j];
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

/* Stores cosine similarity of a word to a reference word - used with qsort */
typedef struct wrdsim_s {
    int wrdinx;
    float cossim;
} WRDSIM;

/* Compare two consine similarity values - used with qsort to order
 * words that are similar to a reference word in descending order,
 * that is, most similar (highest cosine similarity value) first.
 */
int qsort_compare(const void *a, const void *b)
{
    float diff = ((WRDSIM *)b)->cossim - ((WRDSIM *)a)->cossim;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

int run_demo(void)
{
    /* Note that the below parameters as well as the input text
     * were tweaked to yield the desired result of
     *   king - man + woman --> queen
     * To robustly train word embeddings to produce this result
     * the raining will need much more data with greater content
     * variation and large embedding dimention. Nevertheless,
     * the algorithmic principle stays the same.
     */
    printf("\n");
    int seed = 2029831955;
    printf("seed %u\n", seed);
    init_lrng(seed);
    int cxt_size = 4;
    int embedding_dim = 6;
    int num_epochs = 100;
    float learning_rate = 0.025;

    printf("Trains an embedding layer to create word embeddings using\n");
    printf("Continuous Bag of Words (CBOW) method\n");
    printf("context_size = %d, embedding_dim = %d\n"
           "%d epochs, learning_rate = %g\n\n",
           cxt_size,embedding_dim,num_epochs,learning_rate);
    /* Calculate required size of hashmap that stores unique words */
    int sent_cnt = sizeof(sentences) / sizeof(sentences[0]);
    int word_cnt = 0; /* Total number of words    */
    int cxt_cnt = 0;  /* Total number of contexts */
    int mem_size = 0; /* Hashmap memory size for storing words (strings) */
    int msw_cnt = 0;  /* Highest number of words in a sentence  */
    for (int i = 0; i < sent_cnt; i++) {
        mem_size += strlen(sentences[i]);
        int swc = 0;
        const char* w = first_letter(sentences[i]);
        while (w != NULL) {
            const char* e = first_nonletter(w);
            int len = e - w;
            if (len == 0)
                break;
            w = first_letter(e);
            swc++;
        }
        word_cnt += swc;
        cxt_cnt += swc;
        if (swc > msw_cnt)
            msw_cnt = swc;
    }
    /* Create hashmap
     * Assume number of unique words << word_cnt, so hashmap is sparse.
     * Similarily space required to store unique words << mem_size
     */
    HASHMAP* hmap = hashmap_create(word_cnt + 1,mem_size);
    hashmap_str2inx(hmap,"",1); /* Reserve first entry (index 0) for pad */

    /* Create contexts */
    float contexts[cxt_cnt][cxt_size];
    float labels[cxt_cnt][1];
    int cxt_inx = 0;
    for (int i = 0; i < sent_cnt && cxt_inx < cxt_cnt; i++) {
        /* Obtain and store indices of words in a sentences */
        int sw[msw_cnt];
        int swc = 0;
        const char* w = first_letter(sentences[i]);
        while (w != NULL) {
            const char* e = first_nonletter(w);
            int len = e - w;
            if (len == 0)
                break;
            char word[len + 1];
            for (int j = 0; j < len; j++)
                word[j] = tolower(w[j]);
            word[len] = '\0';
            int k = hashmap_str2inx(hmap,word,1);
            sw[swc++] = k;
            w = first_letter(e);
        }
        if (cxt_inx + swc > cxt_cnt)
            swc = cxt_cnt - cxt_inx;

        /* Create contexts from sentence word indices */
        sent2cxt(sw,swc,(fArr2D) contexts[cxt_inx],cxt_size);

        /* Label the context */
        for (int r = 0; r < swc; r++)
            labels[cxt_inx + r][0] = sw[r];

        cxt_inx += swc;
    }
    cxt_cnt = cxt_inx;

    /* Create and initialize layers */
    int vocab_size = hmap->map_used; /* Already includes pad (at index 0) */
    EMBEDDING* embedding = embedding_create(embedding_dim,cxt_size,0);
    embedding_init(embedding,vocab_size,cxt_cnt,1);
    DENSE* dense = dense_create(vocab_size,"softmax");
    dense_init(dense,embedding_dim,cxt_cnt);

    /* Allocate memory for gradients */
    fArr2D dy[2];  /* Gradients with respect to the inputs  */
    fArr2D gWx[2]; /* Gradients with respect to the weights */
    dy[0] = allocmem(embedding->B,embedding->E,float);
    dy[1] = allocmem(dense->B,dense->S,float);
    gWx[0] = allocmem(embedding->D,embedding->E,float);
    gWx[1] = allocmem(dense->D,dense->S,float);

    printf("%d sentences, %d words, %d unique words, %d contexts\n\n",
                                    sent_cnt,word_cnt,vocab_size,cxt_cnt);
    /* Train */
    float losses[num_epochs];
    for (int i = 0; i < num_epochs; i++) {
        fArr2D yp[2]; /* pointers to layers' prediction arrays */
        shuffle_samples(contexts,labels,cxt_cnt,cxt_size);
        float loss = 0;

        /* Forward pass */
        yp[0] = embedding_forward(embedding,contexts,0);
        yp[1] = dense_forward(dense,yp[0],1);
        loss += sparse_cross_entropy_loss(yp[1],labels,cxt_cnt,vocab_size);

        /* Backward pass */
        dLdy_sparse_cross_entropy_loss(yp[1],labels,dy[1],cxt_cnt,vocab_size);
        dense_backward(dense,dy[1],yp[0],gWx[1],dy[0],1);
        embedding_backward(embedding,dy[0],contexts,gWx[0],NULL,NULL,0);

        /* Update weights */
        update(embedding->Wx,gWx[0],embedding->D,embedding->E,learning_rate);
        update(dense->Wx,gWx[1],dense->D,dense->S,learning_rate);

        loss /= cxt_cnt;
        losses[i] = loss;
        printf("Epoch %5d loss %7.4f\r",i + 1,loss);
        fflush(stdout);
    }
    printf("\n\n");

    float* man_vec = word_embedding(embedding,hashmap_str2inx(hmap,"man",0));
    float* woman_vec = word_embedding(embedding,hashmap_str2inx(hmap,"woman",0));
    float* king_vec = word_embedding(embedding,hashmap_str2inx(hmap,"king",0));
    float* queen_vec = word_embedding(embedding,hashmap_str2inx(hmap,"queen",0));

    printf("Similarity of 'man' and 'woman' embeding vector:  %7.4f\n",
                        cosine_similarity(man_vec,woman_vec,embedding_dim));
    printf("Similarity of 'king' and 'queen' embeding vector: %7.4f\n",
                        cosine_similarity(king_vec,queen_vec,embedding_dim));

    const char* test_word[] = {
        "king","man", "warrior",
        "queen","woman", "potter"
    };
    int test_words_cnt = sizeof(test_word) / sizeof(test_word[0]);
    float* test_vec[test_words_cnt];
    for (int i = 0; i < test_words_cnt; i++) {
        int wrdinx = hashmap_str2inx(hmap,test_word[i],0);
        test_vec[i] = word_embedding(embedding,wrdinx);
    }
    /* Reference:
     * Linguistic Regularities in Continuous Space Word Representations
     * https://aclanthology.org/N13-1090.pdf
     */
    float female_king_vec[embedding_dim];
    for (int i = 0; i < embedding_dim; i++)
        female_king_vec[i] = king_vec[i] - man_vec[i] + woman_vec[i];

    WRDSIM tstwrdsim[test_words_cnt];
    for (int i = 0; i < test_words_cnt; i++) {
        tstwrdsim[i].wrdinx = hashmap_str2inx(hmap,test_word[i],0);
        tstwrdsim[i].cossim = cosine_similarity(
                                   female_king_vec,test_vec[i],embedding_dim);
    }
    qsort(tstwrdsim,test_words_cnt,sizeof(WRDSIM),qsort_compare);
    printf("\nSimilarity of test words to king - man + woman\n");
    for (int i = 0; i < test_words_cnt; i++) {
        int wrdinx = tstwrdsim[i].wrdinx;
        const char* wrdstr = hashmap_inx2str(hmap,wrdinx);
        float cossim = tstwrdsim[i].cossim;
        printf("%10s %7.4f\n",wrdstr,cossim);
    }

#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        int nsamples = test_words_cnt + 1;
        float x[nsamples][embedding_dim];
        int rdim = 2;
        float r[nsamples][rdim];
        const char* vector_names[nsamples];

        /* test words embeddings + one entry for  king - man + woman */
        for (int i = 0; i < nsamples - 1; i++) {
            fltcpy(x[i],test_vec[i],embedding_dim);
            vector_names[i] = test_word[i];
        }
        /* add king - man + woman vector; it's index == test_words_cnt */
        fltcpy(x[nsamples - 1],female_king_vec,embedding_dim);
        vector_names[nsamples - 1] = "king - man + woman";

        float mean[embedding_dim];
        float sdev[embedding_dim];
        calculate_mean_sdev(x,nsamples,embedding_dim,mean,sdev,0);
        for (int i = 0; i < embedding_dim; i++) sdev[i] = 1;
        normalize(x,nsamples,embedding_dim,mean,sdev,0);

        PCA(x,r,nsamples,embedding_dim,rdim);
        plot_embeddings(r,nsamples,vector_names,"PCA of word embeddings");
    }
#endif
    (void) losses;
    freemem(dy[0]);
    freemem(dy[1]);
    freemem(gWx[0]);
    freemem(gWx[1]);
    embedding_free(embedding);
    dense_free(dense);
    hashmap_free(hmap);
    return 0;
}

int main()
{
    run_tests();
    run_demo();
    return 0;
}
