/* Copyright (c) 2026 Gilad Odinak */
/*
 * This file includes some basic tests for the LMEMB (language-model token
 * embedding) layer, the decoder-only-transformer counterpart of the word2vec
 * EMBEDDING layer. Unlike EMBEDDING, which sums M context vectors into one
 * output row, LMEMB gathers one embedding row per token position:
 *   ids[B*T] -> h[B*T][E],  h[t] = Wx[ids[t]]
 * with a sparse, stamp-tracked scatter-add backward.
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
#include "lmemb.h"

#define EPS 1e-3
#define TOL 1e-2

/* Test dimensions */
#define VOCAB 7   /* V: vocabulary size (index 0 is the pad row) */
#define EMBD  4   /* E: embedding dimension                      */
#define STEPS 3   /* T: sequence length                          */
#define BATCH 2   /* B: number of sequences in a batch           */
#define PAD   0   /* pad index                                   */
#define NBT   (BATCH * STEPS)

int failures = 0;

/* Builds and initializes a fresh LMEMB layer with the test dimensions.
 * training = 1, tied = 0 (owns its own weights and gradients).
 */
static LMEMB* make_lmemb(void)
{
    LMEMB* l = lmemb_create(EMBD,STEPS,PAD);
    lmemb_init(l,VOCAB,BATCH,1,0);
    return l;
}

/* Fills a [B*T] index array with random token indices in [0, V).
 * Includes the pad index, so pad handling is exercised.
 */
static void fill_random_ids(int* ids, int bt, int V)
{
    for (int i = 0; i < bt; i++)
        ids[i] = (int) urand(0.0,(float) V);
}

/* Scalar loss L = sum_{i,k} dy[i][k] * h[i][k], where h is the forward output.
 * A random dy (rather than all ones) makes the gradient check sensitive to
 * per-row/column mistakes, not just to row sums.
 */
static float lmemb_loss(LMEMB* l, const int* ids, fArr2D dy_)
{
    typedef float (*ArrBTE)[l->E];
    ArrBTE h  = (ArrBTE) lmemb_forward(l, ids, 0);
    ArrBTE dy = (ArrBTE) dy_;
    float L = 0;
    for (int i = 0; i < l->BT; i++)
        for (int k = 0; k < l->E; k++)
            L += dy[i][k] * h[i][k];
    return L;
}

/* With all weights zero, the forward output must be zero. */
static void test_lmemb_zero_forward(LMEMB* l)
{
    printf("Test: lmemb zero forward\n");

    int ids[NBT];
    fill_random_ids(ids, l->BT, l->V);
    fltclr(l->Wx, l->V * l->E);

    typedef float (*ArrBTE)[l->E];
    ArrBTE h = (ArrBTE) lmemb_forward(l, ids, 0);

    for (int i = 0; i < l->BT; i++)
        for (int k = 0; k < l->E; k++)
            if (fabsf(h[i][k]) > 1e-9f) {
                printf("FAIL: h[%d][%d]=%g, expected 0\n", i, k, h[i][k]);
                failures++;
                return;
            }
    printf("PASS\n");
}

/* With Wx[w][k] = w (each row equals its own index), the output row for each
 * token must equal that token's index (a pure gather, no summing). Also
 * confirms the pad row (index 0, value 0) gathers zeros.
 */
static void test_lmemb_forward_gather(LMEMB* l)
{
    printf("Test: lmemb forward gather\n");

    typedef float (*ArrVE)[l->E];
    typedef float (*ArrBTE)[l->E];

    ArrVE Wx = (ArrVE) l->Wx;
    for (int w = 0; w < l->V; w++)
        for (int k = 0; k < l->E; k++)
            Wx[w][k] = (float) w;

    int ids[NBT];
    for (int i = 0; i < l->BT; i++)
        ids[i] = i % l->V;          /* deterministic indices */

    ArrBTE h = (ArrBTE) lmemb_forward(l, ids, 0);

    for (int i = 0; i < l->BT; i++) {
        float expect = (float) ids[i];   /* row value == index */
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
static void test_lmemb_finite_diff(LMEMB* l)
{
    printf("Test: lmemb finite-difference gradients (gWx)\n");

    typedef float (*ArrVE)[l->E];

    int   ids[NBT];
    float dy[NBT][EMBD];

    ArrVE Wx = (ArrVE) l->Wx;
    ArrVE g  = (ArrVE) l->gWx;    /* the layer owns gWx (untied) */

    fill_random_ids(ids, l->BT, l->V);
    for (int i = 0; i < l->BT; i++)
        for (int k = 0; k < l->E; k++)
            dy[i][k] = urand(-1.0, 1.0);

    /* Analytic gradient. backward does not read Wx, so later perturbing Wx
     * for the finite difference leaves g untouched. */
    lmemb_forward(l, ids, 0);
    lmemb_backward(l, (fArr2D) dy, 0);

    /* Pad row must have zero gradient. */
    for (int k = 0; k < l->E; k++)
        if (fabsf(g[PAD][k]) > 1e-9f) {
            printf("FAIL: gWx[pad=%d][%d]=%g, expected 0\n", PAD, k, g[PAD][k]);
            failures++;
            return;
        }

    /* Finite-difference check for every non-pad weight. */
    for (int w = 0; w < l->V; w++) {
        if (w == l->padinx)
            continue;
        for (int k = 0; k < l->E; k++) {
            float old = Wx[w][k];
            Wx[w][k] = old + EPS;
            float Lp = lmemb_loss(l, ids, (fArr2D) dy);
            Wx[w][k] = old - EPS;
            float Ln = lmemb_loss(l, ids, (fArr2D) dy);
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
    printf("PASS\n");
}

/* The sparse touched-row set must list exactly the distinct non-pad token
 * rows, with no duplicates, and the accumulated gradient must match a plain
 * dense reference computed independently.
 */
static void test_lmemb_sparse_rows(LMEMB* l)
{
    printf("Test: lmemb sparse touched rows and gradient\n");

    typedef float (*ArrVE)[l->E];

    int   ids[NBT];
    float dy[NBT][EMBD];

    fill_random_ids(ids, l->BT, l->V);
    for (int i = 0; i < l->BT; i++)
        for (int k = 0; k < l->E; k++)
            dy[i][k] = urand(-1.0, 1.0);

    /* Layer's sparse scatter-add. */
    lmemb_forward(l, ids, 0);
    lmemb_backward(l, (fArr2D) dy, 0);
    ArrVE g = (ArrVE) l->gWx;

    /* Independent dense reference: scatter dy into a plain [V][E] array. */
    fArr2D ref_ = allocmem(l->V, l->E, float);
    ArrVE ref = (ArrVE) ref_;
    fltclr(ref_, l->V * l->E);
    for (int i = 0; i < l->BT; i++) {
        int id = ids[i];
        if (id == l->padinx)
            continue;
        for (int k = 0; k < l->E; k++)
            ref[id][k] += dy[i][k];
    }

    /* Gradients must match on every row (including pad, which stays zero). */
    for (int w = 0; w < l->V; w++)
        for (int k = 0; k < l->E; k++)
            if (fabsf(g[w][k] - ref[w][k]) > 1e-6f) {
                printf("FAIL: gWx[%d][%d] layer=%g ref=%g\n",
                       w, k, g[w][k], ref[w][k]);
                failures++;
                freemem(ref_);
                return;
            }

    /* Expected distinct non-pad rows from ids. */
    int seen[VOCAB];
    for (int w = 0; w < l->V; w++)
        seen[w] = 0;
    int expect = 0;
    for (int i = 0; i < l->BT; i++) {
        int w = ids[i];
        if (w != l->padinx && !seen[w]) {
            seen[w] = 1;
            expect++;
        }
    }
    if (l->ntouched != expect) {
        printf("FAIL: ntouched=%d, expected %d distinct non-pad rows\n",
               l->ntouched, expect);
        failures++;
        freemem(ref_);
        return;
    }
    /* touched[] must contain exactly those rows, with no duplicates. */
    for (int r = 0; r < l->ntouched; r++) {
        int w = l->touched[r];
        if (w == l->padinx || w < 0 || w >= l->V || !seen[w]) {
            printf("FAIL: touched[%d]=%d is not a distinct non-pad row\n", r, w);
            failures++;
            freemem(ref_);
            return;
        }
        seen[w] = 0;  /* clear so a duplicate would be caught next time */
    }
    freemem(ref_);
    printf("PASS\n");
}

/* A token index appearing at several positions must accumulate its gradient
 * (scatter-add), and produce a single touched-row entry.
 */
static void test_lmemb_repeated_token(LMEMB* l)
{
    printf("Test: lmemb repeated-token accumulation\n");

    typedef float (*ArrVE)[l->E];

    int   ids[NBT];
    float dy[NBT][EMBD];

    /* All positions reference the same non-pad token. */
    int tok = 3;
    for (int i = 0; i < l->BT; i++)
        ids[i] = tok;
    for (int i = 0; i < l->BT; i++)
        for (int k = 0; k < l->E; k++)
            dy[i][k] = urand(-1.0, 1.0);

    lmemb_forward(l, ids, 0);
    lmemb_backward(l, (fArr2D) dy, 0);
    ArrVE g = (ArrVE) l->gWx;

    /* Exactly one touched row, and it is 'tok'. */
    if (l->ntouched != 1 || l->touched[0] != tok) {
        printf("FAIL: ntouched=%d touched[0]=%d, expected 1 and %d\n",
               l->ntouched, (l->ntouched > 0 ? l->touched[0] : -1), tok);
        failures++;
        return;
    }

    /* gWx[tok] must equal the column sum of dy over all positions. */
    for (int k = 0; k < l->E; k++) {
        float expect = 0;
        for (int i = 0; i < l->BT; i++)
            expect += dy[i][k];
        if (fabsf(g[tok][k] - expect) > 1e-5f) {
            printf("FAIL: gWx[%d][%d]=%g, expected %g\n",
                   tok, k, g[tok][k], expect);
            failures++;
            return;
        }
    }
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
    printf("vocab %d embd %d steps %d batch %d pad %d (BT %d)\n",
           VOCAB, EMBD, STEPS, BATCH, PAD, NBT);

    LMEMB* l;

    l = make_lmemb(); test_lmemb_zero_forward(l);    lmemb_free(l);
    l = make_lmemb(); test_lmemb_forward_gather(l);   lmemb_free(l);
    l = make_lmemb(); test_lmemb_finite_diff(l);      lmemb_free(l);
    l = make_lmemb(); test_lmemb_sparse_rows(l);      lmemb_free(l);
    l = make_lmemb(); test_lmemb_repeated_token(l);   lmemb_free(l);

    if (failures == 0)
        printf("\nALL TESTS PASSED\n");
    return 0;
}

int main(void)
{
    run_tests();
    return 0;
}
