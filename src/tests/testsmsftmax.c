/* Copyright (c) 2026 Gilad Odinak    */
/* Gradient check for smsftmax_loss   */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "random.h"
#include "smsftmax.h"


/* Fixed seed used before every loss evaluation so the sampled negatives
   and the positive-rejection draws are identical across evaluations. 
 */
#define RNG_SEED   12345

/* Recompute the per-batch loss the way smsftmax_loss does, but with the
 * per-position weight FROZEN to the supplied values (one per row). This is
 * the reference the analytic gradient is checked against. The negatives are
 * regenerated here from the same RNG state, so the caller must init_lrng()
 * to the same seed immediately before calling this. Rows whose target is PAD
 * / out of range are skipped exactly as in smsftmax_loss, and contribute a
 * frozen weight of 0 (unused).
 *
 * w_frozen[i] must hold the weight smsftmax_loss used for row i at the
 * unperturbed parameters.
 */
static double reference_loss_frozen_w(SMSFTMAX* l,
                                      const float* Wo_flat,   /* [K*E] live */
                                      const float* h_flat,    /* [B*E] live */
                                      const float* labels,
                                      int cnt,
                                      const double* w_frozen)
{
    const int E = l->E, K = l->K, k = l->n_neg;
    typedef const float (*ArrKE)[E];
    typedef const float (*ArrBE)[E];
    ArrKE Wo = (ArrKE) Wo_flat;
    ArrBE h  = (ArrBE) h_flat;

    int cand[k];
    double loss = 0.0;
    for (int i = 0; i < cnt; i++) {
        int target = (int) labels[i];
        if (target <= 0 || target >= K)
            continue;

        /* Same draw sequence as smsftmax_loss: k negatives, reject target. */
        for (int c = 0; c < k; c++) {
            int neg;
            do {
                neg = l->dist[(int) urand(0,l->dist_size)];
            } while (neg == target);
            cand[c] = neg;
        }

        double qtarget = exp((double) l->logq[target]);
        double log1mqt = log1p(-qtarget);

        double pos = 0.0;
        for (int j = 0; j < E; j++)
            pos += (double) Wo[target][j] * (double) h[i][j];

        double maxl = -INFINITY;
        double logit[k];
        for (int c = 0; c < k; c++) {
            int w = cand[c];
            double dot = 0.0;
            for (int j = 0; j < E; j++)
                dot += (double) Wo[w][j] * (double) h[i][j];
            logit[c] = dot - (double) l->logq[w] + log1mqt;
            if (logit[c] > maxl) maxl = logit[c];
        }
        double sum = 0.0;
        for (int c = 0; c < k; c++)
            sum += exp(logit[c] - maxl);
        double logzneg = maxl + log(sum);

        /* Frozen weight -> loss contribution w * (logzneg - pos). */
        loss += w_frozen[i] * (logzneg - pos);
    }
    return loss;
}

/* Recover, for each row, the weight smsftmax_loss would compute at the
 * current (unperturbed) parameters, so the reference can freeze it. Mirrors
 * the weight computation in smsftmax_loss exactly. Reseeds internally, so
 * caller must reseed afterwards before the analytic call.
 */
static void compute_weights(SMSFTMAX* l,
                            const float* Wo_flat, const float* h_flat,
                            const float* labels, int cnt, double* w_out)
{
    const int E = l->E, K = l->K, k = l->n_neg;
    typedef const float (*ArrKE)[E];
    typedef const float (*ArrBE)[E];
    ArrKE Wo = (ArrKE) Wo_flat;
    ArrBE h  = (ArrBE) h_flat;
    const double lnk = log((double) k);

    int cand[k];
    for (int i = 0; i < cnt; i++) {
        w_out[i] = 0.0;
        int target = (int) labels[i];
        if (target <= 0 || target >= K)
            continue;
        for (int c = 0; c < k; c++) {
            int neg;
            do { neg = l->dist[(int) urand(0,l->dist_size)]; }
            while (neg == target);
            cand[c] = neg;
        }
        double qtarget = exp((double) l->logq[target]);
        double log1mqt = log1p(-qtarget);
        double pos = 0.0;
        for (int j = 0; j < E; j++)
            pos += (double) Wo[target][j] * (double) h[i][j];
        double maxl = -INFINITY, logit[k];
        for (int c = 0; c < k; c++) {
            int w = cand[c];
            double dot = 0.0;
            for (int j = 0; j < E; j++)
                dot += (double) Wo[w][j] * (double) h[i][j];
            logit[c] = dot - (double) l->logq[w] + log1mqt;
            if (logit[c] > maxl) maxl = logit[c];
        }
        double sum = 0.0;
        for (int c = 0; c < k; c++) sum += exp(logit[c] - maxl);
        double logzneg = maxl + log(sum);
        double logzest = logzneg - lnk;
        double d = pos - logzest;
        double weight = (d >= 0.0) ? (exp(-d)/(1.0+exp(-d)))
                                   : (1.0/(1.0+exp(d)));
        w_out[i] = weight;
    }
}

int main(void)
{
    const int K = 40;      /* vocab                                   */
    const int E = 6;       /* embedding / model dim                   */
    const int B = 4;       /* batch rows                              */
    const int n_neg = 5;   /* negatives per position                  */
    const int cnt = B;
    const float eps = 1e-3;
    const double tol = 1e-4; /* rel-error tolerance (float storage)   */

    printf("\nGradient check for smsftmax_loss\n");
    init_lrng(RNG_SEED);     /* for building Wo (nrand) and data        */

    SMSFTMAX* l = smsftmax_create(K,n_neg);
    smsftmax_init(l,E,B);
    
    /* Build a dist table with non-uniform frequencies, excluding PAD (0).
     * Word c (1..K-1) appears c times -> varied q_c. 
     */
    int dist_size = 0;
    for (int c = 1; c < K; c++) dist_size += c;
    int* dist = allocmem(dist_size,1,int);
    int p = 0; 
    for (int c = 1; c < K; c++) 
        for (int t = 0; t < c; t++) 
            dist[p++] = c;
    smsftmax_set_dist(l,dist,dist_size);

    /* Inputs h [B][E] and labels [B][1], targets in 1..K-1 */
    float* h = allocmem(B,E,float);
    for (int i = 0; i < B*E; i++) h[i] = nrand(0.0f,1.0f);
    float* labels = allocmem(B,1,float);
    labels[0] = 3; labels[1] = 17; labels[2] = 29; labels[3] = 8;

    /* Analytic gradients from the layer. gWo is [K][E]; dh is [B][E] */
    float* gWo = allocmem(K,E,float);
    float* dh  = allocmem(B,E,float);

    /* Access Wo as a flat live buffer for perturbation */
    float* Wo = (float*) l->Wo;

    /* Freeze the per-row weights at the unperturbed point */
    double* wfz = allocmem(B,1,double);
    init_lrng(RNG_SEED);
    compute_weights(l,Wo,h,labels,cnt,wfz);

    /* Analytic pass: reseed so the negatives match the reference draws */
    init_lrng(RNG_SEED);
    smsftmax_loss(l,(fArr2D)h,(fArr2D)labels,(fArr2D)gWo,(fArr2D)dh,cnt,NULL);

    /* Check gWo */
    double max_rel_gWo = 0.0; int bad_gWo = 0;
    /* Only rows that were touched have meaningful analytic gradient; but
     * untouched rows must have gWo == 0. Check a representative subset:
     * every entry of the target rows and the drawn-negative rows is covered
     * by perturbing all K*E is expensive but K,E are tiny here, so do all.
     */
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < E; b++) {
            int idx = a*E + b;
            float orig = Wo[idx];

            Wo[idx] = orig + eps;
            init_lrng(RNG_SEED);
            double Lp = reference_loss_frozen_w(l,Wo,h,labels,cnt,wfz);

            Wo[idx] = orig - eps;
            init_lrng(RNG_SEED);
            double Lm = reference_loss_frozen_w(l,Wo,h,labels,cnt,wfz);

            Wo[idx] = orig;
            double num = (Lp - Lm) / (2.0*eps);
            double ana = (double) ((float(*)[E])gWo)[a][b];

            double denom = fmax(1.0, fmax(fabs(num), fabs(ana)));
            double rel = fabs(num - ana) / denom;
            if (rel > max_rel_gWo) max_rel_gWo = rel;
            if (rel > tol) {
                bad_gWo++;
                if (bad_gWo <= 10)
                    printf("gWo[%d][%d]: analytic % .6e  fd % .6e  rel %.3e\n",
                           a,b,ana,num,rel);
            }
        }
    }

    /* Check dh */
    double max_rel_dh = 0.0;
    int bad_dh = 0;
    for (int i = 0; i < B; i++) {
        for (int b = 0; b < E; b++) {
            int idx = i*E + b;
            float orig = h[idx];

            h[idx] = orig + eps;
            init_lrng(RNG_SEED);
            double Lp = reference_loss_frozen_w(l,Wo,h,labels,cnt,wfz);

            h[idx] = orig - eps;
            init_lrng(RNG_SEED);
            double Lm = reference_loss_frozen_w(l,Wo,h,labels,cnt,wfz);

            h[idx] = orig;
            double num = (Lp - Lm) / (2.0*eps);
            double ana = (double) ((float(*)[E])dh)[i][b];

            double denom = fmax(1.0, fmax(fabs(num), fabs(ana)));
            double rel = fabs(num - ana) / denom;
            if (rel > max_rel_dh) max_rel_dh = rel;
            if (rel > tol) {
                bad_dh++;
                if (bad_dh <= 10)
                    printf("dh[%d][%d]:  analytic % .6e  fd % .6e  rel %.3e\n",
                           i,b,ana,num,rel);
            }
        }
    }

    printf("max rel error  gWo = %.3e   dh = %.3e   (tol %.1e)\n",
           max_rel_gWo,max_rel_dh,tol);
    int pass = (bad_gWo == 0 && bad_dh == 0);
    printf("%s  (gWo fails %d, dh fails %d)\n",
           pass ? "PASS" : "FAIL", bad_gWo, bad_dh);

    freemem(wfz); freemem(dh); freemem(gWo);
    freemem(labels); freemem(h); freemem(dist);
    smsftmax_free(l);
    return pass ? 0 : 1;
}
