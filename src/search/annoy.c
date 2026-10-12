/* Copyright (c) 2023-2024 Gilad Odinak                   */
/* Approximate Nearest Neighbors structures and functions */
/* Implements Erik Bernhardsson's "Approximate Nearset Neighbors Oh Yeah"
 * Reference:
 * https://erikbern.com/2015/09/24/nearest-neighbor-methods-vector-models-part-1
 * https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
 */ 
#include <stdio.h>
#include <stdlib.h>  /* realloc() */
#include <memory.h>  /* memset()  */
#include "mem.h"
#include "float.h"
#include "arraytypes.h"
#include "norm.h"
#include "cossim.h"
#include "random.h"
#include "annoy.h"

/* Used internally to search the tree */
typedef struct cossim_s {
    int data_ix;  /* Index of a data vector */
    float cossim; /* Cosine similarity of this vector to the query vector */
} COSSIM;

static ANNOY_NODE_IX build_tree(ANNOY* annoy, int* data_ix, int nvec);
static int search_tree(ANNOY* annoy, ANNOY_NODE_IX node_ix, 
                       const fVec query, float search_q, 
                       COSSIM* similar, int size, int cnt);
static void similarity_sort(COSSIM* sim, int cnt, int dir);
                            

/* Creates an Approximate Nearest Neighbors search tree.
 *
 * Parameters:
 *   data      - An array of vectors to be searched.
 *   num_vec   - Number of data vectors.
 *   vec_dim   - Vectors' dimension, number of elements in each vector.
 *   num_trees - Number of search trees.
 *
 * Returns:
 *   A pointer to Annoy search tree.
 *
 * Notes:
 *   Annoy only keeps a reference to the data, it does not make a copy of it.
 *   The data used by Annoy must not change until after calling annoy_free().
 *   Internally Annoy uses the row indices of data to retrieve vectors.
 *   Annoy uses vectors cosine similarity to compare vectors.
 */
ANNOY* annoy_create(const fArr2D data, int num_vec, int vec_dim, int num_trees)
{
    int leaf_size = ANNOY_LDS;
    ANNOY* annoy = allocmem(1,1,ANNOY);
    annoy->data = data;
    annoy->num_vec = num_vec;
    annoy->vec_dim = vec_dim;
    annoy->num_trees = num_trees;
    annoy->root = allocmem(num_trees,1,ANNOY_NODE_IX);
    annoy->leaf_size = leaf_size;
    annoy->num_nodes = num_trees * ((num_vec / leaf_size + 1) * 2 + 1);
    annoy->num_used = 0;
    annoy->hpv_cnt = 0;
    annoy->nodes = allocmem(annoy->num_nodes,1,ANNOY_NODE);
    int* data_ix = allocmem(1,num_vec,int);
    for (int i = 0; i < num_vec; i++)
        data_ix[i] = i;
    for (int i = 0; i < annoy->num_trees; i++)
        annoy->root[i] = build_tree(annoy,data_ix,num_vec);
    freemem(data_ix);
    return annoy;
}

/* Finds vectors most similar to a given vector.
 *
 * Parameters: 
 *   annoy      - Annoy search tree.
 *   query      - Vector to search for; its dimension must be the same 
 *                as the dimension of the data vectors used to create
 *                the Annoy search tree.
 *   search_k   - quality vs time; 0: best quality 1: fastest
 *   similar    - Pointer to integer array that receives indices of vectors
 *                in data array passed to annoy_create() that are most similar
 *                to the query vector, in descending similarity order.
 *   similarity - Pointer to optional float array that receives the cosine
 *                similarity values of the vectors, whose corresponding indices
 *                in data are returned in similar, and the query vector.
 *   topn       - The number of entries in similar and similarity arrays.
 *
 * Returns:
 *   The actual number of entries retruend similar, similarity arrays, 
 *   which may be less than topn.
 */
int annoy_most_similar(ANNOY* annoy, const fVec query, float search_q,
                                     int* similar, float* similarity, int topn)
{
    int search_k = annoy->num_trees * topn;
    COSSIM* sim = allocmem(1,search_k,COSSIM);
    annoy->cos_sim_cnt = 0;
    int cnt = 0;
    for (int i = 0; i < annoy->num_trees; i++)
        cnt = search_tree(annoy,annoy->root[i],query,search_q,sim,search_k,cnt);
    
    /* Sort in descending order */
    similarity_sort(sim,cnt,1);
    
    /* Remove duplicates */
    int i, j = 0;
    for (i = 1; i < cnt; i++)
        if (sim[i].data_ix != sim[j].data_ix)
            sim[++j] = sim[i];
    cnt = j + 1;

    for (i = 0; i < cnt && i < topn; i++)
        similar[i] = sim[i].data_ix;
    if (similarity != NULL)
        for (i = 0; i < cnt && i < topn; i++)
            similarity[i] = sim[i].cossim;
    freemem(sim);
    return i;
}

/* Frees the memory allocated by annoy_create()
 */
void annoy_free(ANNOY* annoy)
{
    freemem(annoy->nodes);
    freemem(annoy->root);
    freemem(annoy);
}

static void add_nodes(ANNOY* annoy)
{
    int new_num = annoy->num_nodes + 
              annoy->num_trees * ((annoy->num_vec / annoy->leaf_size + 1) + 1);
    ANNOY_NODE* nodes = realloc(annoy->nodes,new_num * sizeof(ANNOY_NODE));
    if (nodes == NULL) {
        fflush(stdout);
        fprintf(stderr,"\nIn function '%s': "
                "out of memory at file '%s' line %d\n",
                __FUNCTION__,__FILE__,__LINE__);
        exit(-1);
    }
    memset(nodes + annoy->num_nodes,0,
           (new_num - annoy->num_nodes) * sizeof(ANNOY_NODE));
    annoy->nodes = nodes;
    annoy->num_nodes = new_num;
}

static inline ANNOY_NODE_IX alloc_node(ANNOY* annoy) 
{
    if (annoy->num_used >= annoy->num_nodes)
        add_nodes(annoy);
    return annoy->num_used++;
}

/* Returns a vector in mpv, that is the midpoint between
 * vectors v1, v2 whose dimension is dim 
 */
static inline void midpoint(
    const fVec restrict v1,
    const fVec restrict v2,
    fVec restrict mpv,
    int dim)
{
    for (int i = 0; i < dim; i++)
        mpv[i] = v1[i] + v2[i];
    for (int i = 0; i < dim; i++)
        mpv[i] /= 2;
}

/* Returns a vector in hpv, that is orthogonal to a hyperplane that is
 * equidistant between the vectors v1, v2 whose dimension is dim 
 */
static inline void hyperplane(
    const fVec restrict v1,
    const fVec restrict v2,
    fVec restrict hpv,
    int dim)
{
    for (int i = 0; i < dim; i++)
        hpv[i] = v2[i] - v1[i];
    float n = vecnorm(hpv,dim);
    for (int i = 0; i < dim; i++)
        hpv[i] /= n;
}

/* Calculates the distance of a data point from hypeplane that 
 * passes through a midpoint.
 *
 * Parameters:
 *   vec - data point vector
 *   mpv - midpoint vector
 *   hpv - a vector orthogonal to the hyperplane.
 *   dim - the dimension of the vectors
 *
 * Returns:
 * Signed distance, positive value is 'right' of the hyperplane
 *
 * Notice that instead of calculating  the distance relative to
 * a hyperplane that passes through the midpoint, the calculation
 * is relative to a parallel hyperplane that passes through the origin.
 */ 
static inline float project(
    const fVec restrict vec,
    const fVec restrict mpv,
    const fVec restrict hpv,
    int dim)
{
    float dist = 0;
    for (int j = 0; j < dim; j++)
         dist += (vec[j] - mpv[j]) * hpv[j];
    return dist;
}

/* Builds a search (sub)tree for the data vectors whose indices are passed
 * in the array pointed by data_ix. The number of entries in the array is
 * passed in nvec. Returns the index of the tree's root node.
 */
static ANNOY_NODE_IX build_tree(ANNOY* annoy, int* data_ix, int nvec)
{
    int D = annoy->vec_dim;
    typedef float (*ArrND)[D];
    ArrND restrict data = (ArrND) annoy->data;

    /* alloc_node may changes annoy->nodes */
    ANNOY_NODE_IX node_ix = alloc_node(annoy);
    /* obtain a pointer to the new node _after_ alloc_node is called */
    ANNOY_NODE* restrict node = annoy->nodes + node_ix;

    node->vcnt = nvec;
    if (nvec <= annoy->leaf_size) { 
        /* Remaining data fits in a node => it's a leaf */
        for (int i = 0; i < nvec; i++)
            node->data[i] = data_ix[i];
        return node_ix;
    }
    fVec hpv = allocmem(1,D,float);
    fVec mpv = allocmem(1,D,float);
    int* left = allocmem(1,nvec,int);
    int lcnt = 0;
    int* right = allocmem(1,nvec,int);
    int rcnt = 0;
    int idx[2];
    idx[0] = idx[1] = (int) urand(0,nvec);
    idx[1] = (int) urand(0,nvec - 1);
    if (idx[1] >= idx[0])
        idx[1]++;
    node->split[0] = data_ix[idx[0]];
    node->split[1] = data_ix[idx[1]];
    hyperplane(data[node->split[0]],data[node->split[1]],hpv,D);
    midpoint(data[node->split[0]],data[node->split[1]],mpv,D);
    for (int i = 0; i < nvec; i++) {
        float dist = project(data[data_ix[i]],mpv,hpv,D);
        if (dist > 0)
            right[rcnt++] = data_ix[i];
        else
            left[lcnt++] = data_ix[i];
    }
    /* recursive build_tree may changes annoy->nodes */
    ANNOY_NODE_IX tmp_left = build_tree(annoy,left,lcnt);
    node = annoy->nodes + node_ix;
    node->left = tmp_left;
    /* recursive build_tree may changes annoy->nodes */
    ANNOY_NODE_IX tmp_right = build_tree(annoy,right,rcnt);
    node = annoy->nodes + node_ix;
    node->right = tmp_right;
    freemem(right);
    freemem(left);
    freemem(mpv);
    freemem(hpv);
    annoy->hpv_cnt++;
    return node_ix;
}

/* Searches (sub)tree for data vectors most similar to the query vector.
 *
 * Parameters:
 *   node_ix - The index of a node that is the root of the (sub)tree.
 *   query   - Target vector, find data vectors most similar to it.
 *   similar - Array that receives tuples of the indices of the similar
 *             vectors and the cosine similarity to the query vector.
 *   size    - Number of entries in similar array.
 *   cnt     - Number of entries in similar array that are already filled.
 *
 * Returns:
 *   The updated number of entries in similar array that are filled.
 */
static int search_tree(ANNOY* annoy, ANNOY_NODE_IX node_ix, 
                       const fVec query, float search_q, 
                       COSSIM* similar, int size, int cnt)
{
    int D = annoy->vec_dim;
    typedef float (*ArrND)[D];
    ArrND restrict data = (ArrND) annoy->data;
    ANNOY_NODE* restrict node = annoy->nodes + node_ix;
    if (node->vcnt <= annoy->leaf_size) {
        for  (int i = 0; i < node->vcnt; i++) {
            int data_ix = node->data[i];
            float cossim = cosine_similarity(query,data[data_ix],D);
            annoy->cos_sim_cnt++;
            if (cnt < size) {
                similar[cnt].data_ix = data_ix;
                similar[cnt].cossim = cossim;
                cnt++;
                similarity_sort(similar,cnt,0); /* Sort in ascending order */
            }
            else
            if (cossim > similar[0].cossim) {
                similar[0].data_ix = data_ix;
                similar[0].cossim = cossim;
                similarity_sort(similar,cnt,0); /* Sort in ascending order */
            }
        }
        return cnt;
    }
    fVec hpv = allocmem(1,D,float);
    hyperplane(data[node->split[0]],data[node->split[1]],hpv,D);
    fVec mpv = allocmem(1,D,float);
    midpoint(data[node->split[0]],data[node->split[1]],mpv,D);
    float query_dist = project(query,mpv,hpv,D);
    freemem(mpv);
    ANNOY_NODE_IX nearer, farther;
    if (query_dist > 0) {
        nearer = node->right;
        farther = node->left;
    }
    else {
        nearer = node->left;
        farther = node->right;
    }
    cnt = search_tree(annoy,nearer,query,search_q,similar,size,cnt);
    if (cnt < size) /* Not enough vectors, get more from far side */
        cnt = search_tree(annoy,farther,query,search_q,similar,size,cnt);
    else { /* Enough vectors, still try to get better (more similar) ones */
        /* Find distance of farthest similar data point vector */
        float maxdist = 0;
        for (int i = 0; i < cnt; i++) {
            float dist = project(query,data[similar[i].data_ix],hpv,D);
            dist = fabsf(dist);
            if (dist > maxdist)
                maxdist = dist;
        }
        if (fabsf(query_dist) < maxdist * search_q)
            cnt = search_tree(annoy,farther,query,search_q,similar,size,cnt);
    }
    freemem(hpv);
    return cnt;
}

/* Sorts by similarity, then by vector index in ascending order */
static int qsort_compare_asc(const void *a_, const void *b_)
{
    COSSIM* a = (COSSIM*)a_;
    COSSIM* b = (COSSIM*)b_;
    if (a->cossim < b->cossim) return -1;
    if (a->cossim > b->cossim) return 1;
    if (a->data_ix < b->data_ix) return -1;
    if (a->data_ix > b->data_ix) return 1;
    return 0;
}

/* Sorts by similarity, then by vector index in descending order */
static int qsort_compare_desc(const void *a_, const void *b_)
{
    COSSIM* a = (COSSIM*)a_;
    COSSIM* b = (COSSIM*)b_;
    if (a->cossim < b->cossim) return 1;
    if (a->cossim > b->cossim) return -1;
    if (a->data_ix < b->data_ix) return 1;
    if (a->data_ix > b->data_ix) return -1;
    return 0;
}

/* Sorts entries by similarity order, then by vector index order.
 * In the sorted array duplicate entries will be adjecant, so they 
 * can be easily removed.
 */
static void similarity_sort(COSSIM* sim, int cnt, int dir)
{
    qsort(sim,cnt,sizeof(COSSIM),((dir)?qsort_compare_desc:qsort_compare_asc));
}

