/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read speech feature files                 */
#ifndef FEATFILE_H
#define FEATFILE_H

#define FEAT_CNT            14  /* LPC features per frame               */
#define EXPENDED_FEAT_CNT   70  /* LPC features with deltas             */
#define MAX_FEAT_CNT        64  /* Maximum features per frame           */
#define DELTA_EXPANSION      5  /* Values per feature after expansion   */

#define TIMIT_PHONEME_CNT   64
#define REDUCED_PHONEME_CNT 39

#define SIL 0                   /* Silence */
#define EOP REDUCED_PHONEME_CNT /* End of phoneme marker */

extern const char* timit_phoneme_names[TIMIT_PHONEME_CNT] /* = {
    "","aa","ae","ah","ao","aw","ax","axr",
    "ax-h","ay","b","bcl","ch","d","dcl","dh",
    "dx","eh","el","em","en","eng","er","ey",
    "f","g","gcl","h","hh","hv","ih","ix",
    "iy","jh","k","kcl","l","m","n","ng",
    "nx","ow","oy","p","pcl","q","r","s",
    "sh","t","tcl","th","uh","uw","ux","v",
    "w","wh","y","z","zh","pau","epi","h#"
}*/;

extern const char* reduced_phoneme_names[REDUCED_PHONEME_CNT] /* = {
    "sil","aa","ae","ah","aw","ay","b","ch",
    "d","dh","dx","eh","er","ey","f","g",
    "hh","ih","iy","jh","k","l","m","n",
    "ng","ow","oy","p","r","s","sh","t",
    "th","uh","uw","v","w","y","z"
}*/;

extern const int timit2reduced[TIMIT_PHONEME_CNT] /* = {
    0,  ""->sil     1,  aa->aa      2,  ae->ae      3,  ah->ah  
    1,  ao->aa      4,  aw->aw      3,  ax->ah     12,  axr->er  
    3,  ax-h->ah    5,  ay->ay      6,  b->b        0,  bcl->sil  
    7,  ch->ch      8,  d->d        0,  dcl->sil    9,  dh->dh  
   10,  dx->dx     11,  eh->eh     21,  el->l      22,  em->m  
   23,  en->n      24,  eng->ng    12,  er->er     13,  ey->ey  
   14,  f->f       15,  g->g        0,  gcl->sil   16,  h->hh   
   16,  hh->hh     16,  hv->hh     17,  ih->ih     17,  ix->ih  
   18,  iy->iy     19,  jh->jh     20,  k->k        0,  kcl->sil  
   21,  l->l       22,  m->m       23,  n->n       24,  ng->ng  
   23,  nx->n      25,  ow->ow     26,  oy->oy     27,  p->p  
    0,  pcl->sil    0,  q->sil     28,  r->r       29,  s->s  
   30,  sh->sh     31,  t->t        0,  tcl->sil   32,  th->th  
   33,  uh->uh     34,  uw->uw     34,  ux->uw     35,  v->v  
   36,  w->w       36,  wh->w      37,  y->y       38,  z->z  
   30,  zh->sh      0,  pau->sil    0,  epi->sil    0  h#->sil
}*/;

/* Reads speech samples from speech feature file and its label file.
 *
 * The feature file contains all the frames of one utterance, in time order,
 * one frame per text line: the frame index followed by the frame's features.
 * The label file contains the utterance's phonemes, in order, one phoneme
 * per text line: name, label, start and end time in seconds, start and end
 * position in (fractional) frames. Frame i is labeled with the phoneme
 * whose span contains i; phonemes that contain no frame are not assigned
 * to any frame, but are included in the returned phoneme sequence.
 *
 * The feature file contains F features per frame (F up to MAX_FEAT_CNT),
 * which are expanded to include deltas and delta-deltas of their values
 * (DELTA_EXPANSION * F values per row of x). F is determined by the file's
 * header line (see feature_dim()).
 *
 * Parameters
 *   fp   - A pointer to a feature file open for read.
 *   lfp  - A pointer to a label file open for read.
 *   maxs - The maximum number of samples (number of rows in x,
 *          and number of elements in yc)
 *   dim  - Number of values per row of x; must be DELTA_EXPANSION * F
 *   maxp - The maximum number of phonemes (number of elements in ph)
 *   ph   - If not NULL, updated with the utterance's phoneme labels
 *   phcnt- If not NULL, updated with the number of labels in ph
 *   
 * Returns:
 *   An array of feature vectors in x, their corresponding labels in yc.
 *   
 *   This function returns the actual number of samples.
 *   That also is the returned number of rows in x, elements in yc.
 *   Returns 0 if an error occured.
 *
 * Notes:
 *   The set of phonemes can be CMU 39 phonemes, or TIMIT 61 phonemes.
 *   In the later case TIMIT phonemes are mapped to their CMU equivalents.
 *   Reference: Speaker-Independent Phone Recognition, Lee and Hon 1989
 */
int read_feature_file(FILE* fp, FILE* lfp, int maxs, int dim,
                      float* x, int yc[],
                      int maxp, int ph[], int* phcnt);

/* Returns the number of values per row of x that read_feature_file()
 * produces for a feature file, based on the file's header line:
 * DELTA_EXPANSION * F for F features.
 * The file position is restored. Returns 0 if the header is invalid.
 */
int feature_dim(FILE* fp);

/* Reads speech samples from speech feature files.
 *
 * Each feature file contains all the frames of one utterance, one frame
 * per text line, and is accompanied by a label file containing the
 * utterance's phonemes (see read_feature_file()).
 *
 * Feature files contain features, which are expanded to include deltas
 * and delta-deltas of their values (see read_feature_file()). All files
 * must have the same number of features per frame.
 *
 * Parameters
 * - input_dir is a directory containing speech feature files, whose name
 *   ends with '.FEAT', and label files, whose name ends with '.LAB'.
 * - file_list is a text file containing the names of the files to be loaded.
 * - max_sequences is the maximum number of sequences (size of seq_length[])
 * - max_samples is the maximum number of samples (number of rows in x, and
 *   number of elements in yc)
 * - dim is the number of values per row of x (see feature_files_dim())
 * - max_phonemes is the maximum number of phonemes (number of elements
 *   in ph)
 * - ph, if not NULL, is updated with the phoneme labels of all sequences.
 * - ph_count, if not NULL, is updated with the number of phoneme labels
 *   of each sequence (same number of elements as seq_length).
 *   
 * Returns:
 *   An array of feature vectors in x, their corresponding labels in yc,
 *   and the length of each sequence of feature vectors in seq_length.
 *   
 *   This function returns the actual number of sequences.
 *   That also is the number of elements in seq_length.
 *   Returns 0 if an error occured.
 *
 *   The actual number of samples returned can be calculated by summing
 *   all values in seq_length.
 *
 * Notes:
 *   The set of phonemes can be CMU 39 phonemes, or TIMIT 61 phonemes.
 *   In the later case TIMIT phonemes are mapped to their CMU equivalents.
 *   Reference: Speaker-Independent Phone Recognition, Lee and Hon 1989
 */
int read_feature_files(const char* input_dir, const char* file_list,
                       int max_sequences, int *seq_length,
                       int max_samples, int dim, float* x, int yc[],
                       int max_phonemes, int ph[], int ph_count[]);

/* Returns the number of values per row of x that read_feature_files()
 * produces for the files in file_list, based on the first file's header
 * line (see feature_dim()). Returns 0 if the file cannot be read.
 */
int feature_files_dim(const char* input_dir, const char* file_list);

#endif
