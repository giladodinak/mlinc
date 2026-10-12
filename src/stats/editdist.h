/* Copyright (c) 2023-2024 Gilad Odinak */
/* Levenshtein edit distance function   */
#ifndef EDITDIST_H
#define EDITDIST_H

#ifdef __cplusplus
extern "C" {
#endif

/* Computes the Levenshtein distance between two sequences.
 * 
 * Calculates the minimum number of single-token edits (insertions, 
 * deletions, or substitutions) required to change one sequence into the other.
 *
 * Parameters:
 *   p - a sequence of token predictions
 *   m - length of sequence pointed by p
 *   t - a sequence of true tokens
 *   n - length of sequence pointed by t
 *
 * Returns:
 *   The edit distance between the two sequences.
 *
 * Reference:
 *   https://en.wikipedia.org/wiki/Levenshtein_distance
 */
int edit_dist(const int* p, int n, const int* t, int m);

#ifdef __cplusplus
}
#endif

#endif
