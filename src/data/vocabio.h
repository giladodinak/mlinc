/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store a vocabulary */
#ifndef VOCABIO_H
#define VOCABIO_H
#include <stdio.h>
#include "vocab.h"

/* read_vocab - Read a vocabulary from a file
 *
 * Reads a vocabulary from the file pointed to by fp. Frequency and negative-
 * sampling distribution tables are loaded only when present in the file.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read vocabulary if successful, NULL otherwise
 */
VOCAB* read_vocab(FILE* fp);

/* write_vocab - Write a vocabulary to a file
 *
 * Writes the vocabulary pointed to by v to the file pointed to by fp.
 * Frequency and negative-sampling distribution tables are written only when
 * they exist.
 *
 * Parameters:
 *   v  - Pointer to the vocabulary to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_vocab(const VOCAB* v, FILE* fp);

/* load_vocab - Load a vocabulary from a file
 *
 * Opens the file specified by the filename parameter for reading and loads a
 * vocabulary from it.
 *
 * Parameters:
 *   filename - Name of the file to load the vocabulary from
 *
 * Returns:
 *   Pointer to the loaded vocabulary if successful, NULL otherwise
 */
VOCAB* load_vocab(const char* filename);

/* store_vocab - Store a vocabulary into a file
 *
 * Opens the file specified by the filename parameter for writing and stores
 * the vocabulary pointed to by v into it.
 *
 * Parameters:
 *   v        - Pointer to the vocabulary to be stored
 *   filename - Name of the file to store the vocabulary in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_vocab(const VOCAB* v, const char* filename);

#endif
