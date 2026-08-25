/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN sampled-softmax layer */
#ifndef SMSFTMAXIO_H
#define SMSFTMAXIO_H
#include <stdio.h>
#include "smsftmax.h"

/* read_smsftmax - Read a sampled-softmax layer from a file
 *
 * Reads a sampled-softmax layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read layer if successful, NULL otherwise
 */
SMSFTMAX* read_smsftmax(FILE* fp);

/* write_smsftmax - Write a sampled-softmax layer to a file
 *
 * Writes the sampled-softmax layer pointed to by l to the file pointed
 * to by fp.
 *
 * Parameters:
 *   l  - Pointer to the layer to be written
 *   fp - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_smsftmax(const SMSFTMAX* l, FILE* fp);

/* load_smsftmax - Load a sampled-softmax layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a sampled-softmax layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the layer from
 *
 * Returns:
 *   Pointer to the loaded layer if successful, NULL otherwise
 */
SMSFTMAX* load_smsftmax(const char* filename);

/* store_smsftmax - Store a sampled-softmax layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the sampled-softmax layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the layer to be stored
 *   filename - Name of the file to store the layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_smsftmax(const SMSFTMAX* l, const char* filename);

#endif
