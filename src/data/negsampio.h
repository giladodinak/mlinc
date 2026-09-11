/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store NN negative-sampling layer */
#ifndef NEGSAMPIO_H
#define NEGSAMPIO_H
#include <stdio.h>
#include "negsample.h"

/* read_negsample - Read a negative-sampling layer from a file
 *
 * Reads a negative-sampling layer from the file pointed to by fp.
 *
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 *
 * Returns:
 *   Pointer to the read layer if successful, NULL otherwise
 */
NEGSAMPLE* read_negsample(FILE* fp);

/* write_negsample - Write a negative-sampling layer to a file
 *
 * Writes the negative-sampling layer pointed to by l to the file pointed
 * to by fp.
 *
 * Parameters:
 *   l     - Pointer to the layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_negsample(const NEGSAMPLE* l, int final, FILE* fp);

/* load_negsample - Load a negative-sampling layer from a file
 *
 * Opens the file specified by the filename parameter for reading and
 * loads a negative-sampling layer from it.
 *
 * Parameters:
 *   filename - Name of the file to load the layer from
 *
 * Returns:
 *   Pointer to the loaded layer if successful, NULL otherwise
 */
NEGSAMPLE* load_negsample(const char* filename);

/* store_negsample - Store a negative-sampling layer into a file
 *
 * Opens the file specified by the filename parameter for writing and
 * stores the negative-sampling layer pointed to by l into it.
 *
 * Parameters:
 *   l        - Pointer to the layer to be stored
 *   filename - Name of the file to store the layer in
 *
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_negsample(const NEGSAMPLE* l, const char* filename);

#endif
