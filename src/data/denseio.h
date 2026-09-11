/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store NN dense layer */
#ifndef DENSEIO_H
#define DENSEIO_H
#include <stdio.h>
#include "dense.h"

/* read_dense - Read a dense layer from a file
 * 
 * Reads a dense layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read dense layer if successful, NULL otherwise
 */
DENSE* read_dense(FILE* fp);

/* write_dense - Write a dense layer to a file
 * 
 * Writes the dense layer pointed to by d to the file pointed to by fp.
 * 
 * Parameters:
 *   d     - Pointer to the dense layer to be written
 *   final - If not zero, record the layer as inference-only
 *   fp    - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_dense(const DENSE* d, int final, FILE* fp);

/* load_dense - Load a dense layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads a dense layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the dense layer from
 * 
 * Returns:
 *   Pointer to the loaded dense layer if successful, NULL otherwise
 */
DENSE* load_dense(const char* filename);

/* store_dense - Store a dense layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the dense layer pointed to by d into it.
 * 
 * Parameters:
 *   d        - Pointer to the dense layer to be stored
 *   filename - Name of the file to store the dense layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_dense(const DENSE* d, const char* filename);

#endif
