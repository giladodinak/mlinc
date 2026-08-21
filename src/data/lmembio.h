/* Copyright (c) 2026 Gilad Odinak */
/* Functions to load and store the language-model token embedding layer */
#ifndef LMEMBIO_H
#define LMEMBIO_H
#include <stdio.h>
#include "lmemb.h"

/* read_lmemb - Read a token embedding layer from a file
 *
 * Reads an LMEMB layer from the file pointed to by fp.
 *
 * Returns:
 *   Pointer to the read layer if successful, NULL otherwise
 *
 * Notes:
 *   - The layer is (re)built with lmemb_create()/lmemb_init() from the
 *     stored V, E, T, padinx, B, training and tied fields, so its owned
 *     and scratch buffers exactly match a freshly initialized layer.
 *   - When tied, Wx is borrowed from an output projection and is not stored;
 *     it is left NULL for the caller to attach via lmemb_set_tied_weights().
 *     Otherwise Wx is read from the file.
 */
LMEMB* read_lmemb(FILE* fp);

/* write_lmemb - Write a token embedding layer to a file
 *
 * Writes the LMEMB layer pointed to by l to the file pointed to by fp.
 *
 * Parameters:
 *   l     - Pointer to the layer to be written
 *   final - If not zero, record the layer as inference-only (training 0)
 *   fp    - Output file
 *
 * Returns:
 *   1 if successful, 0 otherwise
 *
 * Notes:
 *   - Only Wx is persisted, and only when the layer owns it (untied). The
 *     gradient/moment and sparse-touch buffers are training scratch and are
 *     reallocated by read_lmemb().
 */
int write_lmemb(const LMEMB* l, int final, FILE* fp);

/* load_lmemb - Load a token embedding layer from a file (opens/closes it). */
LMEMB* load_lmemb(const char* filename);

/* store_lmemb - Store a token embedding layer into a file (opens/closes it). */
int store_lmemb(const LMEMB* l, const char* filename);

#endif
