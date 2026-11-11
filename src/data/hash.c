/* Copyright (c) 2023-2024 Gilad Odinak  */
/* Hash map functions                    */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mem.h"
#include "hash.h"

#define BLOCK_HDR ((int)sizeof(char*)) /* Link to previous block */

/* Creates an hash map with map_size entries. Hashed strings are stored
 * in memory blocks of mem_size bytes.
 * Returns a pointer to the hash map
 */
HASHMAP* hashmap_create(int map_size, int mem_size)
{
    if (mem_size <= BLOCK_HDR)
        return NULL;
    HASHMAP* m = allocmem(1,1,HASHMAP);
    m->map_size = map_size;
    m->i2s = allocmem(1,map_size,int);
    m->s2i = allocmem(1,map_size,int);
    m->map = allocmem(1,map_size,char*);
    for (int i = 0; i < m->map_size; i++)
        m->map[i] = NULL;
    m->map_used = 0;
    m->mem = NULL;
    m->mem_size = mem_size;
    m->mem_used = 0;
    return m;
}

/* Frees memory allocated by hashmap_create()
 */
void hashmap_free(HASHMAP* m)
{
    if (m == NULL) return;
    char* blk = m->mem;
    while (blk != NULL) {
        char* prev;
        memcpy(&prev,blk,BLOCK_HDR);
        freemem(blk);
        blk = prev;
    }
    freemem(m->i2s);
    freemem(m->s2i);
    freemem(m->map);
    freemem(m);
}

/* Copies str into string memory, starting a new block
 * if the current one is too small.
 * Returns a pointer to the copy, NULL on error.
 */
static char* store_str(HASHMAP* restrict m, const char* str)
{
    int len = strlen(str) + 1; /* len includes terminating '\0' */
    if (m->mem == NULL || len > m->mem_size - m->mem_used) {
        if (len > m->mem_size - BLOCK_HDR)
            return NULL;
        char* blk = allocmem(1,m->mem_size,char);
        memcpy(blk,&m->mem,BLOCK_HDR); /* Link to previous block */
        m->mem = blk;
        m->mem_used = BLOCK_HDR;
    }
    char* s = m->mem + m->mem_used;
    memcpy(s,str,len);
    m->mem_used += len;
    return s;
}

/* Returns the index of the passed in string, -1 on error.
 */
int hashmap_str2inx_int(HASHMAP* restrict m, const char* str, int ins)
{
    int hinx = hash(str) % m->map_size;
    int first = hinx;
    for (;;) {
        if (m->map[hinx] == NULL) { /* End of search for this hash - not found */
            if (!ins) /* Do not add */
                return -1;
            char* s = store_str(m,str);
            if (s == NULL) /* String too long */
                goto err2;
            m->map[hinx] = s;
            m->i2s[m->map_used] = hinx;
            m->s2i[hinx] = m->map_used;
            return m->map_used++;
        }
        if (strcmp(m->map[hinx],str) == 0) /* Already exists */
            return m->s2i[hinx];
        /* Linear search */
        if (++hinx >= m->map_size) /* Wrap around    */
            hinx = 0;
        if (hinx == first) { /* Entire table searched - not found */
            if (ins && m->map_used >= m->map_size)  /* Table full */
                goto err1;
            return -1;
        }
    }
err1:
    fflush(stdout);
    fprintf(stderr,"\nIn hashmap_str2inx: table full");
    exit(-1);
err2:
    fflush(stdout);
    fprintf(stderr,"\nIn hashmap_str2inx: string too long");
    exit(-1);
}
