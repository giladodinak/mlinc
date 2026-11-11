/* Copyright (c) 2023-2024 Gilad Odinak  */
/* Hashing data structures and functions */
#ifndef HASH_H
#define HASH_H
#include <limits.h>
#include <string.h>

/* Adapted from djb2_hash                   */
/* https://theartincode.stanis.me/008-djb2/ */
static inline unsigned int djb2_hash(const char* str)
{
    unsigned int hash = 5381;
    for (int i = 0; str[i] != '\0'; i++)
        hash = hash * 33 + ((unsigned char) str[i]);
    return hash;
}

/* Returns a hash value for the passed in string.
 */
static inline int hash(const char* str)
{
    return (int)(djb2_hash(str) & INT_MAX);
}

typedef struct hashmap_s {
    int* i2s;       /* Maps index to map slot */
    int* s2i;       /* Maps map slot to index */
    char** map;     /* Stored String or NULL  */
    int map_size;
    int map_used;
    char* mem;      /* Current string block   */
    int mem_size;   /* Block size             */
    int mem_used;   /* Bytes used block       */
} HASHMAP;

/* Creates an hash map with map_size entries.
 * Hashed strings are stored in memory blocks of mem_size bytes.
 * Returns a pointer to the hash map.
 */
HASHMAP* hashmap_create(int map_size, int mem_size);

/* Frees memory allocated by hashmap_create()
 */
void hashmap_free(HASHMAP* m);

/* Returns the index of the passed in string, if it exists in the hashmap.
 * If the string is not in the hashmap and ins is not zero, inserts
 * it into the hashmap and returns its index; otherwise returns -1.
 * Returns -1 on error.
 */
static inline int hashmap_str2inx(HASHMAP* restrict m, const char* str, int ins)
{
    int hashmap_str2inx_int(HASHMAP* restrict m, const char* str, int ins);
    int hinx = hash(str) % m->map_size;
    if (m->map[hinx] != NULL && strcmp(m->map[hinx],str) == 0)
        return m->s2i[hinx];
    return hashmap_str2inx_int(m,str,ins);
}

/* Returns the string of the passed in index. Returns blank string
 * if an hashmap entry for the index does not exist.
 */
static inline const char* hashmap_inx2str(HASHMAP* restrict m, int inx)
{
    if (inx >= 0 && inx < m->map_used)
        return m->map[m->i2s[inx]];
    return "";
}

#endif
