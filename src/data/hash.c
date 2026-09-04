/* Copyright (c) 2023-2024 Gilad Odinak  */
/* Hash map functions */
#include <strings.h>
#include <stdlib.h>  /* realloc() */
#include <memory.h>  /* memset()  */
#include "mem.h"
#include "hash.h"

/* Creates an hash map with map_size entries. Hashed strings are stored
 * in memory with initial size of mem_size.
 * Returns a pointer to the has map
 */
HASHMAP* hashmap_create(int map_size, int mem_size)
{
    HASHMAP* m = allocmem(1,1,HASHMAP);
    m->map_size = map_size;
    m->i2s = allocmem(1,map_size,int);
    m->s2i = allocmem(1,map_size,int);
    m->map = allocmem(1,map_size,int);
    for (int i = 0; i < m->map_size; i++)
        m->map[i] = -1;
    m->map_used = 0;
    m->mem_size = mem_size;
    m->mem = allocmem(1,mem_size,char);
    m->mem_used = 0;
    return m;
}

/* Frees memory allocated by hashmap_create()
 */
void hashmap_free(HASHMAP* m)
{
    if (m == NULL) return;
    freemem(m->i2s);
    freemem(m->s2i);
    freemem(m->map);
    freemem(m->mem);
    freemem(m);
}

/* Returns the index of the passed in string, -1 on error.
 */
int hashmap_str2inx_int(HASHMAP* restrict m, const char* str, int ins)
{
    int hinx = hash(str) % m->map_size;
    int first = hinx;
    for (;;) {
        if (m->map[hinx] < 0) { /* End of search for this hash - not found */
            if (!ins) /* Do not add */
                return -1;
            if (m->map_used >= m->map_size) /* Table full */
                return -1;
            int len = strlen(str) + 1; /* len includes terminating '\0' */
            if (m->mem_used + len >= m->mem_size) {
                /* Increase strings memory */
                int new_size = m->mem_size * 3 / 2 + len;
                void* mem = realloc(m->mem,new_size);
                if (mem == NULL) /* Out of memory */
                    return -1;
                m->mem = mem;
                m->mem_size = new_size;
                memset(m->mem + m->mem_used,0,m->mem_size - m->mem_used);
            }
            memcpy(m->mem + m->mem_used,str,len);
            m->map[hinx] = m->mem_used;
            m->mem_used += len;
            m->i2s[m->map_used] = hinx;
            m->s2i[hinx] = m->map_used;
            return m->map_used++;
        }
        if (strcmp(m->mem + m->map[hinx],str) == 0) /* Already exists */
            return m->s2i[hinx];
        /* Linear search */
        if (++hinx >= m->map_size) /* Wrap around    */
            hinx = 0;
        if (hinx == first)  /* Entire table searched - not found */
            return -1;
    }
}
