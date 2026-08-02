/* Copyright (c) 2026 Gilad Odinak */

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "hash.h"
#include "cossim.h"
#include "wembio.h"

const char *usage =
    "Usage: wordembd -i <word-embedding-file>\n"
    "\n"
    "Interactive word-embedding expression evaluator.\n"
    "\n"
    "Enter an expression, then enter '=' on a separate line to evaluate it.\n"
    "Enter '.' on a separate line to exit.\n"
    "Enter '?' to print this help message.\n"
    "\n"
    "Operators:\n"
    "  +    Add two embeddings\n"
    "  -    Subtract two embeddings\n"
    "  *    Multiply embedding by a scalar\n"
    "  /    Divide an embedding by a scalar\n"
    "  @    Compute cosine similarity between two embeddings\n"
    "  ()   Group expressions\n"
    "\n"
    "Operator precedence, highest to lowest:\n"
    "  ()  unary + -  @  * /  + -\n"
    "\n"
    "Example:\n"
    "  > boy - man + woman\n"
    "  > =\n"
    "  Result: girl\n";

/* Maximum number of tokens in an expression. It also bounds the
 * shunting-yard queue and stacks, and the temporaries pool, since
 * none of them can hold more entries than there are tokens.
 */
#define MAXTKN 256

/* Value type for evaluation */
enum { VAL_VEC, VAL_SCALAR, VAL_OP };
typedef struct {
    int type;    /* one of: VAL_VEC, VAL_SCALAR, VAL_OP */
    union {
        float *v;    /* for VAL_VEC    */
        float s;     /* for VAL_SCALAR */
        char op;     /* for VAL_OP     */
    };
} Val;

/* Operator precedence */
int precedence(char op)
{
    switch (op) {
        case '+': case '-': return 1;
        case '*': case '/': return 2;
        case '@': return 3;
        case '#': case '~': return 4; /* Unary + , - */
    }
    return 0;
}

/* Unary operators are right associative. */
int right_assoc(char op)
{
    return op == '#' || op == '~';
}

/* Evaluates an embedding expression and prints the result.
 *
 * The expression is given as a token array. Each token is either a vector,
 * a scalar, or an operator.
 *
 * Supported operators:
 *   '+'  vector addition
 *   '-'  vector subtraction
 *   '*'  multiplication of embedding by scalar
 *   '/'  division of vector by scalar
 *   '@'  cosine similarity (scalar result)
 *   '(' ')' grouping
 *
 * Operator precedence, highest to lowest:
 *   Parentheses
 *   Unary '+' and '-' (right associative)
 *   '@' (cosine similarity)
 *   '*' and '/' (left associative)
 *   '+' and '-' (left associative)
 *
 * Type rules:
 * - '+' and '-' operate on embeddings only
 * - '/' operates on embedding and scalar and yields an embedding
 * - '*' operates on embedding and scalar in either order, yields an embeddinbg
 * - '@' operates on two embeddings and yields a scalar
 * - Once a scalar is produced, no further operations are allowed
 *
 * Grammar:
 * expression := term { ('+' | '-') term }
 * term       := similarity { ( '*' | '/') similarity }
 * similarity := factor { '@' factor }
 * factor     := { ('+' | '-') } (embedding | number | '(' expression ')')
 *
 * Behavior:
 *   - Prints the expression using vocabulary strings
 *   - Prints either the resulting embedding or scalar value
 *   - If the result is an embedding, prints the nearest vocabulary word
 *
 * Reference: https://en.wikipedia.org/wiki/Shunting_yard_algorithm
 *
 * Parametes:
 * tokens        - tokenized expression
 * tkncnt        - number of tokens
 * hmap          - word <=> index hashmap
 * embeddings    - vocab_size x embedding_dim array
 * tv            - temporaries pool MAXTKN x embedding_dim,
 * vocab_size    - number of embeddings
 * embedding_dim - embedding dimensionality
 */
void eval_embd_expr(
    Val* tokens, int tkncnt,
    HASHMAP* hmap, fArr2D embeddings, fArr2D tv_,
    int vocab_size, int embedding_dim)
{
    typedef float (*ArrNE)[embedding_dim];
    ArrNE E = (ArrNE) embeddings;
    ArrNE tv = (ArrNE) tv_;

    Val out[MAXTKN]; int outn = 0; /* Output queue    */
    Val ops[MAXTKN]; int opsn = 0; /* Operators stack */

    Val st[MAXTKN];
    int sp = 0; 
    int tp = 0;

    int expect_operand = 1;
    for (int i = 0; i < tkncnt; i++) {
        Val t = tokens[i];
        if (t.type != VAL_OP) { /* Token is a "number"          */
            out[outn++] = t;    /* Put it into the output queue */
            expect_operand = 0;
            continue;
        }

        char op = t.op;
        if (op == '(') {
            ops[opsn++] = t;
            expect_operand = 1;
            continue;
        }
        if (op == ')') {
            /* While operator at top of the stack is not  left parenthesis */
            while (opsn > 0 && ops[opsn - 1].op != '(')
                out[outn++] = ops[--opsn]; /* Pop it into the output queue */
            if (opsn == 0) {
                printf("Error: Missing '('\n");
                return;
            }
            opsn--; /* Pop left parenthesis  */
            expect_operand = 0;
            continue;
        }

        /* Distinguish unary operators from binary ones. */
        if (expect_operand && (op == '-' || op == '+')) {
            t.op = (op == '-') ? '~' : '#';
            op = t.op;
        }

        while (opsn > 0) {
            char top = ops[opsn - 1].op;
            if (top != '(' &&
                (precedence(top) > precedence(op) ||
                 (precedence(top) == precedence(op) && !right_assoc(op))))
                out[outn++] = ops[--opsn];
            else
                break;
        }
        ops[opsn++] = t;
        expect_operand = 1;
    }
    /* Pop remaining operatos into output queue */
    while (opsn > 0) {
        if (ops[opsn - 1].op == '(') { /* Not matched above with a ')' */
            printf("Error: Missing ')'\n");
            return;
        }
        out[outn++] = ops[--opsn];
    }

    /* Evaluate the expression */
    for (int i = 0; i < outn; i++) {
        Val t = out[i];
        if (t.type != VAL_OP) {
            st[sp++] = t;
            continue;
        }
        /* t.type == VAL_OP */
        char op = t.op;
        if (op == '~' || op == '#') {
            if (sp < 1) {
                printf("Error: missing operand\n");
                return;
            }
            if (op == '~') {
                if (st[sp - 1].type == VAL_VEC) {
                    if (tp >= MAXTKN) {
                        printf("Error: expression is too complex\n");
                        return;
                    }
                    float *r = tv[tp++];
                    for (int k = 0; k < embedding_dim; k++)
                        r[k] = -st[sp - 1].v[k];
                    st[sp - 1] = (Val){ VAL_VEC, .v = r };
                }
                else if (st[sp - 1].type == VAL_SCALAR) {
                    st[sp - 1].s = -st[sp - 1].s;
                }
                else {
                    printf("Type error: unary '-' requires an operand\n");
                    return;
                }
            }
            continue;
        }

        if (sp < 2) {
            printf("Error: missing operand\n");
            return;
        }

        /* Binary operator */
        Val b = st[--sp];
        Val a = st[--sp];

        if (op == '@') {
            if (a.type != VAL_VEC || b.type != VAL_VEC) {
                printf("Type error: '@' requires two embeddings\n");
                return;
            }
            float r = cosine_similarity(a.v, b.v, embedding_dim);
            st[sp++] = (Val){ VAL_SCALAR, .s = r };
            continue;
        }
        if (op == '*' || op == '/') {
            if (op == '*') {
                if (!((a.type == VAL_VEC && b.type == VAL_SCALAR) ||
                      (a.type == VAL_SCALAR && b.type == VAL_VEC))) {
                    printf("Type error: '*' requires embedding and scalar\n");
                    return;
                }
                if (a.type == VAL_SCALAR) {
                    Val t = a; a = b; b = t;
                }
            }
            else
            if (a.type != VAL_VEC || b.type != VAL_SCALAR) {
                printf("Type error: '/' requires vector / scalar\n");
                return;
            }
            if (op == '/' && b.s == 0.0f) {
                printf("Error: division by zero\n");
                return;
            }
            if (tp >= MAXTKN) {
                printf("Error: expression is too complex\n");
                return;
            }
            float* r = tv[tp++];
            for (int k = 0; k < embedding_dim; k++)
                r[k] = (op == '*') ? a.v[k] * b.s : a.v[k] / b.s;
            st[sp++] = (Val){ VAL_VEC, .v = r };
            continue;
        }
        if (op == '+' || op == '-') {
            if (a.type != VAL_VEC || b.type != VAL_VEC) {
                printf("Type error: '+' and '-' require embeddings\n");
                return;
            }
            if (tp >= MAXTKN) {
                printf("Error: expression is too complex\n");
                return;
            }
            float* r = tv[tp++];
            for (int k = 0; k < embedding_dim; k++)
                r[k] = (op == '+') ? a.v[k] + b.v[k]
                                   : a.v[k] - b.v[k];
            st[sp++] = (Val){ VAL_VEC, .v = r };
            continue;
        }
    }

    if (sp != 1) {
        printf("Invalid expression\n");
        return;
    }
    if (st[0].type == VAL_SCALAR) {
        printf("= %f\n", st[0].s);
        return;
    }
    if (st[0].type != VAL_VEC) {
        printf("Invalid expression\n");
        return;
    }

    /* Find nearest word */
    int best = -1;
    float best_sim = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        float s = cosine_similarity(st[0].v, E[i], embedding_dim);
        if (s > best_sim) {
            best_sim = s;
            best = i;
        }
    }

    printf("Result: ");
    for (int i = 0; i < 4 && i < embedding_dim; i++)
        printf("%.6f ", st[0].v[i]);
    if (embedding_dim > 4) printf("...");
    printf("\nNearest: %s (%.6f)\n",
           hashmap_inx2str(hmap, best), best_sim);
}

int main(int argc, char** argv)
{
    char* embfile;

    if (argc < 3 || strcmp(argv[1],"-i") != 0 || strlen(argv[2]) == 0) {
        fprintf(stderr,usage);
        exit(1);
    }
    embfile = argv[2];

    HASHMAP* hmap;
    fArr2D embeddings;
    int vocab_size, embedding_dim;
    if (!load_word_embeddings(embfile,&vocab_size,&embedding_dim,
                              NULL,NULL,NULL,&hmap,&embeddings))
        exit(1);
    typedef float (*ArrWE)[embedding_dim];
    ArrWE word_embeddings = (ArrWE) embeddings;

    Val tokens[MAXTKN];
    int tkninx = 0;

    fArr2D tv = allocmem(MAXTKN,embedding_dim,float);

    for (;;) {
        char line[256];
        fprintf(stdout,"> ");
        fflush(stdout);
        char* p = fgets(line,sizeof(line),stdin);
        if (p == NULL)
            break;
        for (; *p != '\0'; p++)
            if (*p < 0x20)
                *p = ' ';
        while (--p >= line && *p == ' ') 
            *p = '\0';

        p = line;
        if (strcmp(p,".") == 0)
            break;
        if (strcmp(p,"?") == 0) {
            puts(usage);
            continue;
        }
        if (strcmp(p,"=") == 0) {
            eval_embd_expr(tokens,tkninx,hmap,
                           (fArr2D) word_embeddings,tv,
                           vocab_size,embedding_dim);
            printf("\n");
            tkninx = 0;
            continue;
        }

        while (*p != '\0' && tkninx < MAXTKN) {
            while (*p == ' ') p++;
            if (*p == '\0') break;

            switch (*p) {
                case '+': case '-': 
                case '*': case '/': 
                case '@': 
                case '(': case ')':
                    tokens[tkninx++] = (Val){ VAL_OP, .op = *p++ };
                continue;
            }

            if (isalpha((unsigned char) *p) || *p == '\'') {
                char word[256], *w = word;
                while (*p != '\0' && 
                       (isalpha((unsigned char) *p) || *p == '\''))
                    if (w < word + sizeof(word) - 1)
                        *w++ = tolower(*p++);
                *w = '\0';

                int winx = hashmap_str2inx(hmap, word, 0);
                if (winx < 0 || winx >= vocab_size)
                    winx = 0;

                tokens[tkninx++] = (Val){ VAL_VEC, .v = word_embeddings[winx] };

                printf("%s:", word);
                for (int i = 0; i < 4 && i < embedding_dim; i++)
                    printf(" %.6f", word_embeddings[winx][i]);
                if (embedding_dim > 4) printf(" ...");
                printf("\n");
                continue;
            } 

            char* e;
            float scalar = strtof(p,&e);
            if (p != e) {
                tokens[tkninx++] = (Val){ VAL_SCALAR, .s = scalar };
                p = e;
                continue;
            }
            p++;
        }
    }

    hashmap_free(hmap);
    freemem(word_embeddings);
    freemem(tv);
    return 0;
}
