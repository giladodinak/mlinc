#!/usr/bin/env bash
# Must be executed at project root as scripts/getgutenberg.sh
#
# Downloads a Project gutenberg plain-text mirror if needed, then creates a
# cleaned English corpus split into ~1000-word chunks.

set -euo pipefail
export LC_ALL=C

RSYNC_MIRROR="aleph.gutenberg.org::gutenberg"
# RSYNC_MIRROR="ftp.ibiblio.org::gutenberg"
# RSYNC_MIRROR="rsync.mirrorservice.org::gutenberg"

MIRROR_DIR="data/gutenberg/mirror"
OUTDIR="data/gutenberg/data"
OUT=""
ENGLISH=1
MINCHARS=200
CHUNK_WORDS=1000
CATALOG_URL="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

# Download mirror if it does not already exist 
if [ ! -d "$MIRROR_DIR" ]; then
    mkdir -p "$MIRROR_DIR"

    echo "Mirroring Project Gutenberg plain-text files"
    echo "  mirror : $RSYNC_MIRROR"
    echo "  dest   : $MIRROR_DIR"
    echo

    curl -fsSL -o "$MIRROR_DIR/pg_catalog.csv" "$CATALOG_URL"

    # All encodings are fetched
    rsync -rlptv --prune-empty-dirs \
          --include='*/' \
          --include='*-0.txt' \
          --include='*-8.txt' \
          --include='*[0-9].txt' \
          --exclude='*' \
          "$RSYNC_MIRROR" "$MIRROR_DIR"

    echo
    echo "Text mirrored under: $MIRROR_DIR"
else
    echo "Mirror already exists: $MIRROR_DIR"
    echo "Skipping download"
fi

[ -d "$MIRROR_DIR" ] || {
    echo "error: mirror directory not found: $MIRROR_DIR" >&2
    exit 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Strip PG boilerplate 
cat > "$TMP/strip.awk" <<'AWK'
{ L[NR] = $0 }
END {
    n = NR
    if (n == 0) exit
    lim = int(n/3); if (lim < 600) lim = 600; if (lim > n) lim = n
    start = 1
    for (i = 1; i <= lim; i++) {
        s = tolower(L[i])
        if (s ~ /\*\*\* *start of (the|this) project gutenberg ebook/ ||
            s ~ /\*\*\*start of (the|this) project gutenberg ebook/  ||
            s ~ /\*end\* *the small print/                            ||
            s ~ /\*end the small print/)
            start = i + 1
    }
    end = n + 1
    for (i = start; i <= n; i++) {
        e = tolower(L[i])
        if (e ~ /\*\*\* *end of (the|this) project gutenberg ebook/ ||
            e ~ /\*\*\*end of (the|this) project gutenberg ebook/   ||
            e ~ /end of project gutenberg/) {
            end = i; break
        }
    }
    for (i = start; i < end; i++) print L[i]
}
AWK

# Split cleaned books into ~1000-word chunks
# Prefer paragraph boundaries. Oversized paragraphs are split on sentence
# ends where possible. Chunk names are gutenberg_<book-id>_<chunk>.txt.
cat > "$TMP/chunk.awk" <<'AWK'
function nwords(s,    a,n,t) {
    t = s
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
    if (t == "") return 0
    n = split(t,a,/[[:space:]]+/)
    return n
}
function flush(    fn) {
    if (words == 0) return
    fn = sprintf("%s/gutenberg_%s_%04d.txt",outdir,id,++chunk)
    printf "%s",buf > fn
    close(fn)
    buf = ""
    words = 0
}
function add(s,    nw) {
    if (s == "") return
    nw = nwords(s)
    if (words > 0 && words + nw > target && words >= target * 0.8)
        flush()
    buf = buf s "\n"
    words += nw
}
function add_sentences(p,    rest,s,nw) {
    rest = p
    while (match(rest,/^[^.!?]*[.!?]+[[:space:]]+/)) {
        s = substr(rest,1,RLENGTH)
        nw = nwords(s)
        if (words > 0 && words + nw > target)
            flush()
        add(s)
        rest = substr(rest,RLENGTH + 1)
    }
    if (rest != "") add(rest)
}
function emit_para(    nw) {
    if (para == "") return
    nw = nwords(para)
    if (nw > target * 1.2)
        add_sentences(para)
    else
        add(para)
    if (buf != "") buf = buf "\n"
    para = ""
}
BEGIN { buf = ""; para = ""; words = 0; chunk = 0 }
{
    if ($0 ~ /^[[:space:]]*$/) {
        emit_para()
    } else {
        if (para != "") para = para " "
        para = para $0
    }
}
END { emit_para(); flush() }
AWK

# Parse pg_catalog.csv, emit English book ids
cat > "$TMP/english.awk" <<'AWK'
function parsecsv(line,   i,c,field,infield,nf) {
    nf = 0; field = ""; infield = 0
    for (i = 1; i <= length(line); i++) {
        c = substr(line,i,1)
        if (c == "\"") {
            if (infield && substr(line,i+1,1) == "\"") {
                field = field "\""; i++
            } else {
                infield = !infield
            }
        } else if (c == "," && !infield) {
            nf++; F[nf] = field; field = ""
        } else {
            field = field c
        }
    }
    nf++; F[nf] = field
    return nf
}
NR == 1 { next }
{
    parsecsv($0)
    id = F[1]; lang = F[5]
    gsub(/ /,"",lang)
    m = split(lang,La,";")
    for (k = 1; k <= m; k++)
        if (tolower(La[k]) == "en") { print id; break }
}
AWK

# Build English-id set
declare -A EN
if [ "$ENGLISH" -eq 1 ]; then
    CAT="$MIRROR_DIR/pg_catalog.csv"
    [ -f "$CAT" ] || {
        echo "error: $CAT not found" >&2
        exit 1
    }
    while read -r eid; do
        EN["$eid"]=1
    done < <(awk -f "$TMP/english.awk" "$CAT")
    echo "catalog: ${#EN[@]} English book ids" >&2
fi

# Pick best encoding file per book id
find "$MIRROR_DIR" -type f -name '*.txt' -print \
| awk '
{
    path = $0
    k = split(path,parts,"/")
    fname = parts[k]
    if (fname ~ /^[0-9]+(-[0-9]+)?\.txt$/) {
        base = fname; sub(/\.txt$/,"",base)
        id = base; suf = ""
        if (index(base,"-") > 0) {
            split(base,a,"-"); id = a[1]; suf = a[2]
        }
        # prefer UTF-8 (-0) over 8-bit (-8) over plain.
        # rank = (suf == "0") ? 0 : (suf == "8") ? 1 : (suf == "") ? 2 : 3
        # prefer plain over 8-bit (-8) over UTF-8 (-0)
        rank = (suf == "") ? 0 : (suf == "8") ? 1 : (suf == "0") ? 2 : 3
        if (!(id in br) || rank < br[id]) {
            br[id] = rank; bp[id] = path
        }
    }
}
END { for (id in bp) print id "\t" bp[id] }
' > "$TMP/best.tsv"

nbooks=$(wc -l < "$TMP/best.tsv")
echo "found $nbooks books" >&2

# Clean and chunk each book
mkdir -p "$OUTDIR"
[ -n "$OUT" ] && : > "$OUT"

kept=0
skipped=0
i=0
while IFS=$'\t' read -r id path; do
    i=$((i + 1))

    if [ "$ENGLISH" -eq 1 ] && [ -z "${EN[$id]:-}" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    awk -f "$TMP/strip.awk" "$path" > "$TMP/body.txt" 2>/dev/null || {
        skipped=$((skipped + 1))
        continue
    }

    chars=$(wc -c < "$TMP/body.txt")
    if [ "$chars" -lt "$MINCHARS" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    [ -n "$OUT" ] && {
        cat "$TMP/body.txt" >> "$OUT"
        printf '\n\n' >> "$OUT"
    }

    awk -v outdir="$OUTDIR" -v id="$id" -v target="$CHUNK_WORDS" \
        -f "$TMP/chunk.awk" "$TMP/body.txt"

    kept=$((kept + 1))
    if [ $((i % 100)) -eq 0 ]; then
        echo "  processed $i, kept $kept" >&2
    fi
done < "$TMP/best.tsv"

echo "done: kept $kept, skipped $skipped" >&2
echo "chunks written under: $OUTDIR" >&2
