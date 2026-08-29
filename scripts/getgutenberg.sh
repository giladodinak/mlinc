#!/usr/bin/env bash
# Must be executed at project root as scripts/getgutenberg.sh
#
# Downloads the Project Gutenberg plain-text archive if needed, then creates
# a cleaned ASCII English corpus split into ~1000-word chunks.

set -euo pipefail
export LC_ALL=C

ARCHIVE_URL="https://www.gutenberg.org/cache/epub/feeds/txt-files.tar.zip"
CATALOG_URL="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

GUTENBERG_DIR="data/gutenberg"
ARCHIVE="$GUTENBERG_DIR/txt-files.tar.zip"
TEXT_ROOT="$GUTENBERG_DIR/text"
TEXT_DIR="$TEXT_ROOT/cache/epub"
CATALOG="$GUTENBERG_DIR/pg_catalog.csv"
OUTDIR="$GUTENBERG_DIR/data"
MINCHARS=200     # Excludes very short files
CHUNK_WORDS=1000

mkdir -p "$GUTENBERG_DIR"

# Download archive if needed
if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading Project Gutenberg plain-text archive"
  echo "  source : $ARCHIVE_URL"
  echo "  dest   : $ARCHIVE"
  echo
  curl -fL -o "$ARCHIVE" "$ARCHIVE_URL"
else
  echo "Archive already exists: $ARCHIVE"
  echo "Skipping download"
fi

# Download catalog if needed
if [ ! -f "$CATALOG" ]; then
   curl -fsSL -o "$CATALOG" "$CATALOG_URL"
fi

# Extract text archive if needed.  The zip contains a tar archive.
if [ ! -f "$TEXT_ROOT/.complete" ]; then
  rm -rf "$TEXT_ROOT"
  mkdir -p "$TEXT_ROOT"
  echo "Extracting Project Gutenberg text files"
  unzip -p "$ARCHIVE" | tar -xf - -C "$TEXT_ROOT"
  touch "$TEXT_ROOT/.complete"
  echo "Text extracted under: $TEXT_DIR"
else
  echo "Text already extracted: $TEXT_DIR"
  echo "Skipping extraction"
fi

if [ ! -d "$TEXT_DIR" ]; then
  echo "error: text directory not found: $TEXT_DIR" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Parse pg_catalog.csv, output English book ids
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
  gsub(/\r/,"",id)
  gsub(/[ \r]/,"",lang)
  m = split(lang,La,";")
  for (k = 1; k <= m; k++) {
    if (tolower(La[k]) == "en") {
      print id
      break
    }
  }
}
AWK

# Strip PG boilerplate and split books into ~1000-word chunks in one pass.
# Each awk process handles a batch of books.
cat > "$TMP/process.awk" <<'AWK'
function nwords(s,    a,n,t) {
  t = s
  gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
  if (t == "")
    return 0
  n = split(t,a,/[[:space:]]+/)
  return n
}
function flush(    fn) {
  if (words == 0)
    return
  fn = sprintf("%s/gutenberg_%s_%04d.txt",outdir,id,++chunk)
  printf "%s",buf > fn
  close(fn)
  buf = ""
  words = 0
}
function add(s,    nw) {
  if (s == "")
    return
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
  if (rest != "")
    add(rest)
}
function output_para(    nw) {
  if (para == "")
    return
  nw = nwords(para)
  if (nw > target * 1.2)
    add_sentences(para)
  else
    add(para)
  if (buf != "")
    buf = buf "\n"
  para = ""
}
function chunk_line(line) {
  if (line ~ /^[[:space:]]*$/) {
    output_para()
  } else {
    if (para != "")
      para = para " "
    para = para line
  }
}
function body_line(line,    i) {
  bodychars += length(line) + 1
  if (!active) {
    hold[++hold_n] = line
    if (bodychars < minchars)
      return
    active = 1
    for (i = 1; i <= hold_n; i++)
      chunk_line(hold[i])
    delete hold
    hold_n = 0
  } else {
    chunk_line(line)
  }
}
function is_start(s) {
  s = tolower(s)
  return (s ~ /\*\*\* *start of (the|this) project gutenberg ebook/ ||
          s ~ /\*\*\*start of (the|this) project gutenberg ebook/  ||
          s ~ /\*end\* *the small print/                            ||
          s ~ /\*end the small print/)
}
function is_end(s) {
  s = tolower(s)
  return (s ~ /\*\*\* *end of (the|this) project gutenberg ebook/ ||
          s ~ /\*\*\*end of (the|this) project gutenberg ebook/   ||
          s ~ /end of project gutenberg/)
}
function start_book(path,    fn) {
  fn = path
  sub(/^.*\//,"",fn)
  id = fn
  sub(/^pg/,"",id)
  sub(/\.txt$/,"",id)

  started = 0
  done = 0
  pre_n = 0
  bodychars = 0
  active = 0
  hold_n = 0
  para = ""
  buf = ""
  words = 0
  chunk = 0
  have_book = 1
}
function begin_without_header(    i) {
  started = 1
  for (i = 1; i <= pre_n; i++)
    body_line(pre[i])
  delete pre
  pre_n = 0
}
function process_line(line) {
  if (done)
    return

  if (!started) {
    if (is_start(line)) {
      started = 1
      delete pre
      pre_n = 0
      return
    }
    if (is_end(line)) {
      begin_without_header()
      done = 1
      return
    }
    pre[++pre_n] = line
    if (pre_n >= 600)
      begin_without_header()
    return
  }

  if (is_end(line)) {
    done = 1
    return
  }
  body_line(line)
}
function finish_book() {
  if (!started)
    begin_without_header()

  if (bodychars < minchars) {
    print "S"
  } else {
    output_para()
    flush()
    print "K"
  }
  have_book = 0
}
index($0,"@@PGFILE@@") == 1 {
  if (have_book)
    finish_book()
  path = $0
  sub(/^@@PGFILE@@/,"",path)
  start_book(path)
  next
}
{
  process_line($0)
}
END {
  if (have_book)
    finish_book()
}
AWK

# Convert each batch to ASCII, preserving book boundaries.
cat > "$TMP/process_batch.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

awkfile="$1"
outdir="$2"
target="$3"
minchars="$4"
shift 4

{
  for path in "$@"; do
    printf '@@PGFILE@@%s\n' "$path"
    cat -- "$path"
  done
} | LC_ALL=C.UTF-8 iconv -f UTF-8 -t 'ASCII//TRANSLIT//IGNORE' \
  | awk -v outdir="$outdir" -v target="$target" -v minchars="$minchars" \
        -f "$awkfile"
SH
chmod +x "$TMP/process_batch.sh"

if [ ! -f "$CATALOG" ]; then
  echo "error: $CATALOG not found" >&2
  exit 1
fi

# Build the input list directly from catalog ids.
awk -f "$TMP/english.awk" "$CATALOG" | while read -r id; do
  path="$TEXT_DIR/$id/pg$id.txt"
  if [ -f "$path" ]; then
    printf '%s\n' "$path"
  fi
done > "$TMP/books.lst"

nbooks=$(wc -l < "$TMP/books.lst")
echo "found $nbooks books" >&2

mkdir -p "$OUTDIR"

JOBS="${JOBS:-$(nproc)}"
BATCH="${BATCH:-64}"
echo "Processing with $JOBS workers, $BATCH books per worker" >&2

# Process independent batches in parallel.  Each worker reads every source
# book once and writes chunks directly to OUTDIR.
setsid xargs -r -P "$JOBS" -n "$BATCH" \
  bash "$TMP/process_batch.sh" "$TMP/process.awk" \
       "$OUTDIR" "$CHUNK_WORDS" "$MINCHARS" \
  < "$TMP/books.lst" > "$TMP/status" &

pid=$!

cleanup()
{
    echo
    trap - INT TERM EXIT

    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
        kill -- -"$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    rm -rf "$TMP"
}

trap cleanup INT TERM EXIT

while kill -0 "$pid" 2>/dev/null; do
    kept=$(grep -c '^K$' "$TMP/status" 2>/dev/null || true)
    skipped=$(grep -c '^S$' "$TMP/status" 2>/dev/null || true)
    echo -ne "\rProcessed $((kept + skipped)), kept $kept, skipped $skipped\r" >&2
    sleep 1
done
echo
wait "$pid"

kept=$(grep -c '^K$' "$TMP/status" || true)
skipped=$(grep -c '^S$' "$TMP/status" || true)

echo "Done: kept $kept, skipped $skipped" >&2
echo "Chunks written under: $OUTDIR" >&2
