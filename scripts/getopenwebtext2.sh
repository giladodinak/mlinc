#!/usr/bin/env bash
# Must be executed at project root as scripts/getopenwebtext2.sh
#
# Downloads OpenWebText2 if needed, extracts the compressed JSONL shards,
# then creates a cleaned ASCII corpus split into ~1000-word chunks.

set -euo pipefail
export LC_ALL=C

ARCHIVE_URL='https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar?download=true'

OPENWEBTEXT_DIR="data/openwebtext2"
ARCHIVE="$OPENWEBTEXT_DIR/openwebtext2.jsonl.zst.tar"
SHARD_DIR="$OPENWEBTEXT_DIR/shards"
OUTDIR="$OPENWEBTEXT_DIR/data"
DONE_DIR="$OPENWEBTEXT_DIR/done"
MINCHARS=200
CHUNK_WORDS=1000

mkdir -p "$OPENWEBTEXT_DIR"

for cmd in wget tar zstd python3 iconv awk find xargs; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "error: required command not found: $cmd" >&2
        exit 1
    fi
done

# Download archive if shards have not already been extracted.
if [ ! -f "$SHARD_DIR/.complete" ]; then
    if [ ! -f "$ARCHIVE" ]; then
        echo "Downloading OpenWebText2"
        echo "  source : $ARCHIVE_URL"
        echo "  dest   : $ARCHIVE"
        echo
        wget -c -O "$ARCHIVE" "$ARCHIVE_URL"
    else
        echo "Archive already exists: $ARCHIVE"
        echo "Skipping download"
    fi

    rm -rf "$SHARD_DIR"
    mkdir -p "$SHARD_DIR"
    echo "Extracting OpenWebText2 compressed JSONL shards"
    tar -xf "$ARCHIVE" -C "$SHARD_DIR"
    touch "$SHARD_DIR/.complete"
    echo "Shards extracted under: $SHARD_DIR"
else
    echo "Shards already extracted: $SHARD_DIR"
    echo "Skipping download and extraction"
fi

TMP="$(mktemp -d)"
pid=""

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

# Read JSONL from stdin and emit only document text, with an explicit document
# boundary marker. OpenWebText2 records contain "text" and "meta" fields.
cat > "$TMP/extract_text.py" <<'PY'
#!/usr/bin/env python3
import json
import sys

for n, line in enumerate(sys.stdin, 1):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"JSON parse error at record {n}: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(obj, str):
        text = obj
    else:
        text = obj.get("text", "")

    if isinstance(text, list):
        text = "\n\n".join(str(x) for x in text)
    elif not isinstance(text, str):
        text = str(text)

    # The marker is ASCII so it survives iconv unchanged.
    sys.stdout.write(f"@@OWTDOC@@{n}\n")
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
PY
chmod +x "$TMP/extract_text.py"

# Split documents into approximately CHUNK_WORDS chunks. This is deliberately
# structured like the Gutenberg chunker, but no Gutenberg header stripping is
# needed because OpenWebText2 already stores cleaned document text.
cat > "$TMP/process.awk" <<'AWK'
function nwords(s,    a,n,t) {
    t = s
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", t)
    if (t == "")
        return 0
    n = split(t,a,/[[:space:]]+/)
    return n
}

function update_status() {
    printf "%d %d %d\n", kept, skipped, chunks > statusfile
    close(statusfile)
}

function flush(    fn) {
    if (words == 0)
        return
    fn = sprintf("%s/openwebtext2_%s_%08d_%04d.txt",
                 outdir, shard, doc, ++chunk)
    printf "%s",buf > fn
    close(fn)
    buf = ""
    words = 0
    chunks++
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

function start_doc(n) {
    doc = n + 0
    bodychars = 0
    active = 0
    hold_n = 0
    para = ""
    buf = ""
    words = 0
    chunk = 0
    have_doc = 1
}

function finish_doc() {
    if (!have_doc)
        return

    if (bodychars < minchars) {
        skipped++
    } else {
        output_para()
        flush()
        kept++
    }

    if ((kept + skipped) % 1000 == 0)
        update_status()

    have_doc = 0
}

index($0,"@@OWTDOC@@") == 1 {
    finish_doc()
    n = $0
    sub(/^@@OWTDOC@@/,"",n)
    start_doc(n)
    next
}

{
    if (have_doc)
        body_line($0)
}

END {
    finish_doc()
    update_status()
}
AWK

cat > "$TMP/process_shard.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

extractor="$1"
awkfile="$2"
outdir="$3"
target="$4"
minchars="$5"
statusdir="$6"
donedir="$7"
shard_path="$8"

base=$(basename "$shard_path")
shard=${base%.jsonl.zst}
shard=$(printf '%s' "$shard" | sed 's/[^A-Za-z0-9._-]/_/g')
statusfile="$statusdir/$shard.status"
donefile="$donedir/$shard.done"

if [ -f "$donefile" ]; then
    exit 0
fi

printf '0 0 0\n' > "$statusfile"

zstd -dc -- "$shard_path" \
  | python3 "$extractor" \
  | LC_ALL=C.UTF-8 iconv -f UTF-8 -t 'ASCII//TRANSLIT//IGNORE' \
  | awk -v outdir="$outdir" \
        -v target="$target" \
        -v minchars="$minchars" \
        -v shard="$shard" \
        -v statusfile="$statusfile" \
        -f "$awkfile"

touch "$donefile"
rm -f -- "$shard_path"
SH
chmod +x "$TMP/process_shard.sh"

mkdir -p "$OUTDIR" "$DONE_DIR" "$TMP/status"

find "$SHARD_DIR" -type f -name '*.jsonl.zst' -print0 \
    | sort -z > "$TMP/shards.lst"

nshards=$(tr -cd '\0' < "$TMP/shards.lst" | wc -c)
if [ "$nshards" -eq 0 ]; then
    # This can be a completed rerun where processed shards were already removed.
    ndone=$(find "$DONE_DIR" -type f -name '*.done' | wc -l)
    if [ "$ndone" -gt 0 ]; then
        echo "No unprocessed shards remain; corpus appears complete" >&2
        echo "Chunks written under: $OUTDIR" >&2
        exit 0
    fi
    echo "error: no .jsonl.zst shards found under $SHARD_DIR" >&2
    exit 1
fi

echo "found $nshards unprocessed shards" >&2

JOBS="${JOBS:-$(nproc)}"
echo "Processing with $JOBS workers" >&2

setsid xargs -0 -r -P "$JOBS" -n 1 \
    bash "$TMP/process_shard.sh" \
         "$TMP/extract_text.py" "$TMP/process.awk" \
         "$OUTDIR" "$CHUNK_WORDS" "$MINCHARS" \
         "$TMP/status" "$DONE_DIR" \
    < "$TMP/shards.lst" &

pid=$!

while kill -0 "$pid" 2>/dev/null; do
    read -r kept skipped chunks < <(
        awk '{k += $1; s += $2; c += $3} END {print k+0, s+0, c+0}' \
            "$TMP"/status/*.status 2>/dev/null || echo '0 0 0'
    )
    done_shards=$(find "$DONE_DIR" -type f -name '*.done' | wc -l)
    echo -ne "\rProcessed $((kept + skipped)) documents, kept $kept, skipped $skipped, chunks $chunks, shards done $done_shards\r" >&2
    sleep 1
done
echo

wait "$pid"
pid=""

read -r kept skipped chunks < <(
    awk '{k += $1; s += $2; c += $3} END {print k+0, s+0, c+0}' \
        "$TMP"/status/*.status 2>/dev/null || echo '0 0 0'
)

echo "Done: kept $kept documents, skipped $skipped, wrote $chunks chunks" >&2
echo "Chunks written under: $OUTDIR" >&2
