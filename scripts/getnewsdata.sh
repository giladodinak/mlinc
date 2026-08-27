#!/bin/bash
# Must be run at project root as scripts/getnewsdata.sh
#
# Obtains the news-articles corpus and builds a cleaned training set:
#   - unzips the corpus into a temporary directory
#   - for each file in selected_files.lst, strips the article header
#     and writes the cleaned files into data/news/data
#   - removes the temporary directory, keeping the downloaded zip
#
# Requires data/news/selected_files.lst

set -euo pipefail

NEWSDIR="data/news"
ZIP="$NEWSDIR/news-articles-corpus.zip"
TMPDIR="$NEWSDIR/tmp"
RAW="$TMPDIR/data"      # the zip contains a leading data/ prefix
OUTDIR="$NEWSDIR/data"
FILETOT=870521          # expected raw article count
FILELIST="selected_files.lst" # subset: fiels with 50 to 800 words

require_inputs() {
  [ -f "$NEWSDIR/$FILELIST" ] || { echo "error: missing $NEWSDIR/$FILELIST" >&2; exit 1; }
}

# Download the zip with the kaggle cli, unless it is already here.
fetch_zip() {
  [ -f "$ZIP" ] && return
  echo "Download the dataset from"
  echo "  https://www.kaggle.com/datasets/sbhatti/news-articles-corpus"
  echo "and place it at $ZIP, or let the kaggle cli fetch it now."
  read -p "Press any key to continue, ^C to abort " _ || true
  echo "Downloading..."
  # If the below command does not work, install kaggle cli by running
  # 'pip install kaggle'. 
  # If it is installed but is not found by this script, run "
  # 'pip uninstall kaggle' to find its location, and add it to the path
  ( cd "$NEWSDIR" && kaggle datasets download -d sbhatti/news-articles-corpus )
}

# Unzip into the temp dir and report whether the raw count is as expected.
extract() {
  echo "Decompressing into $TMPDIR ..."
  rm -rf "$TMPDIR"
  unzip -q -o "$ZIP" -d "$TMPDIR"
  [ -d "$RAW" ] || { echo "error: expected $RAW (zip should contain a data/ prefix)" >&2; exit 1; }
  local n
  n=$(find "$RAW" -maxdepth 1 -name 'article_*.txt' | wc -l)
  if [ "$n" -eq "$FILETOT" ]; then
    echo "Dataset complete: $n articles"
  else
    echo "Dataset incomplete: expected $FILETOT, found $n"
  fi
}

# Strip the leading "By . / PUBLISHED: . / UPDATED: ." header from one article
# and print the cleaned text. The header only ever occupies the first line.
clean_file() {
  awk '
    NR == 1 {
      sub(/^By \. [^.]*\. /, "")            # By . <author> .
      gsub(/PUBLISHED: \. [^.|]*\. ?/, "")  # PUBLISHED: . <datetime> .
      gsub(/UPDATED: \. [^.|]*\. ?/, "")    # UPDATED: . <datetime> .
      sub(/^[ .|]+/, "")                    # leftover leading "| ." / dots
      gsub(/  +/, " ")                      # squeeze doubled spaces
    }
    { print }
  ' "$1"
}

# Clean every article named in one split list into OUTDIR.
clean_split() {
  local list="$NEWSDIR/$1" name
  echo "Cleaning $1 -> $OUTDIR"
  while read -r name; do
    [ -n "$name" ] || continue
    echo -en "Cleaning $name \r"
    clean_file "$RAW/$name" > "$OUTDIR/$name"
  done < "$list"
  echo -en "                               \r"
}

main() {
  mkdir -p "$NEWSDIR"
  require_inputs
  if [ -d "$OUTDIR" ]; then
    echo "$OUTDIR already exists; skipping. Remove it to rebuild."
    exit 0
  fi
  fetch_zip
  [ -f "$ZIP" ] || { echo "error: $ZIP not found" >&2; exit 1; }
  extract
  mkdir -p "$OUTDIR"
  clean_split "$FILELIST"
  rm -rf "$TMPDIR"
  echo "Done"
}

main
