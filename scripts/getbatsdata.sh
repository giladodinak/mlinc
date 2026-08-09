#!/bin/bash
# Downloads the Bigger Analogy Test Set (BATS 3.0) into data/bats/BATS_3.0
# Source: vecto.space BATS project page (dataset hosted on pCloud).
# pCloud public links aren't direct downloads, so we resolve the share
# code to a real URL via the pCloud API first (needs python3 for JSON).
mkdir -p data/bats
pushd data/bats

CODE="XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV"

read HOST DLPATH < <(wget --no-check-certificate -qO- \
    "https://api.pcloud.com/getpublinkdownload?code=${CODE}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["hosts"][0], d["path"])')

if [ -z "$HOST" ] || [ -z "$DLPATH" ]; then
    echo "Failed to resolve pCloud link; see https://vecto.space/projects/BATS/" >&2
    popd; exit 1
fi

wget --no-check-certificate -O BATS_3.0.zip "https://${HOST}${DLPATH}"
unzip -q BATS_3.0.zip
rm -f BATS_3.0.zip
chmod -R a-w BATS_3.0
popd
