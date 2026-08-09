#!/bin/bash
mkdir -p data/simlex
pushd data/simlex
wget --no-check-certificate https://fh295.github.io/SimLex-999.zip
unzip SimLex-999.zip
mv -f SimLex-999/SimLex-999.txt simlex.csv
mv -f SimLex-999/README.txt simlex.txt
chmod 444 simlex.csv simlex.txt
rm -rf SimLex-999 SimLex-999.zip
popd
