#!/bin/bash
mkdir -p data/timit
mkdir -p data/timit/features/train
mkdir -p data/timit/features/validate
mkdir -p data/timit/features/test
if ! [ -d data/timit/LDC93S1/timit ] ; then
  pushd data/timit
  if ! [ -f TIMIT.tar ] ; then
    echo "The TIMIT dataset is available via https://catalog.ldc.upenn.edu/LDC93S1"
    echo "Place the compressed file in '$(pwd)/data/timit' as TIMIT.tar"
    echo "Alternatively, the dataset is available via torrent"
    read -p "Press any key to continue and download via torrent, ^C to abort "
    # If the below command does not work, manually use torrent client
    # and rename the downloaded file TIMIT.tar
    # In that case, manually run the remaining commands
    # Uncomment the below 3 lines to download via torrent
    ####
    echo "Starting ktorrent. You may need to manually close it"
    ktorrent --silent https://academictorrents.com/download/7bb27b6fe1712ff49ec80b0a1fcdb43fb465923e.torrent
    mv ~/'TIMIT Acoustic-Phonetic Continuous Speech Corpus - LDC93S1' TIMIT.tar
    ####
  fi
  if [ -f TIMIT.tar ] ; then 
      tar -xf TIMIT.tar
  else
      echo -e "\nTIMIT.tar does not exist.\n"
  fi
  popd
fi
if [ -d data/timit/LDC93S1/timit ] ; then
bin/timitfeat data/timit/tr_file.lst  data/timit/LDC93S1/timit/TIMIT data/timit/features/train
bin/timitfeat data/timit/vd_file.lst  data/timit/LDC93S1/timit/TIMIT data/timit/features/validate
bin/timitfeat data/timit/te_file.lst  data/timit/LDC93S1/timit/TIMIT data/timit/features/test
fi
