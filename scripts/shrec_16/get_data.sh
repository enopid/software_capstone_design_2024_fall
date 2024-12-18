#!/usr/bin/env bash

TAR_FILE="/data/thom08/rental_seraph_csj/datasets/shrec_16.tar.gz"
TAR_FILE_DIR="/data/thom08/rental_seraph_csj/datasets/"
EXTRACTED_DIR="/local_datasets/Meshdataset/shrec_16/raw"

#download data
if [ -f "$TAR_FILE" ]; then
    echo "The tar file already exists. Skipping download."
else
    cd $TAR_FILE_DIR
    echo "The tar file does not exist. Download now..."
    wget https://www.dropbox.com/s/w16st84r6wc57u7/shrec_16.tar.gz
    echo "Download completed."
fi

#unzip data
if [ -d "$EXTRACTED_DIR" ]; then
    echo "The directory $EXTRACTED_DIR already exists. Skipping unzip."
else
    mkdir -p $EXTRACTED_DIR
    cd $EXTRACTED_DIR
    echo "The directory $EXTRACTED_DIR does not exist. Unzipping now..."
    tar -xzvf $TAR_FILE --strip-components=1
    echo "Unzip completed."
fi
