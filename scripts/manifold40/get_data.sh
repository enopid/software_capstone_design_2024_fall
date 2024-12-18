#!/usr/bin/env bash

TAR_FILE="/data/thom08/rental_seraph_csj/datasets/Manifold40.zip"
TAR_FILE_DIR="/data/thom08/rental_seraph_csj/datasets/"
EXTRACTED_DIR="/local_datasets/Meshdataset/Manifold40/raw"
REMOVE_DIR="/local_datasets/Meshdataset/Manifold40/raw/Manifold40"

#download data
if [ -f "$TAR_FILE" ]; then
    echo "The tar file already exists. Skipping download."
else
    cd $TAR_FILE_DIR
    echo "The tar file does not exist. Download now..."
    wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40.zip
    echo "Download completed."
fi

#unzip data
if [ -d "$EXTRACTED_DIR" ]; then
    echo "The directory $EXTRACTED_DIR already exists. Skipping unzip."
else
    mkdir -p $EXTRACTED_DIR
    cd $EXTRACTED_DIR
    echo "The directory $EXTRACTED_DIR does not exist. Unzipping now..."
    unzip -q $TAR_FILE 
    
    mv ./Manifold40/* ./
    rmdir ./Manifold40

    echo "Unzip completed."
fi