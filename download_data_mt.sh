#!/bin/bash
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
SUBSCRIPTION_KEY=${1:-""}
ORIGINAL_DIR="$REPO/Data/Original_Data"
PREPROCESS_DIR="$REPO/Data/Preprocess_Scripts"
PROCESSED_DIR="$REPO/Data/Processed_Data"
mkdir -p $PROCESSED_DIR

OUTPATH=$ORIGINAL_DIR/MT_EN_HI/temp
mkdir -p $OUTPATH
if [ ! -f $OUTPATH/CMUHinglishDoG.zip ]; then
    wget -c http://festvox.org/cedar/data/notyet/CMUHinglishDoG.zip -P $OUTPATH -q --show-progress
    unzip -qq $OUTPATH/CMUHinglishDoG.zip -d $OUTPATH
fi
if [ ! -f $OUTPATH/618a14f.zip ]; then
    wget -c https://github.com/festvox/datasets-CMU_DoG/archive/618a14f.zip -P $OUTPATH -q --show-progress
    unzip -qq $OUTPATH/618a14f.zip -d $OUTPATH
fi

python $PREPROCESS_DIR/preprocess_mt_en_hi.py $OUTPATH $ORIGINAL_DIR/MT_EN_HI/ $PROCESSED_DIR/MT_EN_HI

rm -rf $OUTPATH
echo "Downloaded MT EN HI"
