#!/bin/bash

OUTDIR=$OUT_DIR

if [ ! -d ${OUTDIR} ]; then
    mkdir ${OUTDIR}
fi

python scripts/train_classifier.py \
    --seed 12345 \
    --save_path ${OUTDIR}/derm_12345.pt \
    --device 7 \
    --train_dataset_class ISICSexDataset \
    --test_dataset_class ISICSexDataset

python scripts/train_classifier.py \
    --seed 23456 \
    --save_path ${OUTDIR}/derm_23456.pt \
    --device 7 \
    --train_dataset_class ISICSexDataset \
    --test_dataset_class ISICSexDataset

python scripts/train_classifier.py \
    --seed 34567 \
    --save_path ${OUTDIR}/derm_34567.pt \
    --device 7 \
    --train_dataset_class ISICSexDataset \
    --test_dataset_class ISICSexDataset

python scripts/train_classifier.py \
    --seed 45678 \
    --save_path ${OUTDIR}/derm_45678.pt \
    --device 7 \
    --train_dataset_class ISICSexDataset \
    --test_dataset_class ISICSexDataset

python scripts/train_classifier.py \
    --seed 56789 \
    --save_path ${OUTDIR}/derm_56789.pt \
    --device 7 \
    --train_dataset_class ISICSexDataset \
    --test_dataset_class ISICSexDataset