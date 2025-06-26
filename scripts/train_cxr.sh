#!/bin/bash
OUTDIR=$OUT_DIR

if [ ! -d ${OUTDIR} ]; then
    mkdir ${OUTDIR}
fi

python scripts/train_classifier.py \
    --seed 12345 \
    --save_path ${OUTDIR}/cxr_123456.pt \
    --device 6 \
    --n_epochs 30 \
    --train_dataset_class CheXpertRaceDataset \
    --test_dataset_class MIMICCXRRaceDataset

python scripts/train_classifier.py \
    --seed 23456 \
    --save_path ${OUTDIR}/cxr_23456.pt \
    --device 6 \
    --n_epochs 30 \
    --train_dataset_class CheXpertRaceDataset \
    --test_dataset_class MIMICCXRRaceDataset

python scripts/train_classifier.py \
    --seed 34567 \
    --save_path ${OUTDIR}/cxr_34567.pt \
    --device 6 \
    --n_epochs 30 \
    --train_dataset_class CheXpertRaceDataset \
    --test_dataset_class MIMICCXRRaceDataset

python scripts/train_classifier.py \
    --seed 45678 \
    --save_path ${OUTDIR}/cxr_45678.pt \
    --device 6 \
    --n_epochs 30 \
    --train_dataset_class CheXpertRaceDataset \
    --test_dataset_class MIMICCXRRaceDataset

python scripts/train_classifier.py \
    --seed 56789 \
    --save_path ${OUTDIR}/cxr_56789.pt \
    --device 6 \
    --n_epochs 30 \
    --train_dataset_class CheXpertRaceDataset \
    --test_dataset_class MIMICCXRRaceDataset