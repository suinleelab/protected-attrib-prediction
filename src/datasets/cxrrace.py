#!/usr/bin/env python
import glob
from PIL import Image

import numpy as np
import pandas as pd
import torch
import os

BASE_DIR_CHEXPERT = os.environ['BASE_DIR_CHEXPERT']
BASE_DIR_MIMIC = os.environ['BASE_DIR_MIMIC']

DIAGNOSES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']


class CXRBaseDataset(torch.utils.data.Dataset):
    def __init__(self,
            image_transforms,
            img_dir, 
            metadata_path,
            split='train', 
            seed=12345, 
            val_portion=0.1,
            test_portion=0.1
        ):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata.race.notna()].query("race == 'WHITE' or race == 'BLACK/AFRICAN AMERICAN'")
        self.img_dir = img_dir
        self.transforms = image_transforms
        self.seed = seed

        rng = np.random.default_rng(self.seed)
        test_idxs = rng.choice(
                np.arange(len(self.metadata)),
                size=int(len(self.metadata)*test_portion),
                replace=False)
        train_idxs = np.setdiff1d(
                np.arange(len(self.metadata)),
                test_idxs
                )

        val_idxs = rng.choice(
                train_idxs,
                size=int(len(self.metadata)*val_portion),
                replace=False)
        train_idxs = np.setdiff1d(
                train_idxs,
                val_idxs
                )

        if split == 'train':
            self.metadata = self.metadata.iloc[train_idxs]
        elif split == 'val':
            self.metadata = self.metadata.iloc[val_idxs]
        else:
            self.metadata = self.metadata.iloc[test_idxs]


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image = Image.open(self.img_dir + "/" + row.path).convert("RGB")
        image = self.transforms(image)
        if row.race == 'WHITE':
            label = 0
        elif row.race == 'BLACK/AFRICAN AMERICAN':
            label = 1

        return image, label


class CheXpertRaceDataset(CXRBaseDataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_CHEXPERT, 
            metadata_path=BASE_DIR_CHEXPERT+'/chexpert_batch_1_valid_and_csv/train_with_race_labels.csv',
            split='train', 
            seed=12345, 
            val_portion=0.1,
            test_portion=0.1
        ):
        super().__init__(image_transforms,
        img_dir,
        metadata_path,
        split,
        seed,
        val_portion,
        test_portion)
    


class MIMICCXRRaceDataset(CXRBaseDataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_MIMIC+"/files", 
            metadata_path=BASE_DIR_MIMIC+'/metadata_with_race_frontal_and_lateral.csv',
            split='train', 
            seed=12345, 
            val_portion=0,
            test_portion=0
        ):
        super().__init__(image_transforms,
        img_dir,
        metadata_path,
        split,
        seed,
        val_portion,
        test_portion)