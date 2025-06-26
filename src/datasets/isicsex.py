#!/usr/bin/env python
import glob
from PIL import Image

import numpy as np
import pandas as pd
import torch
import os

BASE_DIR_ISIC = os.environ['BASE_DIR_ISIC']

TRAIN_SITES = [
        "Anonymous",
        "Department of Dermatology, Hospital Clínic de Barcelona",
        "Department of Dermatology, Medical University of Vienna",
        "ViDIR Group, Department of Dermatology, Medical University of Vienna",
        "Dermoscopedia",
        "For educational purpose only",
        "Hospital Clínic de Barcelona",
        "Hospital Italiano de Buenos Aires",
        "Konstantinos Liopyris"
        ]

TEST_SITES = [
        "Memorial Sloan Kettering Cancer Center",
        "Sydney Melanoma Diagnostic Center at Royal Prince Alfred Hospital",
        "Sydney Melanoma Diagnostic Center at Royal Prince Alfred Hospital, Pascale Guitera",
        "The University of Queensland Diamantina Institute, The University of Queensland, Dermatology Research Centre"
        ]

class ISICSexDataset(torch.utils.data.Dataset):
    def __init__(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1):

        self._initialize_without_trainval_split(
                image_transforms=image_transforms,
                verbose=verbose,
                split=split,
                img_dir=img_dir,
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion
                )

        if self.split in ['train', 'val']:
            rng = np.random.default_rng(self.seed)
            val_idxs = rng.choice(
                    np.arange(len(self.metadata)),
                    size=int(len(self.metadata)*self.val_portion),
                    replace=False)
            train_idxs = np.setdiff1d(
                    np.arange(len(self.metadata)),
                    val_idxs
                    )
            if self.split == 'train':
                self.metadata = self.metadata.iloc[train_idxs]
            elif self.split == 'val':
                self.metadata = self.metadata.iloc[val_idxs]

    def _initialize_without_trainval_split(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1):

        self.img_dir = img_dir
        self.transforms = image_transforms
        self.metadata = pd.read_csv(metadata_path)
        self.split = split
        self.seed = seed
        self.val_portion = val_portion

        original_len = len(self.metadata)
        self.metadata = self.metadata.query("sex=='male' | sex=='female'")
        new_len = len(self.metadata)
        if verbose:
            print(f"...Removed {original_len-new_len} images without patient sex label")

        original_len = len(self.metadata)
        self.metadata = self.metadata.query('image_type == "dermoscopic"')
        new_len = len(self.metadata)
        if verbose:
            print(f"...Removed {original_len-new_len} non-dermoscopic images")
        self._verify()

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f'split must be one of "train", "val", "test". (Received "{self.split}")')
        if self.split in ['train', 'val']:
            self.metadata = self.metadata.query('attribution in @TRAIN_SITES')
        elif self.split == 'test':
            self.metadata = self.metadata.query('attribution in @TEST_SITES')


    def _verify(self):
        '''Verify that all relevant images listed in metadata.csv are also 
        present in the image directory.'''
        img_list = [x.split("/")[-1] for x in glob.glob(self.img_dir + "/*.jpg")]
        img_list = [img[:-4] for img in img_list]
        img_list = set(img_list)
        for i in range(len(self)):
            assert self.metadata.isic_id.iloc[i] in img_list
            assert self.metadata.sex.iloc[i] == self.metadata.sex.iloc[i] # false if NaN
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        isic_id = self.metadata.isic_id.iloc[idx]
        image = Image.open(self.img_dir + "/" + isic_id+'.jpg')
        image = self.transforms(image)
        if self.metadata.sex.iloc[idx] == 'male':
            label = 1
        else:
            label = 0
        return image, label


class ISICSexDatasetSegmented(torch.utils.data.Dataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv'):

        self.img_dir = img_dir
        self.transforms = image_transforms
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        isic_id = self.metadata.image_id.iloc[idx]
        image = Image.open(self.img_dir + "/" + isic_id + '.jpg')
        segmentation = Image.open(self.img_dir + "/" + isic_id + '_segmentation.png')

        only_lesion_image = Image.fromarray(np.array(image) * np.array(segmentation)[:, :, np.newaxis])/255
        only_lesion_image = only_lesion_image.astype(int)
        only_background_image = Image.fromarray(np.array(image) * ~np.array(segmentation)[:, :, np.newaxis])/255
        only_background_image = only_background_image.astype(int)
        
        inpainted_image = Image.open(self.img_dir + "/" + isic_id + '_inpaint_lesion.jpg')

        image = self.transforms(image)
        inpainted_image = self.transforms(inpainted_image)
        only_lesion_image = self.transforms(only_lesion_image)
        only_background_image = self.transforms(only_background_image)

        if self.metadata.sex.iloc[idx] == 'male':
            label = 1
        else:
            label = 0

        return image, inpainted_image, only_lesion_image, only_background_image, label