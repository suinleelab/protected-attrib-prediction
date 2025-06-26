#!/usr/bin/env python
from PIL import Image
import pandas as pd
import torch

from .isicsex import ISICSexDataset, BASE_DIR_ISIC
from .cxrrace import CheXpertRaceDataset, BASE_DIR_CHEXPERT, BASE_DIR_MIMIC
import numpy as np

MELANOMAS = {
        'melanoma'
        # 'melanoma metastasis'
        }
LOOKALIKES = {
        'seborrheic keratosis',
        'solar lentigo',
        'lentigo NOS',
        'nevus',
        'pigmented benign keratosis',
        'dermatofibroma',
        'lentigo simplex',
        'cafe-au-lait macule'
        }

class MelanomaDataset(ISICSexDataset):
    def _initialize_without_trainval_split(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1
            ):
        super()._initialize_without_trainval_split(
                image_transforms=image_transforms,
                verbose=verbose,
                split=split,
                img_dir=img_dir,
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion
                )
        self.metadata = self.metadata.query('diagnosis in @MELANOMAS | diagnosis in @LOOKALIKES')

    def __getitem__(self, idx):
        isic_id = self.metadata.isic_id.iloc[idx]
        image = Image.open(self.img_dir + "/" + isic_id+'.jpg')
        image = self.transforms(image)
        if self.metadata.diagnosis.iloc[idx] in MELANOMAS:
            label = 1
        else:
            label = 0
        return image, label

class ISICClusterDataset(ISICSexDataset):
    def __init__(self, transform, verbose=True, split='train', img_dir=BASE_DIR_ISIC, filter_sex=None):
        super().__init__(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir)
        
        if filter_sex is not None:
            self.metadata = self.metadata.query("sex == @filter_sex")
    
    def __getitem__(self, idx):
        isic_id = self.metadata.isic_id.iloc[idx]
        image = Image.open(self.img_dir + "/" + isic_id+'.jpg')
        image = self.transforms(image)
        if self.metadata.sex.iloc[idx] == 'male':
            label = 1
        else:
            label = 0
        return image, label, isic_id

class EngineeredMelanomaDataset(MelanomaDataset):
    def __init__(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1,
            odds_ratio=2):

        self.odds_ratio = odds_ratio
        super().__init__(
            image_transforms=image_transforms, 
            verbose=verbose, 
            split=split, 
            img_dir=img_dir, 
            metadata_path=metadata_path,
            seed=seed,
            val_portion=val_portion)

    def _initialize_without_trainval_split(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1
            ):

        super()._initialize_without_trainval_split(
                image_transforms=image_transforms,
                verbose=verbose,
                split=split,
                img_dir=img_dir,
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion
                )
        df00 = self.metadata.query('sex == "male"   & diagnosis in @LOOKALIKES')
        df01 = self.metadata.query('sex == "male"   & diagnosis in @MELANOMAS')
        df10 = self.metadata.query('sex == "female" & diagnosis in @LOOKALIKES')
        df11 = self.metadata.query('sex == "female" & diagnosis in @MELANOMAS')
        print(len(df00), len(df01), len(df10), len(df11))

        sqrt_or = self.odds_ratio**0.5
        if len(df00) < len(df01)*sqrt_or:
            df01 = df01.sample(int(len(df00)/sqrt_or), random_state=self.seed)
        elif len(df00) > len(df01)*sqrt_or:
            df00 = df00.sample(int(len(df01)*sqrt_or), random_state=self.seed)

        if len(df10)*sqrt_or < len(df11):
            df11 = df11.sample(int(len(df10)*sqrt_or), random_state=self.seed)
        elif len(df10)*sqrt_or > len(df11):
            df10 = df10.sample(int(len(df11)/sqrt_or), random_state=self.seed)

        self.metadata = pd.concat([df00, df01, df10, df11], ignore_index=True)

        # checks
        df00 = self.metadata.query('sex == "male"   & diagnosis in @LOOKALIKES')
        df01 = self.metadata.query('sex == "male"   & diagnosis in @MELANOMAS')
        df10 = self.metadata.query('sex == "female" & diagnosis in @LOOKALIKES')
        df11 = self.metadata.query('sex == "female" & diagnosis in @MELANOMAS')
        print(len(df00), len(df01), len(df10), len(df11))


class CXRBaseDiagnosisDataset(torch.utils.data.Dataset):
    def __init__(self,
            image_transforms,
            img_dir, 
            metadata_path,
            split='train', 
            seed=12345, 
            val_portion=0.1,
            test_portion=0.1,
            target='Pleural Effusion'
        ):
        self.target = target
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[(self.metadata[target] == 0) | (self.metadata[target] == 1)]
            
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
        # print(row[self.target])
        return image, int(row[self.target])

class CheXpertDiagnosisDataset(CXRBaseDiagnosisDataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_CHEXPERT, 
            metadata_path=BASE_DIR_CHEXPERT+'/chexpert_batch_1_valid_and_csv/train_with_race_labels.csv',
            split='train', 
            seed=12345, 
            val_portion=0.1,
            test_portion=0.1,
            target='Pleural Effusion'
        ):
        super().__init__(image_transforms,
        img_dir,
        metadata_path,
        split,
        seed,
        val_portion,
        test_portion,
        target)
    
class MIMICCXRDiagnosisDataset(CXRBaseDiagnosisDataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_MIMIC+"/files/", 
            metadata_path=BASE_DIR_MIMIC+'/metadata_with_race.csv',
            split='train', 
            seed=12345, 
            val_portion=0,
            test_portion=0,
            target='Pleural Effusion'
        ):
        super().__init__(image_transforms,
        img_dir,
        metadata_path,
        split,
        seed,
        val_portion,
        test_portion,
        target)

class EngineeredMelanomaDatasetDiscrimination(MelanomaDataset):
    def __init__(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1,
            target='female',
            odds_ratio=2):

        self.odds_ratio = odds_ratio
        self.target = target
        super().__init__(
            image_transforms=image_transforms, 
            verbose=verbose, 
            split=split, 
            img_dir=img_dir, 
            metadata_path=metadata_path,
            seed=seed,
            val_portion=val_portion)

    def _initialize_without_trainval_split(self,
            image_transforms, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv',
            seed=12345,
            val_portion=0.1
            ):

        super()._initialize_without_trainval_split(
                image_transforms=image_transforms,
                verbose=verbose,
                split=split,
                img_dir=img_dir,
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion
                )

        df00 = self.metadata.query('sex == "male"   & diagnosis in @LOOKALIKES')
        df01 = self.metadata.query('sex == "male"   & diagnosis in @MELANOMAS')
        df10 = self.metadata.query('sex == "female" & diagnosis in @LOOKALIKES')
        df11 = self.metadata.query('sex == "female" & diagnosis in @MELANOMAS')
        print(len(df00), len(df01), len(df10), len(df11))

        if self.target == 'female':
            if len(df10)*self.odds_ratio < len(df11):
                df11 = df11.sample(int(len(df10)*self.odds_ratio), random_state=self.seed)
            elif len(df10)*self.odds_ratio > len(df11):
                df10 = df10.sample(int(len(df11)/self.odds_ratio), random_state=self.seed)
        elif self.target == 'male':
            if len(df00) < len(df01)*self.odds_ratio:
                df01 = df01.sample(int(len(df00)/self.odds_ratio), random_state=self.seed)
            elif len(df00) > len(df01)*self.odds_ratio:
                df00 = df00.sample(int(len(df01)*self.odds_ratio), random_state=self.seed)
        else:
            raise ValueError(f"invalid target {self.target}. Must be one of ['female', 'male']")


        self.metadata = pd.concat([df00, df01, df10, df11], ignore_index=True)

        # checks
        df00 = self.metadata.query('sex == "male"   & diagnosis in @LOOKALIKES')
        df01 = self.metadata.query('sex == "male"   & diagnosis in @MELANOMAS')
        df10 = self.metadata.query('sex == "female" & diagnosis in @LOOKALIKES')
        df11 = self.metadata.query('sex == "female" & diagnosis in @MELANOMAS')
        print(len(df00), len(df01), len(df10), len(df11))
        try:
            print(len(df00)*len(df11)/(len(df10)*len(df01)))
        except ZeroDivisionError:
            print('inf')


class EngineeredCheXpertDataset(CheXpertDiagnosisDataset):
     def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_CHEXPERT, 
            metadata_path=BASE_DIR_CHEXPERT+'/chexpert_batch_1_valid_and_csv/train_with_race_labels.csv',
            split='train', 
            seed=12345, 
            val_portion=0.1,
            test_portion=0.1,
            odds_ratio=2,
            target='Pleural Effusion'
        ):
        self.odds_ratio = odds_ratio
        self.target = target
        
        super().__init__(
            image_transforms=image_transforms,
            img_dir=img_dir,
            metadata_path=metadata_path,
            split=split,
            seed=seed,
            val_portion=val_portion,
            test_portion=test_portion
        )

        df00 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 1)]
        df01 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 0)]
        df10 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 1)]
        df11 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 0)]
        print("Odds ratio:", self.odds_ratio)
        print("Before equalization")
        print(len(df00), len(df01), len(df10), len(df11))

        sqrt_or = self.odds_ratio**0.5
        if len(df00) < len(df01)*sqrt_or:
            df01 = df01.sample(int(len(df00)/sqrt_or), random_state=self.seed)
        elif len(df00) > len(df01)*sqrt_or:
            df00 = df00.sample(int(len(df01)*sqrt_or), random_state=self.seed)

        if len(df10)*sqrt_or < len(df11):
            df11 = df11.sample(int(len(df10)*sqrt_or), random_state=self.seed)
        elif len(df10)*sqrt_or > len(df11):
            df10 = df10.sample(int(len(df11)/sqrt_or), random_state=self.seed)

        self.metadata = pd.concat([df00, df01, df10, df11], ignore_index=True)

        # checks
        df00 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 1)]
        df01 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 0)]
        df10 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 1)]
        df11 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 0)]
        print("After equalization")
        print(len(df00), len(df01), len(df10), len(df11))

class EngineeredMIMICCXRDataset(MIMICCXRDiagnosisDataset):
     def __init__(self,
            image_transforms,
            seed=12345,
            odds_ratio=2,
            target='Pleural Effusion'
        ):
        self.odds_ratio = odds_ratio
        self.target = target
        
        super().__init__(
            image_transforms=image_transforms,
            split='train',
            seed=seed,
            val_portion=0,
            test_portion=0
        )

        df00 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 1)]
        df01 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 0)]
        df10 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 1)]
        df11 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 0)]
        print("Odds ratio:", self.odds_ratio)
        print("Before equalization")
        print(len(df00), len(df01), len(df10), len(df11))

        sqrt_or = self.odds_ratio**0.5
        if len(df00) < len(df01)*sqrt_or:
            df01 = df01.sample(int(len(df00)/sqrt_or), random_state=self.seed)
        elif len(df00) > len(df01)*sqrt_or:
            df00 = df00.sample(int(len(df01)*sqrt_or), random_state=self.seed)

        if len(df10)*sqrt_or < len(df11):
            df11 = df11.sample(int(len(df10)*sqrt_or), random_state=self.seed)
        elif len(df10)*sqrt_or > len(df11):
            df10 = df10.sample(int(len(df11)/sqrt_or), random_state=self.seed)

        self.metadata = pd.concat([df00, df01, df10, df11], ignore_index=True)

        # checks
        df00 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 1)]
        df01 = self.metadata[(self.metadata['race'] == 'WHITE') & (self.metadata[target] == 0)]
        df10 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 1)]
        df11 = self.metadata[(self.metadata['race'] == 'BLACK/AFRICAN AMERICAN') & (self.metadata[target] == 0)]
        print("After equalization")
        print(len(df00), len(df01), len(df10), len(df11))

class ISICAttributeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_dir, image_transforms, attribute):
        self.img_dir = img_dir
        self.transforms = image_transforms
        self.metadata = pd.read_csv(data_dir)
        self.attribute = attribute

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        df_row = self.metadata.iloc[idx]
        isic_id = df_row.isic_id
        if not isic_id.endswith(".jpg"):
            isic_id = isic_id + ".jpg"

        image = Image.open(self.img_dir + "/" + isic_id)
        image = self.transforms(image)
        label = df_row[self.attribute]

        return image, label*1


class CXRAttributeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_dir, image_transforms, attribute):
        self.img_dir = img_dir
        self.transforms = image_transforms
        self.metadata = pd.read_csv(data_dir)
        self.attribute = attribute

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        df_row = self.metadata.iloc[idx]
        path = df_row.path

        image = Image.open(self.img_dir+"/files/"+ path).convert("RGB")
        image = self.transforms(image)
        label = df_row[self.attribute]

        return image, label*1

class CXRClusterDataset(CheXpertRaceDataset):
    def __init__(self,
            image_transforms,
            img_dir=BASE_DIR_CHEXPERT, 
            metadata_path=BASE_DIR_CHEXPERT+'/chexpert_batch_1_valid_and_csv/train_with_race_labels.csv',
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

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        path = row.path
        image = Image.open(self.img_dir + "/" + row.path).convert("RGB")
        image = self.transforms(image)
        if row.race == 'WHITE':
            label = 0
        elif row.race == 'BLACK/AFRICAN AMERICAN':
            label = 1

        return image, label, path