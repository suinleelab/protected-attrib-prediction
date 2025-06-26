#!/usr/bin/env python
import numpy as np
import pandas as pd

from .isicsex import ISICSexDataset, BASE_DIR_ISIC

def equalize_with_sex(grouped_df, metadata_downsampled, seed=23456):
    group_1 =  grouped_df.loc[grouped_df['sex'] == 'male']
    group_2 = grouped_df.loc[grouped_df['sex'] == 'female']

    if len(group_1) > len(group_2):
        group_1 = group_1.sample(len(group_2), random_state=seed)
    else:
        group_2 = group_2.sample(len(group_1), random_state=seed)

    grouped_concat = pd.concat([group_1, group_2], axis=0)
    if len(metadata_downsampled) == 0:
        metadata_downsampled = grouped_concat
    else:
        metadata_downsampled = pd.concat([metadata_downsampled, grouped_concat], axis=0) 

    return metadata_downsampled

def equalize_with_sex_trainval(grouped_df, metadata_train, metadata_val, seed=None, val_portion=0.1):
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    group_1 =  grouped_df.loc[grouped_df['sex'] == 'male']
    group_2 = grouped_df.loc[grouped_df['sex'] == 'female']

    n_available_samples = min(len(group_1), len(group_2))
    n_val_samples = int(n_available_samples*val_portion)
    n_train_samples = n_available_samples - n_val_samples

    group_1_val_idxs = rng.choice(
            np.arange(len(group_1)),
            size=n_val_samples,
            replace=False
            )
    group_1_train_idx_options = np.setdiff1d(
            np.arange(len(group_1)),
            group_1_val_idxs
            )
    group_1_train_idxs = rng.choice(
            group_1_train_idx_options,
            size=n_train_samples,
            replace=False
            )

    group_2_val_idxs = rng.choice(
            np.arange(len(group_2)),
            size=n_val_samples,
            replace=False
            )
    group_2_train_idx_options = np.setdiff1d(
            np.arange(len(group_2)),
            group_2_val_idxs
            )
    group_2_train_idxs = rng.choice(
            group_2_train_idx_options,
            size=n_train_samples,
            replace=False
            )

    grouped_val = pd.concat([
        group_1.iloc[group_1_val_idxs],
        group_2.iloc[group_2_val_idxs]],
        axis=0
        )
    grouped_train = pd.concat([
        group_1.iloc[group_1_train_idxs],
        group_2.iloc[group_2_train_idxs]],
        axis=0
        )

    if len(metadata_train) == 0:
        metadata_train = grouped_train
    else:
        metadata_train = pd.concat([metadata_train, grouped_train], axis=0)

    if len(metadata_val) == 0:
        metadata_val = grouped_val
    else:
        metadata_val = pd.concat([metadata_val, grouped_val], axis=0)

    return metadata_train, metadata_val

class BalancedBinaryHairISICSexDataset(ISICSexDataset):
    def __init__(self, 
            transform, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            seed=12345, 
            val_portion=0.1,
            metadata_path='data/metadata_with_hair.csv'):

        self._initialize_without_trainval_split(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir, 
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion)

        sex_grouped = self.metadata.groupby('hair')

        if self.split == 'test':
            metadata_downsampled = []
            for hair in [True, False]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_downsampled = equalize_with_sex(grouped_sex_df, metadata_downsampled)

            metadata_downsampled = metadata_downsampled.reset_index(drop=True)
            self.metadata = metadata_downsampled
        else:
            metadata_train = []
            metadata_val = []
            for hair in [True, False]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_train, metadata_val = equalize_with_sex_trainval(
                        grouped_sex_df, 
                        metadata_train, 
                        metadata_val,
                        self.seed,
                        self.val_portion
                        )
            if self.split == 'train':
                self.metadata = metadata_train.reset_index(drop=True)
            elif self.split == 'val':
                self.metadata = metadata_val.reset_index(drop=True)


class BalancedQuantitativeHairISICSexDataset(ISICSexDataset):
    hair_grades_path = 'labeling/hair_grades.csv'
    def __init__(self, 
            transform, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            seed=12345, 
            val_portion=0.1):

        self._initialize_without_trainval_split(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir, 
                metadata_path=BASE_DIR_ISIC+'/metadata.csv',
                seed=seed,
                val_portion=val_portion)

        # incorporate hair grades into metadata
        hair_grades_df = pd.read_csv(self.hair_grades_path)
        grades = []
        for i in range(len(self.metadata)):
            isic_id = self.metadata.isic_id.iloc[i]
            grade = hair_grades_df.query('isic_id == @isic_id')
            if len(grade) == 0:
                grades.append(np.nan)
            elif len(grade) == 1:
                grades.append(int(grade.grade_alex))
            elif len(grade) > 1:
                raise AssertionError(f'multiple entries for isic_id {isic_id}')
        self.metadata.insert(len(self.metadata.columns), 'hair', grades)
        self.metadata = self.metadata.query('hair == hair') # filters nan entries

        sex_grouped = self.metadata.groupby('hair')

        if self.split == 'test':
            metadata_downsampled = []
            for hair in [1, 2, 3, 4, 5]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_downsampled = equalize_with_sex(grouped_sex_df, metadata_downsampled)

            metadata_downsampled = metadata_downsampled.reset_index(drop=True)
            self.metadata = metadata_downsampled

        else:
            metadata_train = []
            metadata_val = []
            for hair in [True, False]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_train, metadata_val = equalize_with_sex_trainval(
                        grouped_sex_df, 
                        metadata_train, 
                        metadata_val,
                        self.seed,
                        self.val_portion
                        )
            if self.split == 'train':
                self.metadata = metadata_train.reset_index(drop=True)
            elif self.split == 'val':
                self.metadata = metadata_val.reset_index(drop=True)

class BalancedAgeISICSexDataset(ISICSexDataset):
    def __init__(self, 
            transform, 
            verbose=True, 
            split='train', 
            img_dir=BASE_DIR_ISIC, 
            metadata_path=BASE_DIR_ISIC+'/metadata.csv', 
            seed=12345, 
            val_portion=0.1):

        self._initialize_without_trainval_split(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir, 
                metadata_path=metadata_path,
                seed=seed,
                val_portion=val_portion)


        age_keys = list(self.metadata.groupby('age_approx').groups.keys())
        age_grouped = self.metadata.groupby('age_approx')

        if self.split == 'test':
            metadata_downsampled = []
            for age in age_keys:
                grouped_age_df = age_grouped.get_group(age)
                metadata_downsampled = equalize_with_sex(grouped_age_df, metadata_downsampled)

            metadata_downsampled = metadata_downsampled.reset_index(drop=True)
        else:
            metadata_train = []
            metadata_val = []
            for age in age_keys:
                grouped_age_df = age_grouped.get_group(age)
                metadata_train, metadata_val = equalize_with_sex_trainval(
                        grouped_age_df, 
                        metadata_train, 
                        metadata_val,
                        self.seed,
                        self.val_portion
                        )
            if self.split == 'train':
                self.metadata = metadata_train.reset_index(drop=True)
            elif self.split == 'val':
                self.metadata = metadata_val.reset_index(drop=True)

class BalancedBinaryHairManualISICSexDataset(ISICSexDataset):
     def __init__(self, 
             transform, 
             verbose=True, 
             split='train', 
             img_dir=BASE_DIR_ISIC,
             seed=12345,
             val_portion=0.1):

        self._initialize_without_trainval_split(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir, 
                metadata_path='data/metadata_with_hair.csv',
                seed=seed,
                val_portion=val_portion)

        male_manual_labels_isic_ids = list(pd.read_csv('data/male_hair_labels_manual.csv', header=None)[0].apply(lambda x: x[:-4]))
        female_manual_labels_isic_ids = list(pd.read_csv('data/female_hair_labels_manual.csv', header=None)[0].apply(lambda x: x[:-4]))
        self.metadata = self.metadata.query("isic_id in @male_manual_labels_isic_ids or isic_id in @female_manual_labels_isic_ids")

        sex_grouped = self.metadata.groupby('hair')

        if self.split == 'test':
            metadata_downsampled = []
            for hair in [True, False]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_downsampled = equalize_with_sex(grouped_sex_df, metadata_downsampled)

            metadata_downsampled = metadata_downsampled.reset_index(drop=True)
            self.metadata = metadata_downsampled
        else:
            metadata_train = []
            metadata_val = []
            for hair in [True, False]:
                grouped_sex_df = sex_grouped.get_group(hair)
                metadata_train, metadata_val = equalize_with_sex_trainval(
                        grouped_sex_df, 
                        metadata_train, 
                        metadata_val,
                        self.seed,
                        self.val_portion
                        )
            if self.split == 'train':
                self.metadata = metadata_train.reset_index(drop=True)
            elif self.split == 'val':
                self.metadata = metadata_val.reset_index(drop=True)            

class BalancedHairAndStickerISICSexDataset(BalancedBinaryHairISICSexDataset):
    def __init__(self, 
            transform, 
            verbose=True,
            split='train', 
            img_dir=BASE_DIR_ISIC,
            seed=12345,
            val_portion=0.1):

        self._initialize_without_trainval_split(
                transform, 
                verbose=verbose, 
                split=split, 
                img_dir=img_dir, 
                metadata_path='data/metadata_with_sticker_and_hair.csv',
                seed=seed,
                val_portion=val_portion)

        if split == 'test':
            metadata_downsampled = []

            for hair in [True, False]:
                metadata_filtered = self.metadata.loc[self.metadata['hair'] == hair]
                sticker_grouped = metadata_filtered.groupby('sticker')

                for sticker in sticker_grouped.groups.keys():
                    grouped_sticker_df = metadata_filtered.loc[metadata_filtered['sticker'] == sticker]
                    metadata_downsampled = equalize_with_sex(grouped_sticker_df, metadata_downsampled)
            
                metadata_downsampled = metadata_downsampled.reset_index(drop=True)
    
            self.metadata = metadata_downsampled
        else:
            metadata_train = []
            metadata_val = []
            for hair in [True, False]:
                metadata_filtered = self.metadata.loc[self.metadata['hair'] == hair]
                sticker_grouped = metadata_filtered.groupby('sticker')

                for sticker in sticker_grouped.groups.keys():
                    grouped_sticker_df = metadata_filtered.loc[metadata_filtered['sticker'] == sticker]
                    metadata_train, metadata_val = equalize_with_sex_trainval(
                            grouped_sticker_df, 
                            metadata_train, 
                            metadata_val,
                            self.seed,
                            self.val_portion
                            )
            if self.split == 'train':
                self.metadata = metadata_train.reset_index(drop=True)
            elif self.split == 'val':
                self.metadata = metadata_val.reset_index(drop=True)
        
        print("split=", split)
        print(pd.crosstab(self.metadata['hair'], self.metadata['sex']))
        print(pd.crosstab(self.metadata['sticker'], self.metadata['sex']))