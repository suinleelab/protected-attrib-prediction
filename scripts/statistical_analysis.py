#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
from requests import get

from src.datasets.isicsex import ISICSexDataset
from src.utils import get_statistical_analysis_args
from src.datasets.cxrrace import MIMICCXRRaceDataset, CheXpertRaceDataset, DIAGNOSES
from torchvision import transforms


colors = {
        'train': '#6e260e',
        'test': '#50a0c0'
        }

def map_diagnosis(x):
    if x == 'lentigo NOS':
        return 'lentigo simplex'
    else: return x

def safenan(x):
    if isinstance(x, str):
        return False
    else:
        return np.isnan(x)

def counts_to_odds(labels, female_counts, male_counts):
    female_odds = []
    for i in range(len(female_counts)):
        aa = female_counts[i]
        ab = sum(female_counts) - aa
        ba = male_counts[i]
        bb = sum(male_counts) - ba
        try:
            odds = aa*bb/(ab*ba)
        except ZeroDivisionError:
            #print(labels[i], aa, ab, ba, bb)
            odds = np.nan
        female_odds.append(odds)
    return female_odds

def analyse_age_derm():
    ages = [5*i for i in range(18)]
    for split in ['train', 'test']:
        print(f"Analysing {split} set")
        ds = ISICSexDataset(None, split=split)
        ds.metadata = ds.metadata.query('age_approx == age_approx')
        metadata = ds.metadata
        female = metadata.query('sex == "female"')
        male = metadata.query('sex == "male"')
        #ages = sorted(metadata.age_approx.unique())
        female_counts = []
        male_counts = []
        for age in ages:
            if np.isnan(age):
                female_counts.append(len(female.query('age_approx != age_approx')))
                male_counts.append(len(male.query('age_approx != age_approx')))
            else:
                female_counts.append(len(female.query('age_approx == @age')))
                male_counts.append(len(male.query('age_approx == @age')))

        odds = counts_to_odds(ages, female_counts, male_counts)
        for age, ratio in zip(ages, odds):
            print(age, ratio)

def analyse_age_cxr():
    ages = [5*i for i in range(4, 19)]
    image_size = 224

    train_set = CheXpertRaceDataset(image_transforms=[transforms.Resize((image_size, image_size))])
    
    train_set.metadata.loc[:, 'Age'] = train_set.metadata.Age.apply(lambda x: 5 * round(x/5))
    test_set = MIMICCXRRaceDataset(image_transforms=[transforms.Resize((image_size, image_size)), ])
    test_set.metadata.rename(columns={'age': 'Age'}, inplace=True)
    test_set.metadata.loc[:, 'Age'] = test_set.metadata.Age.apply(lambda x: 5 * round(x/5))

    for i, ds in enumerate([train_set, test_set]):
        print(f"Analysing {'train' if i == 0 else 'test'} set")
        
        ds.metadata = ds.metadata.query('Age == Age')
        metadata = ds.metadata
        white = metadata.query('race == "WHITE"')
        black = metadata.query('race == "BLACK/AFRICAN AMERICAN"')
        white_counts = []
        black_counts = []

        for age in ages:
            if np.isnan(age):
                white_counts.append(len(white.query('Age != Age')))
                black_counts.append(len(black.query('Age != Age')))
            else:
                white_counts.append(len(white.query('Age == @age')))
                black_counts.append(len(black.query('Age == @age')))

        odds = counts_to_odds(ages, white_counts, black_counts)
        for age, ratio in zip(ages, odds):
            print(age, ratio)

def analyse_diagnoses_derm():
    count_threshold = 10
    ratios = {}
    counts = {}
    all_diagnoses = []
    for split in ['train', 'test']:
        ratios[split] = {}
        counts[split] = {}
        
        ds = ISICSexDataset(None, split=split)
        print(len(ds))
        ds.metadata = ds.metadata.query('diagnosis == diagnosis') # filter nan
        metadata = ds.metadata
        metadata.diagnosis = [map_diagnosis(diagnosis) for diagnosis in metadata.diagnosis]
        female = metadata.query('sex == "female"')
        male = metadata.query('sex == "male"')
        female_counts = []
        male_counts = []
        diagnoses = metadata.diagnosis.unique()
        all_diagnoses.extend(diagnoses)
        for diagnosis in diagnoses:
            if safenan(diagnosis):
                female_count = len(female.query('diagnosis != diagnosis'))
                male_count = len(male.query('diagnosis != diagnosis'))
            female_count = len(female.query('diagnosis == @diagnosis'))
            male_count = len(male.query('diagnosis == @diagnosis'))
            if split == 'test':
                print(diagnosis, female_count + male_count)
            female_counts.append(female_count)
            male_counts.append(male_count)
        for diagnosis, female_count, male_count in zip(diagnoses, female_counts, male_counts):
            counts[split][diagnosis] = female_count + male_count
        odds = counts_to_odds(diagnoses, female_counts, male_counts)
        for diagnosis, ratio in zip(diagnoses, odds):
            ratios[split][diagnosis] = ratio

    all_diagnoses = list(set(all_diagnoses))
    def _filter(d):
        try:
            if counts['train'][d] > count_threshold and counts['test'][d] > count_threshold:
                return True
            else:
                return False
        except KeyError:
            return False
    filtered_diagnoses = [d for d in all_diagnoses if _filter(d)]

    def sorting_key(x):
        if ratios['train'][x] > 1:
            return ratios['train'][x]
        elif ratios['train'][x] > 0:
            return 1/ratios['train'][x]
        else:
            return np.inf
    sorted_diagnoses = sorted(filtered_diagnoses, key=sorting_key, reverse=True)

    for d in sorted_diagnoses:
        if repr(d) == 'nan':
            continue
        print(
                d.ljust(40), 
                "{:.02f}".format(ratios['train'][d]).rjust(10),
                "{:.02f}".format(ratios['test'][d]).rjust(10),
                "{:04d}".format(counts['train'][d]).rjust(10),
                "{:04d}".format(counts['test'][d]).rjust(10)
                )

def analyse_diagnoses_cxr():
    count_threshold = 10
    ratios = {}
    counts = {}
    all_diagnoses = []
    for split in ['train', 'test']:
        ratios[split] = {}
        counts[split] = {}
        if split == 'train':
            ds = CheXpertRaceDataset(None, split=split)
        else:
            ds = MIMICCXRRaceDataset(None, split='train')

        metadata = ds.metadata
        all_diagnoses.extend(DIAGNOSES)

        white = metadata.query('race == "WHITE"')
        black = metadata.query('race == "BLACK/AFRICAN AMERICAN"')
        white_counts = []
        black_counts = []
        for diagnosis in DIAGNOSES:
            white_count = len(white.loc[white[diagnosis] == 1])
            black_count = len(black.loc[black[diagnosis] == 1])
            if split == 'test':
                print(diagnosis, white_count + black_count)
            white_counts.append(white_count)
            black_counts.append(black_count)
        
        print("split", split)

        for diagnosis, white_count, black_count in zip(DIAGNOSES, white_counts, black_counts):
            counts[split][diagnosis] = white_count + black_count
        odds = counts_to_odds(DIAGNOSES, black_counts, white_counts)
        for diagnosis, ratio in zip(DIAGNOSES, odds):
            ratios[split][diagnosis] = ratio

    all_diagnoses = list(set(all_diagnoses))
    def _filter(d):
        try:
            if counts['train'][d] > count_threshold and counts['test'][d] > count_threshold:
                return True
            else:
                return False
        except KeyError:
            return False
    filtered_diagnoses = [d for d in all_diagnoses if _filter(d)]
    print(filtered_diagnoses)

    def sorting_key(x):
        if ratios['train'][x] > 1:
            return ratios['train'][x]
        elif ratios['train'][x] > 0:
            return 1/ratios['train'][x]
        else:
            return np.inf
    sorted_diagnoses = sorted(filtered_diagnoses, key=sorting_key, reverse=True)

    for d in sorted_diagnoses:
        if repr(d) == 'nan':
            continue
        print(
                d.ljust(40), 
                "{:.02f}".format(ratios['train'][d]).rjust(10),
                "{:.02f}".format(ratios['test'][d]).rjust(10),
                "{:04d}".format(counts['train'][d]).rjust(10),
                "{:04d}".format(counts['test'][d]).rjust(10)
                )

def analyse_dermoscopic_types():
    ratios = {}
    counts = {}
    all_types = []
    for split in ['train', 'test']:
        ratios[split] = {}
        counts[split] = {}
        
        ds = ISICSexDataset(None, split=split)
        ds.metadata = ds.metadata.query('dermoscopic_type == dermoscopic_type') # filter nan
        metadata = ds.metadata
        female = metadata.query('sex == "female"')
        male = metadata.query('sex == "male"')
        female_counts = []
        male_counts = []
        types = metadata.dermoscopic_type.unique()
        all_types.extend(types)
        for dermoscopic_type in types:
            if safenan(dermoscopic_type):
                female_count = len(female.query('dermoscopic_type != dermoscopic_type'))
                male_count = len(male.query('dermoscopic_type != dermoscopic_type'))
            female_count = len(female.query('dermoscopic_type == @dermoscopic_type'))
            male_count = len(male.query('dermoscopic_type == @dermoscopic_type'))
            if split == 'test':
                print(dermoscopic_type, female_count + male_count)
            female_counts.append(female_count)
            male_counts.append(male_count)
        for dermoscopic_type, female_count, male_count in zip(types, female_counts, male_counts):
            counts[split][dermoscopic_type] = female_count + male_count
        odds = counts_to_odds(types, female_counts, male_counts)
        for dermoscopic_type, ratio in zip(types, odds):
            ratios[split][dermoscopic_type] = ratio
    print(counts, ratios)

if __name__ == "__main__":
    args = get_statistical_analysis_args()

    if args.dataset == 'derm':
        if args.task == 'diagnosis':
            analyse_diagnoses_derm()
        elif args.task == 'age':
            analyse_age_derm()
        elif args.task == 'dermoscopy_type':
            analyse_dermoscopic_types()
    elif args.dataset == 'cxr':
        if args.task == 'diagnosis':
            analyse_diagnoses_cxr()
        elif args.task == 'age':
            analyse_age_cxr()
        elif args.task == 'dermoscopy_type':
            raise ValueError("Dermoscopy type analysis is only applicable for derm datasets.")
    else:
        raise ValueError(f"{args.dataset} modality is not included in the current analysis. Please choose from 'derm' or 'cxr'.")
