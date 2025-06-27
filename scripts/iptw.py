#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn.linear_model

from src.datasets.cxrrace import MIMICCXRRaceDataset
from src.datasets.isicsex import ISICSexDataset
import argparse
import os

def diagnosis_map(diagnosis):
    if diagnosis == 'lentigo NOS':
        return 'lentigo simplex'
    else:
        return diagnosis

def combinations(signals):
    c = []
    c.append([])
    for a in signals:
        c.append([a])
    c.append(signals)
    return c

def main(
    output_paths, 
    signals, 
    dataset='derm'
):
    if dataset == 'derm':
        test_ds = ISICSexDataset(None, split='test')
    elif dataset == 'cxr':
        test_ds = MIMICCXRRaceDataset(None)
    else:
        raise ValueError(f"{dataset} modality is not included in the current analysis. \
                          Please choose from 'cxr' or 'derm'.")
    
    signal_ds = pd.read_csv(f"data/{dataset}_annotated_all_images.csv")
    all_paths = set(signal_ds.path)
    test_ds.metadata = test_ds.metadata.query("path in @all_paths")

    if 'age' in signals:
        age_list = [age for age in test_ds.metadata.age]
        signal_ds['age']= age_list

    signal_ds.set_index('path', inplace=True)

    signal_column = []
    classifier_column = []
    performance_column = []
    # calculate performance for each classifier replicate
    for signal_set in combinations(signals):
        print(signal_set)
        for output_path in output_paths:
            output_ds = pd.read_csv(output_path)
            output_ds = output_ds.query('path in @all_paths')
            # check that the paths are aligned
            for i in range(len(test_ds)):
                assert test_ds.metadata.path.iloc[i] == output_ds.path.iloc[i]

            if len(signal_set) > 0:
                # create a dataset of the signals of interest, to be used for the 
                # logistic regression model that calculates propensity scores
                signal_dict = {}
                for signal in signal_set:
                    signal_dict[signal] = [signal_ds.loc[path][signal] for path in test_ds.metadata.path]
                
                if dataset == 'derm':
                    df = pd.DataFrame.from_dict({
                        **signal_dict,
                        'sex': [0 if sex=='female' else 1 for sex in test_ds.metadata.sex]
                    })
                elif dataset == 'cxr':
                    df = pd.DataFrame.from_dict({
                        **signal_dict,
                        'race': [1 if race=='BLACK/AFRICAN AMERICAN' else 0 for race in test_ds.metadata.race]
                    })
                else:
                    raise ValueError(f"{dataset} modality is not included in the current analysis. \
                                      Please choose from 'cxr' or 'derm'.")

                # x and y for the propensity score model
                x = np.stack([np.array(df[signal]) for signal in signal_set], axis=1)
                y = df.race
                # scale inputs for more stable learning
                x_transformed = sklearn.preprocessing.StandardScaler().fit_transform(x)

                # fit the logistic regression model and get propensity scores
                lm = sklearn.linear_model.LogisticRegression()
                lm.fit(x_transformed, y)
                propensity = lm.predict_proba(x_transformed)

                # convert propensity scores to inverse probability of `treatment' weights.
                iptw = [1/propensity[i][1] if y[i] == 1 else 1/propensity[i][0] for i in range(len(test_ds))]
                df['iptw'] = iptw
                auc = roc_auc_score(output_ds.label, output_ds.prediction, sample_weight=df.iptw)
                print('weighted: ', '{:.04f}'.format(auc), ' || {:.04f} - {:.04f}'.format(min(iptw), max(iptw)))
            else:
                auc = roc_auc_score(output_ds.label, output_ds.prediction)
                print('unweighted: ', '{:.04f}'.format(auc))

            signal_column.append(signal_set)
            classifier_column.append(output_path)
            performance_column.append(auc)

    df = pd.DataFrame.from_dict({
        'signal': signal_column,
        'classifier': classifier_column,
        'performance': performance_column
        })
    df.to_csv(f'data/{dataset}_iptw_quantification.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Type of dataset to use. Either "derm" or "cxr" are supported for now.',
        default='derm',
        type=str
    )
    parser.add_argument('--signals',
            help='identified signals to quantify',
            nargs='+',
            default='hair',
            type=str
    )
    args = parser.parse_args()

    all_seeds = [12345, 23456, 34567, 45678, 56789]
    model_paths = [f"{os.environ['OUT_DIR']}/{args.dataset}_{seed}.pt" for seed in all_seeds]
    output_paths = []

    for model_path in model_paths:
        name = os.path.splitext(os.path.basename(model_path))[0]
        output_paths.append(os.path.join('data/', name+'_predictions.csv'))

    main(
        output_paths, 
        args.signals, 
        args.dataset
    )
