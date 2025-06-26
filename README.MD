# Towards a framework for discovering mechanisms underlying AI prediction of protected attributes

This repository provides code and step-by-step instructions to execute the framework introduced in the paper "Towards a framework for discovering mechanisms underlying AI prediction of protected attributes", as shown in the figure below.

![Concept Figure](figures/concept_figure.pdf)

## Installation
1. Git clone this repository
2. ```cd protected-attrib-predicition```
3. Create and activate the specified conda environment by running

```
conda env create -f environment.yml
conda activate protected-attrib-prediciton-env
```
4. Install the protected-atttrib-prediction package and necessary dependencies for development by running pip install -e ".[dev]"

## Set up directory paths
Create the following environment variables for the base directory to the datasets and the output directory to store the trained models.

```
export BASE_DIR_CHEXPERT=<PATH_TO_CHEXPERT_DIR>
export BASE_DIR_MIMIC=<PATH_TO_MIMIC_DIR>
export BASE_DIR_ISIC=<PATH_TO_ISIC_DIR>
export OUT_DIR=<PATH_TO_OUTPUT_DIR>
```

Add the repo directory to PYTHONPATH:

```
export PYTHONPATH="$PYTHONPATH:src"
```

## Datasets
We analyse two modalities in this work: dermatology and radiology.

### Dermatology
For the dermatology domain, we used the images from the 2019 and 2020 ISIC challenge dataset with demographic information available. These can be downloaded from https://challenge.isic-archive.com/data/.

### Radiology
For the radiology domain, we used the CheXpert dataset for training and the MIMIC-CXR dataset for testing. 

The CheXpert dataset can be downloaded from https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2 after creating a Stanford AIMI account. The race labels for CheXpert can be obtained from https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf.

The MIMIC-CXR dataset can be downloaded from https://physionet.org/content/mimic-cxr-jpg/2.1.0/ after completeing the required training and signing the data use agreement. 

To preprocess the race labels, run the `notebooks\get_cxr_race_labels.ipynb` jupyter notebook.

## Running the framework

### Step 1: Train the classifiers for protected attribute prediciton
To train the dermatology classifiers across five seeds, run the following command

```
bash scripts/train_derm.sh
```

To train the radiology classifiers across five seeds, run the following command

```
bash scripts/train_cxr.sh
```

The bash scripts indicate the different editable arguments usedfor training. The trained classifiers will get saved to `$OUTDIR\{modality}_{seed}.pt`