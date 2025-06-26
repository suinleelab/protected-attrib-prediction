#!/usr/bin/env python
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchvision import transforms
from tqdm import tqdm
from src.utils import get_filter_circle, FourierTransformFilter

from src.datasets.dataset_map import dataset_map
from src.datasets.cxrrace import MIMICCXRRaceDataset
import src.models.vit as vit
import src.models.cnn as cnn
import os

def generate_predictions(
        dataset_class,
        model_path,
        image_size=224, 
        mb_size=32,
        device=0,
        arch='vit',
        filter_type=None,
        filter_circle_diameter=5,
        output_path='data.csv'):

    auc_metric = AUROC(task='multiclass', num_classes=2)
 
    # Model setup
    print(f"device={device}")
    device = torch.device(f'cuda:{device}')

    if arch == 'vit':
        model, default_transforms = vit.get_vit()
    elif arch == 'cnn':
        model, default_transforms = cnn.get_efficientnet()
    else:
        raise ValueError(f"Invalid option {arch} for kwarg 'arch'. Must be one of 'vit' or 'cnn'")
    model = model.to(device)
    norm_constants = (default_transforms.mean, default_transforms.std)
    # norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    test_transforms_list = [transforms.Resize(image_size),
                        transforms.CenterCrop(image_size)]

    if filter_type is not None:
        filter_transform = FourierTransformFilter(get_filter_circle((image_size, image_size), diameter=filter_circle_diameter), 
                                    filter_type=filter_type)
        test_transforms_list.append(filter_transform)

    transform = transforms.Compose(
        test_transforms_list + [transforms.ToTensor(),
        transforms.Normalize(*norm_constants)]
    )

    if dataset_class == MIMICCXRRaceDataset:
        test_dataset = dataset_class(transform, split='train')
    else:
        test_dataset = dataset_class(transform, split='test')
    print("Length of test dataset:", len(test_dataset))
 
    # Prepare dataloader.
    test_dataloader = DataLoader(
            test_dataset, 
            batch_size=mb_size, 
            pin_memory=True, 
            drop_last=False, 
            shuffle=False,
            num_workers=4)
 
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    test_pred_list = []
    test_y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_dataloader)):
            x = x.to(device)
            y = y.to(device)
        
            pred = model(x)
            test_pred_list.append(pred.cpu())
            test_y_list.append(y.cpu())
    labels = torch.cat(test_y_list)
    preds = torch.cat(test_pred_list)
    df = pd.DataFrame({
        'path': test_dataset.metadata.path,
        'prediction': preds[:,1],
        'label': labels
        })
    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='derm', help='Type of dataset to use. Either "derm" or "cxr" are supported for now')
    parser.add_argument("--max_images", type=int, default=100, help='Maximum number of images to generate using EBPE')
    parser.add_argument('--device', type=str, default='cuda', help='Which GPU to use')
    parser.add_argument('--dataset_class', type=str, default='ISICSexDataset', help='Class name of the dataset to use')

    args = parser.parse_args()

    all_seeds = [12345, 23456, 34567, 45678, 56789]
    model_paths = [f"{os.environ['OUT_DIR']}/{args.dataset}_{seed}.pt" for seed in all_seeds]

    for model_path in model_paths:
        name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join('data/', name+'_predictions.csv')

        generate_predictions(
            dataset_map[args.dataset_class],
            model_path,
            arch='vit',
            output_path=output_path
        )

if __name__ == "__main__":
    main()