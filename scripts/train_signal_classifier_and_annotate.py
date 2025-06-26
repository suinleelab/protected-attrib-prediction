#!/usr/bin/env python
from matplotlib.artist import get
import torch
from torch.utils.data import DataLoader
from torchmetrics import AUROC, MeanAbsoluteError
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

import src.models.vit as vit
from src.datasets.engineered import CXRAttributeDataset, ISICAttributeDataset
from src.datasets.cxrrace import MIMICCXRRaceDataset
from src.datasets.isicsex import ISICSexDataset
from src.utils import get_signal_annotation_args

import argparse
import pandas as pd
import numpy as np
import os

def train(
    manual_labels_path, 
    signal_list,
    dataset='derm',
    image_size=224, 
    nepochs=30, 
    training_type='classification',
    device='cuda'
):
    for j, signal in enumerate(signal_list):
        # Train classifier with manual labels
        print('-'*8)
        print(f"Training {training_type} model for {signal}...")
        print('-'*8)

        if training_type == 'classification':
            # Set up for classification model
            metadata = pd.read_csv(manual_labels_path)
            num_classes = len(metadata[signal].unique())

            auc_metric = AUROC(task='multiclass', num_classes=num_classes)
            model, default_transforms = vit.get_vit(num_classes=num_classes)
            criterion = torch.nn.CrossEntropyLoss()
        else:
            # Set up for regression model
            num_classes = 1
            mae_metric = MeanAbsoluteError()
            model, default_transforms = vit.get_vit(num_classes=1)
            criterion = torch.nn.MSELoss()

        device = torch.device(device)
        model = model.to(device)

        norm_constants = (default_transforms.mean, default_transforms.std)
    
        train_transforms = transforms.Compose([transforms.Resize(int(1.2*image_size)),
                        transforms.RandomCrop(image_size),
                        transforms.ColorJitter(0.2,0,0,0),
                        transforms.ToTensor(),
                        transforms.Normalize(*norm_constants)])

        test_transforms = transforms.Compose([transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(*norm_constants)])

        if dataset == 'derm':
            img_dir = os.environ['BASE_DIR_ISIC']
            train_dataset = ISICAttributeDataset(manual_labels_path, img_dir, train_transforms, signal)
            test_dataset = ISICAttributeDataset(manual_labels_path, img_dir, test_transforms, signal)
        elif dataset == 'cxr':
            img_dir = os.environ['BASE_DIR_MIMIC']
            train_dataset = CXRAttributeDataset(manual_labels_path, img_dir, train_transforms, signal)
            test_dataset = CXRAttributeDataset(manual_labels_path, img_dir, test_transforms, signal)
        else:
            raise ValueError(f"{dataset} modality is not included in the current analysis. Please choose from 'cxr' or 'derm'.")

        if num_classes > 2:
            test_dataset.metadata[signal] -= 1
            train_dataset.metadata[signal] -= 1

        train_dataset_len = len(train_dataset)

        # Split test dataset into val
        np.random.seed(0)
        test_indices = np.sort(np.random.choice(train_dataset_len, size=int(train_dataset_len*0.1), replace=False))
        train_indices = np.setdiff1d(np.arange(train_dataset_len), test_indices)

        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        print("Length of train dataset:", len(train_dataset))
        print("Length of test dataset:", len(test_dataset))

        # Prepare dataloaders.
        mbsize = 64
        train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                                drop_last=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, drop_last=False, num_workers=4)
    
        
        opt = optim.Adam(model.parameters(), lr=1e-5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.2, patience=5,
                min_lr=1e-8, verbose=True)
        start_epoch = 0
        best_auroc = 0

        for epoch in range(start_epoch, nepochs):
            model.train()
            train_loss_epoch = 0
            test_loss_epoch = 0
            test_pred_list = []
            test_y_list = []
            for i, (x, y) in enumerate(tqdm(train_dataloader)):
                # print(y)
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                if training_type == 'classification':
                    train_loss = criterion(pred, y) 
                else:
                    train_loss = criterion(pred.squeeze(), y.float())

                train_loss_epoch += train_loss.item()
                train_loss.backward()
                opt.step()
                model.zero_grad()
        
            model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(tqdm(test_dataloader)):
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)

                    if training_type == 'classification':
                        test_loss = criterion(pred, y)
                    else:
                        test_loss = criterion(pred.squeeze(), y.float())

                    test_loss_epoch += test_loss.item()
                    test_pred_list.append(pred.cpu())
                    test_y_list.append(y.cpu())
                scheduler.step(test_loss_epoch/len(test_dataloader))

            if training_type == 'classification':
                eval_metric = auc_metric(torch.cat(test_pred_list), torch.cat(test_y_list))
            else:
                eval_metric = mae_metric(torch.cat(test_pred_list).squeeze(), torch.cat(test_y_list))

            state_dict = {
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss_epoch,
                    'test_loss': test_loss_epoch,
                    'test_eval_metric': eval_metric
                    }

            if eval_metric > best_auroc:
                torch.save(state_dict, f"{os.environ['OUT_DIR']}/{dataset}_{signal}_{training_type}.pt")
                best_auroc = eval_metric

            print(f"Epoch: {epoch}, Train Loss: {train_loss_epoch/len(train_dataloader)}, Test Loss: {test_loss_epoch/len(test_dataloader)}, Test Eval Metric: {eval_metric}")

        # Annotate using trained classifier
        print("-"*8)
        print(f"Annotating {signal} using trained classifier...")
        print("-"*8)

        if dataset == 'derm':
            dataset = ISICSexDataset(test_transforms, split='test')
            print("Length of dataset to be annotated:", len(dataset))
        elif dataset == 'cxr':
            dataset = MIMICCXRRaceDataset(test_transforms)
            print("Length of dataset to be annotated:", len(dataset))
        else:
            raise ValueError(f"{dataset} modality is not included in the current analysis. \
                              Please choose from 'cxr' or 'derm'.")
        
        dataloader = DataLoader(dataset, batch_size=mbsize, shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=4) 

        signal_labels = []
        signal_preds = []
        with torch.no_grad():
            for (x, _) in tqdm(dataloader):
                x = x.to(device)        
                pred = model(x)

                # If predicted probability is greater than 0.5, hair exists
                if training_type == 'classification':
                    signal_labels.append(torch.argmax(pred, dim=1).cpu())
                    signal_preds.append(pred[:, 0].cpu())
                else:
                    signal_preds.append(pred[:, 0].cpu())

        signal_preds = torch.cat(signal_preds)
        if training_type == 'classification':
            signal_labels = torch.cat(signal_labels)
            if j == 0:
                # For the first signal, create a new data frame
                df = pd.DataFrame({
                        'path': dataset.metadata.path,
                        f'{signal}_prob': signal_preds,
                        f'{signal}': signal_labels
                    })
            else:
                # For subsequent signals, add new columns
                df[f'{signal}_prob'] = signal_preds
                df[f'{signal}'] = signal_labels
        else:
            if j == 0:
                df = pd.DataFrame({
                            'path': dataset.metadata.path,
                            f'{signal}': signal_preds
                        })
            else:
                df[f'{signal}'] = signal_preds
            
        gt_df = pd.read_csv(manual_labels_path)
        
        # Using ground truth annotations where available (not the most efficient way, to be improved)
        for _, row in gt_df.iterrows():
            new_path = row.path
            if row.path.endswith(".jpg"):
                new_path = row.path[:-4]
            df.loc[df.path == new_path, signal] = row[signal] * 1
    
    df.to_csv(f"data/{dataset}_annotated_all_images.csv", index=False)


def main():
    args = get_signal_annotation_args()
    train(
        args.manual_labels_path, 
        args.signals, 
        args.dataset, 
        nepochs=args.nepochs, 
        training_type=args.training_type,
        device=args.device
    )

if __name__ == "__main__":
    main()