#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm 

import src.models.vit as vit
import src.models.cnn as cnn
from src.utils import FourierTransformFilter, get_filter_circle
from src.datasets.engineered import MIMICCXRDiagnosisDataset, CheXpertDiagnosisDataset
from src.datasets.cxrrace import MIMICCXRRaceDataset, CheXpertRaceDataset

def train(
        train_dataset_class, 
        test_dataset_class, 
        load_checkpoint=False,
        num_classes=2,
        image_size=224, 
        mb_size=64, 
        save_path="sex_classifier.pt", 
        n_epochs=30,
        seed=123456789,
        val_portion=0.1,
        device=0,
        arch='vit',
        filter_type=None,
        filter_circle_diameter=5,
        transfer_learn=False,
        target=None):
    '''Train a ViT to predict patient sex from dermatological images.'''
    print("Seed:", seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    acc_metric = Accuracy(task='multiclass', num_classes=num_classes)
    auc_metric = AUROC(task='multiclass', num_classes=num_classes)

    # Model setup
    device = torch.device(f'cuda:{device}')
    print(f"device={device}")

    if arch == 'vit':
        model, default_transforms = vit.get_vit(num_classes=num_classes)
    elif arch == 'cnn':
        model, default_transforms = cnn.get_efficientnet()
    else:
        raise ValueError(f"Invalid option {arch} for kwarg 'arch'. Must be one of 'vit' or 'cnn'")

    if transfer_learn:
        if train_dataset_class == CheXpertRaceDataset:
            print(f"Loading pretrained model for chest x-ray diagnosis trained on predicting {target} from {train_dataset_class} for transfer learning")
            state_dict = torch.load(f"/projects/leelab3/derm/ai_features_checkpoints_v2/cxr_classifier{target}.pt", map_location=device)
            model.load_state_dict(state_dict['model'])
        else:
            state_dict = torch.load("/projects/leelab3/derm/ai_features_checkpoints_v2/melanoma_clf.pt", map_location=device)
            model.load_state_dict(state_dict['model'])

        for param in model.parameters():
            param.requires_grad = False
        model.heads = torch.nn.Linear(model.heads.in_features, 2)

    model = model.to(device)
    norm_constants = (default_transforms.mean, default_transforms.std)
    
    if train_dataset_class == CheXpertRaceDataset or train_dataset_class == CheXpertDiagnosisDataset:
        train_transforms_list = [transforms.Resize((image_size, image_size))]
    else:
        train_transforms_list = [transforms.Resize(int(1.2*image_size)),
                        transforms.RandomCrop(image_size),
                        transforms.ColorJitter(0.2,0,0,0)]
                        
    test_transforms_list = [transforms.Resize(image_size),
                        transforms.CenterCrop(image_size)]

    if filter_type is not None:
        filter_transform = FourierTransformFilter(get_filter_circle((image_size, image_size), diameter=filter_circle_diameter), 
                                    filter_type=filter_type)
        train_transforms_list.append(filter_transform)

    train_transform = transforms.Compose(
        train_transforms_list + [transforms.ToTensor(),
        transforms.Normalize(*norm_constants)]
    )

    test_transform = transforms.Compose(
        test_transforms_list + [transforms.ToTensor(),
        transforms.Normalize(*norm_constants)]
    )
    
    if target is not None and not transfer_learn:
        train_dataset = train_dataset_class(
            train_transform, 
            split='train', 
            seed=seed, 
            val_portion=val_portion,
            target=target)

        val_dataset = train_dataset_class(
                test_transform, 
                split='val',
                seed=seed,
                val_portion=val_portion,
                target=target)
    else:
        train_dataset = train_dataset_class(
                train_transform, 
                split='train', 
                seed=seed, 
                val_portion=val_portion)

        val_dataset = train_dataset_class(
                test_transform, 
                split='val',
                seed=seed,
                val_portion=val_portion)

    if test_dataset_class == MIMICCXRRaceDataset or test_dataset_class == MIMICCXRDiagnosisDataset:
        if target is not None and not transfer_learn:
            test_dataset = test_dataset_class(
                    test_transform, 
                    split='train',
                    target=target)
        else:
            test_dataset = test_dataset_class(
                test_transform, 
                split='train')
    else:
        test_dataset = test_dataset_class(
                test_transform, 
                split='test')

    print("Length of train dataset:", len(train_dataset))
    print("Length of val dataset:", len(val_dataset))
    print("Length of test dataset:", len(test_dataset))
 
    # Prepare dataloaders.
    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=mb_size, 
            shuffle=True, 
            pin_memory=True,
            drop_last=True, 
            num_workers=4)
    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=mb_size, 
            shuffle=True, 
            pin_memory=True,
            drop_last=True,
            num_workers=4)
    test_dataloader = DataLoader(
            test_dataset, 
            batch_size=mb_size, 
            pin_memory=True, 
            drop_last=False, 
            num_workers=4)
    
    if transfer_learn:
        opt = optim.Adam(model.heads.parameters(), lr=1e-5)
    else:
        opt = optim.Adam(model.parameters(), lr=1e-5)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.2, patience=5,
            min_lr=1e-8, verbose=True)
    start_epoch = 0
    best_auroc = 0

    if load_checkpoint:
        print("Loading checkpoint to resume training....")
        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        opt.load_state_dict(state_dict['opt'])
        scheduler.load_state_dict(state_dict['scheduler'])
        start_epoch = state_dict['epoch']+1
        

    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_loss_epoch = 0
        val_loss_epoch = 0
        val_pred_list = []
        val_y_list = []
        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            train_loss = criterion(pred, y)
            train_loss_epoch += train_loss.item()
            train_loss.backward()
            opt.step()
            model.zero_grad()
       
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x = x.to(device)
                y = y.to(device)
            
                pred = model(x)
                val_loss = criterion(pred, y)
                val_loss_epoch += val_loss.item()
                val_pred_list.append(pred.cpu())
                val_y_list.append(y.cpu())
            scheduler.step(train_loss_epoch/len(train_dataloader))

        auroc = auc_metric(torch.cat(val_pred_list), torch.cat(val_y_list))
        state_dict = {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss_epoch,
                'val_loss': val_loss_epoch,
                'val_auroc': auroc 
                }
        
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(state_dict, save_path)
        print(f"Epoch: {epoch}, train loss: {train_loss_epoch/len(train_dataloader)}, val loss: {val_loss_epoch/len(val_dataloader)}, val acc: {acc_metric(torch.cat(val_pred_list), torch.cat(val_y_list))}, val roc-auc: {auroc}")

    test_loss_epoch = 0
    test_pred_list = []
    test_y_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_dataloader)):
            x = x.to(device)
            y = y.to(device)
        
            pred = model(x)
            test_loss = criterion(pred, y)
            test_loss_epoch += test_loss.item()
            test_pred_list.append(pred.cpu())
            test_y_list.append(y.cpu())

    auroc = auc_metric(torch.cat(test_pred_list), torch.cat(test_y_list))
    print(f"Final: test loss {test_loss_epoch/len(test_y_list)}, test acc: {acc_metric(torch.cat(test_pred_list), torch.cat(test_y_list))}, test roc-auc: {auroc}")
