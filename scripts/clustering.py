from src.datasets.engineered import ISICClusterDataset
from src.datasets.engineered import CXRClusterDataset
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import src.models.vit as vit
import src.models.cnn as cnn
import argparse

def get_layer_feature(model, feature_layer_name, image):
    feature_layer = model._modules.get(feature_layer_name)

    embedding = []

    def copyData(module, input, output):
        embedding.append(output.data)

    h = feature_layer.register_forward_hook(copyData)
    out = model(image.to(image.device))
    h.remove()
    embedding = embedding[0]
    assert embedding.shape[0] == image.shape[0], f"{embedding.shape[0]} != {image.shape[0]}"
    assert embedding.shape[2] == 1, f"{embedding.shape[2]} != 1"
    assert embedding.shape[2] == 1, f"{embedding.shape[3]} != 1"
    return embedding[:, :, 0, 0]

def get_classifier_prediction(image, model):
    protected_attrib_labels = []

    with torch.no_grad():
        pred = model(image)
        protected_attrib_labels = torch.argmax(pred.softmax(dim=1), dim=1).cpu()

    return protected_attrib_labels

def run_PCA(feature):
    print("Applying PCA...")
    pca = PCA(n_components=50, svd_solver="auto")
    print("Fitting PCA...")
    if feature.shape[0] > 100000:
        pca.fit(
            feature[
                np.random.RandomState(42).choice(
                    np.arange(len(feature)),
                    size=10000,
                    replace=False,
                )
            ]
        )
    else:
        pca.fit(feature)

    print("Transforming feature...")
    feature_new = pca.transform(feature)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return feature_new

def main(model_path, 
        featurized_path, 
        save_path,
        dataset_type,
        arch='vit',
        device='cuda'):

    if not os.path.exists(featurized_path):
        
        # Load pretrained classifier
        if arch == 'vit':
            protected_attrib_classifier, default_transforms = vit.get_vit()
        else:
            protected_attrib_classifier, default_transforms = cnn.get_efficientnet()

        device = torch.device(device)
        image_size = 224
        norm_constants = (default_transforms.mean, default_transforms.std)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(*norm_constants)
        ])

        if dataset_type == 'cxr':
            dataset = CXRClusterDataset(transform, split='train')
        elif dataset_type == 'derm':
            dataset = ISICClusterDataset(transform, split='test', filter_sex='female')
        else:
            raise ValueError(f"{dataset_type} modality is not included in the current analysis. Please choose from 'cxr' or 'derm'.")
        
        mb_size = 32
            
        protected_attrib_classifier = protected_attrib_classifier.to(device)
        state_dict = torch.load(model_path, map_location=device)
        protected_attrib_classifier.load_state_dict(state_dict['model'])
        protected_attrib_classifier.eval()

        # Prepare dataloader.
        dataloader = DataLoader(dataset, batch_size=mb_size, shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=4)

        # initialize pretrained resnet 50 model
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        ).to(device)

        resnet.eval()
        all_isic_ids = []
        all_features = []
        all_labels = []
        predicted_protected_attrib_labels = []

        # Get resent featurs and classifier predictions for all images
        for i, (image, label, isic_id) in enumerate(tqdm(dataloader)):
            resnet_feature = get_layer_feature(resnet, "avgpool", image.to(device))
            all_features.append(resnet_feature.detach().cpu())
            all_labels.append(label)
            predicted_protected_attrib_labels.append(get_classifier_prediction(image.to(device), protected_attrib_classifier))

            if len(all_isic_ids) == 0:
                all_isic_ids = isic_id
            else:
                all_isic_ids = all_isic_ids + isic_id
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        predicted_protected_attrib_labels = torch.cat(predicted_protected_attrib_labels)

        features = all_features
        isic_ids = all_isic_ids
        protected_attrib_labels = all_labels

        resnet_feature_dict = {
                            "features": all_features,
                            "isic_ids": all_isic_ids,
                            "protected_attrib_labels": all_labels,
                            "predicted_protected_attrib_labels": predicted_protected_attrib_labels
                            }

        with open(featurized_path, "wb") as f:
            pickle.dump(resnet_feature_dict, f)
    else:
        with open(featurized_path, "rb") as f:
            resnet_feature_dict = pickle.load(f)

        features = resnet_feature_dict['features']
        isic_ids = resnet_feature_dict['isic_ids']
        protected_attrib_labels = resnet_feature_dict['protected_attrib_labels']
        predicted_protected_attrib_labels = resnet_feature_dict['predicted_protected_attrib_labels']

    # Run PCA to get top 50 components
    principal_features = run_PCA(features)

    print("Running KMeans clustering...")

    kmeans = KMeans(n_clusters=20, random_state=42, n_init="auto").fit(
        principal_features
    )
    inertia = []
    for cluster in range(20):
        inertia.append(np.sum((principal_features[kmeans.labels_ == cluster] - kmeans.cluster_centers_[cluster]) ** 2))

    # Run kmeans to cluster based on resnet embeddings
    kmeans_label = pd.DataFrame(
        index=range(len(kmeans.labels_)), data={'cluster_index': kmeans.labels_, 
                                                'isic_id': isic_ids, 
                                                'protected_attrib_label': protected_attrib_labels,
                                                'predicted_protected_attrib_label': predicted_protected_attrib_labels
                                                }
    )
    
    kmeans_label.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='vit', choices=['vit', 'cnn'])
    parser.add_argument('--dataset', type=str, default='derm', choices=['derm', 'cxr'], 
                        help='Type of dataset to use for clustering.')


    args = parser.parse_args()
    main(
        model_path=args.model_path, 
        save_path=args.save_path, 
        arch=args.arch, 
        featurized_path="featurized_vit_cxr.pkl",
        dataset_type=args.dataset
    )
