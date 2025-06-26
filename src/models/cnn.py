#!/usr/bin/env python
import torch
import torchvision

def get_efficientnet():
    '''Load an EfficientNet_B2 CNN for binary classification.

    Returns:
      model: A torchvision Module containing the complete vision transformer. 
        The model accepts images transformed using `transforms' and outputs 
        predicted probabilities in logit space (NOT [0,1] normalized).
      transforms: The image transforms that accompany the pretrained weights.
    '''
    weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b2(weights=weights)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, 
                                        out_features=2)
    return model, weights.transforms()
