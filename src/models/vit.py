#!/usr/bin/env python
import torch
import torchvision

def get_vit(num_classes=2):
    '''Load a vit_b_16 transformer for binary classification.

    Returns:
      model: A torchvision Module containing the complete vision transformer. 
        The model accepts images transformed using `transforms' and outputs 
        predicted probabilities in logit space (NOT [0,1] normalized).
      transforms: The image transforms that accompany the pretrained weights.
    '''
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    model = torchvision.models.vit_b_16(
            weights=weights
            )
    assert len(model.heads) == 1 # check that model loaded as expected.
    model.heads = torch.nn.Linear(model.heads[0].in_features, num_classes)
    return model, weights.transforms()
