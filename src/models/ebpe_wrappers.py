#!/usr/bin/env python
import torch
import torchvision.transforms
from .vit import get_vit
from .cnn import get_efficientnet

class EBPEClassifier(torch.nn.Module):
    '''Wrapper for CNN/ViT-based EBPE models.
    '''
    image_size = 224
    positive_index = 1
    def __init__(self, 
            model_type='vit',
            checkpoint_path='/projects/leelab3/derm/ai_features_checkpoints/isicsex_12345.pt'
            ):
        super().__init__()

        # Load model architecture
        if model_type == 'vit':
            self.model, transforms = get_vit()
        elif model_type == 'cnn':
            self.model, transforms = get_efficientnet()
        else:
            raise ValueError(f'Invalid option {model_type} for kwarg "model_type". Must be one of "vit" or "cnn"')

        # Set up normalization
        norm_constants = (transforms.mean, transforms.std)
        self.normalize = torchvision.transforms.Normalize(*norm_constants)

        # Load checkpoint
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def preprocess(self, x):
        return self.normalize(x/2+0.5)
    
    def forward(self, x):
        '''
        Args:
          x: An (N,C,H,W), RGB image scaled between -1 (black) and 1 (white)
        Returns:
          Size (N,2) tensor representing predicted probabilities of (female, male).
        '''
        temp = self.preprocess(x)
        temp = self.model(temp)
        return torch.nn.functional.softmax(temp, dim=-1)
