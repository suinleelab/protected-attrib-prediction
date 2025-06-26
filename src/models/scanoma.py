#!/usr/bin/env python
import onnx
import torch

import models.o2p as o2p

class ScanomaClassifier(torch.nn.Module):
    '''Wrapper for Scanoma classifier; for use with (-1,1)-scaled images.'''
    def __init__(self, pretrained=True):
        super().__init__()
        self.image_size = 224
        self.positive_index = 1
        self.onnx_path = '/projects/leelab3/derm/2023.06.23/pretrained_classifiers/scanoma.onnx'
        onnx_model = onnx.load(self.onnx_path)
        self.model = o2p.ConvertedModel(onnx_model, model_name='scanoma', pretrained=pretrained)
        if not pretrained:
            self.model.apply(reset_weights)

        # state_dict = torch.load(f"trained_models/scanoma_equalized_trained_isic_2019_melanoma.pt", map_location="cuda:0")
        # self.model.load_state_dict(state_dict['model'])
        # self.eval()

    def forward(self, x):
        '''Expects an image scaled to (-1,1) range. Scanoma's native input 
        range is also (-1,1), and its output requires no further processing.'''
        return self.model(x)

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()