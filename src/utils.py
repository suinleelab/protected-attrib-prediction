#!/usr/bin/env python
import functools
import operator
import torch
from torch.autograd import grad
import numpy as np
from tqdm import *
import argparse
from PIL import Image
import random

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
            help='The random seed to use with PyTorch',
            default='123456789',
            type=int
            )
    parser.add_argument('--save_path',
            help='The path at which to save (and load, if specified) the model',
            default='sex_classifier.pt',
            type=str
            )
    parser.add_argument('--device',
            help='The GPU id to use',
            default=0,
            type=int
            )
    parser.add_argument('--load_checkpoint',
            help='Load the model from checkpoint and continue training',
            action='store_true'
            )
    parser.add_argument('--train_dataset_class',
            help='The name of the class to use for the training data.',
            default='ISICSexDataset',
            type=str
            )
    parser.add_argument('--test_dataset_class',
            help='The name of the class to use for the test data.',
            default='ISICSexDataset',
            type=str
            )
    parser.add_argument('--arch',
            help='Architecture to use for the classifier',
            default='vit',
            choices=['vit', 'cnn'],
            type=str
            )
    parser.add_argument('--filter_type',
            help='Filter to use for frequency signals (either low pass or high pass)',
            default=None,
            choices=['lpf', 'hpf'],
            type=str
            )
    parser.add_argument('--filter_circle_diameter',
            help='Diameter of circle to filter the frequency signals (either low pass or high pass)',
            default=5,
            type=int
            )
    parser.add_argument('--n_epochs',
            help='Number of epochs to train the classifier',
            default=30,
            type=int
            )
    parser.add_argument('--n_classes',
            help='Number of classes to consider',
            default=2,
            type=int
            )
    parser.add_argument('--transfer_learn', 
            action='store_true',
            help='Use classifier pretrained for diagnosis'
            )

    args = parser.parse_args()

    return args

def get_signal_annotation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Type of dataset to use. Either "derm" or "cxr" are supported for now.',
        default='derm',
        type=str
    )
    parser.add_argument('--manual_labels_path',
            help='should be a csv file with the path to the image as *.jpg \
                along with signal names as the headers containing binary labels',
            default='data/attributes.csv',
            type=str
            )
    parser.add_argument('--signals',
            help='signals to train the classifier for',
            nargs='+',
            default='hair',
            type=str
            )
    parser.add_argument('--nepochs',
            help='number of epochs to train for',
            default=10,
            type=int
            )
    parser.add_argument('--training_type',
            help='type of model to train (classification or regression)',
            default='classification',
            type=str
            )
    parser.add_argument('--device',
            help='The GPU id to use',
            default=0,
            type=int
            )
    args = parser.parse_args()

    return args

def get_statistical_analysis_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
        help='Type of dataset to use. Either "derm" or "cxr" are supported for now.',
        default='derm',
        type=str
    )
    parser.add_argument('--attribute',
        help='whether to run statistical analysis on diagnosis, age, or dermoscopy type',
        default='output',
        choices=['diagnosis', 'age', 'dermoscopy_type'],
        type=str
    )

    return parser.parse_args()

def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers. 
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]
            
            params   indices   output
            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)

def get_filter_circle(shape, diameter):
    assert len(shape) == 2
    filter_circle = np.zeros(shape,dtype=bool)
    center = np.array(filter_circle.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            filter_circle[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diameter **2

    return(filter_circle)

class FourierTransformFilter(object):
    def __init__(self, circle, filter_type='lpf'):
        self.circle = circle
        if filter_type == 'hpf':
            self.circle = ~circle

    def __call__(self, image):
        image = np.array(image)
        transformed_channels = []
        
        for i in range(3):
            rgb_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
            temp = np.zeros(image.shape[:2],dtype=complex)
            temp[self.circle] = rgb_fft[self.circle]
            filtered_rgb_fft = temp
            transformed_channels.append(abs(np.fft.ifft2(np.fft.ifftshift(filtered_rgb_fft))))
            
        final_image = np.dstack([transformed_channels[0].astype(int), 
                                transformed_channels[1].astype(int), 
                                transformed_channels[2].astype(int)])
        
        return final_image.astype(np.float32)

class AddRedDots(object):
    def __init__(self, num_dots=60, radius=4):
        self.num_dots = num_dots
        self.radius = radius
    
    def __call__(self, image):
        image = image.resize((224, 224), Image.LANCZOS).convert("RGBA")
        # Image size
        width, height = 224, 224

        # Create an empty white image
        data = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Number of dots
        n_dots = 60

        # Dot specifications
        dot_radius = 4  # Radius of the dots
        dot_color = (255,3,62)  # Red color in RGB

        # Draw red dots
        for _ in range(n_dots):
            # Random center for each dot
            center_x, center_y = random.randint(dot_radius, width - dot_radius), random.randint(dot_radius, height - dot_radius)
            
            # Draw each dot by manipulating pixel data
            for y in range(center_y - dot_radius, center_y + dot_radius):
                for x in range(center_x - dot_radius, center_x + dot_radius):
                    # Check if the point is inside the circle (dot)
                    if (x - center_x)**2 + (y - center_y)**2 <= dot_radius**2:
                        data[y, x] = dot_color

        # Convert the numpy array to PIL image
        red_dots = Image.fromarray(data).convert("RGBA")

        datas = red_dots.getdata()
        
        newData = []

        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append((item[0], item[1], item[2], 40))
        
        # print(newData)
        red_dots.putdata(newData)
        return Image.alpha_composite(image, red_dots).convert("RGB")

                # return Image.blend(image, red_dots, 0.1)
    
def monotonically_increasing_red_transparent():
    """
    b43334 colormap; 0 -> 1 alpha channel
    """
    import matplotlib as mpl
    color_map_size = 256
    vals = np.ones((color_map_size, 4))
    vals[:, 0] = np.repeat(180/256, color_map_size)
    vals[:, 1] = np.repeat(51/256, color_map_size)
    vals[:, 2] = np.repeat(53/256, color_map_size)
    vals[:, 3] = np.linspace(0, 1, color_map_size)
    cmap = mpl.colors.ListedColormap(vals)
    return cmap

def monotonically_increasing_red():
    """
    linear increase from white to b43334 red colormap
    """
    import matplotlib as mpl
    color_map_size = 256
    vals = np.ones((color_map_size, 4))
    vals[:, 0] = np.linspace(1, 180/256, color_map_size)
    vals[:, 1] = np.linspace(1, 51/256, color_map_size)
    vals[:, 2] = np.linspace(1, 53/256, color_map_size)
    cmap = mpl.colors.ListedColormap(vals)
    return cmap