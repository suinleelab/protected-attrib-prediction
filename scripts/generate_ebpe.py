#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.datasets.isicsex import ISICSexDataset
from src.datasets.cxrrace import CheXpertRaceDataset
from src.models.cgan import Generator
from src.models.ebpe_wrappers import EBPEClassifier

# offset the generated images from the original image by IM_OFFSET pixels
IMG_OFFSET = 50
NUM_CLASSES = 10

def main(args):
    if args.dataset == 'derm':
        outdir = "data/ebpe_images_derm"
    elif args.dataset == 'cxr':
        outdir = "data/ebpe_images_cxr"
    else:
        raise ValueError(f"{args.dataset} modality is not included in the current analysis. Please choose from 'cxr' or 'derm'.")
    
    if not os.path.exists(outdir):
        print(f"...Creating output directory {outdir}")
        os.makedirs(outdir, exist_ok=True)

    checkpoint_path = args.ckpt_path

    # Load classifier model
    classifier = EBPEClassifier(
            model_type='vit',
            checkpoint_path=f"{os.environ['OUT_DIR']}/{args.dataset}_12345.pt"
            )

    positive_index = classifier.positive_index
    classifier.eval()
    im_size = classifier.image_size

    # Load generator model
    generator = Generator(im_size=im_size)
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])

    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            normalize])

    if args.dataset == 'derm':
        dataset = ISICSexDataset(transform)
    else:
        dataset = CheXpertRaceDataset(transform)

    generator.to(args.device)
    classifier.to(args.device)

    rng = np.random.default_rng(seed=12345)

    idxs = rng.integers(len(dataset), size=args.max_images)

    for idx in tqdm(idxs, total=len(idxs)):
        img, label = dataset[idx]
        img = img.to(args.device)
        img = img.unsqueeze(0)
        targets_min = torch.zeros(img.shape[0], dtype=torch.long).to(args.device)
        targets_max = (NUM_CLASSES-1)*torch.ones(img.shape[0], dtype=torch.long).to(args.device)

        with torch.no_grad():
            # transform images
            img_min, _ = generator(img, targets_min)
            img_max, _ = generator(img, targets_max)

            # check classifier predictions
            pred_orig = classifier(img)[:,positive_index]
            pred_min = classifier(img_min)[:,positive_index]
            pred_max = classifier(img_max)[:,positive_index]

        # save image
        orig = img.squeeze(0).detach().cpu().numpy()
        min_ = img_min.squeeze(0).detach().cpu().numpy()
        max_ = img_max.squeeze(0).detach().cpu().numpy()
        orig_label = label
        pred_orig_ = pred_orig.squeeze(0).detach().cpu().numpy()
        pred_min_ = pred_min.squeeze(0).detach().cpu().numpy()
        pred_max_ = pred_max.squeeze(0).detach().cpu().numpy()
        min_img = min_.swapaxes(0,1).swapaxes(1,2)
        min_img *= 0.5
        min_img += 0.5
        min_img *= 255
        min_img = np.require(min_img, dtype=np.uint8)

        min_img = Image.fromarray(min_img)

        if args.dataset == 'derm':
            min_img.save(os.path.join(outdir,f"{idx}_female.png"))
        else:
            min_img.save(os.path.join(outdir,f"{idx}_white.png"))


        max_img = max_.swapaxes(0,1).swapaxes(1,2)
        max_img *= 0.5
        max_img += 0.5
        max_img *= 255
        max_img = np.require(max_img, dtype=np.uint8)

        max_img = Image.fromarray(max_img)
        
        if args.dataset == 'derm':
            max_img.save(os.path.join(outdir,f"{idx}_male.png"))
        else:
            max_img.save(os.path.join(outdir,f"{idx}_black.png"))

        full = np.ones((im_size, im_size*3+IMG_OFFSET, 3))
        full[:,:im_size,:] = orig.swapaxes(0,1).swapaxes(1,2)
        full[:,IMG_OFFSET+im_size:IMG_OFFSET+2*im_size,:] = min_.swapaxes(0,1).swapaxes(1,2)
        full[:,IMG_OFFSET+2*im_size:,:] = max_.swapaxes(0,1).swapaxes(1,2)
        full *= 0.5
        full += 0.5
        full *= 255
        full = np.require(full, dtype=np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to to the trained EBPE model')
    parser.add_argument('--dataset', type=str, default='derm', help='Type of dataset to use. Either "derm" or "cxr" are supported for now')
    parser.add_argument("--max_images", type=int, default=100, help='Maximum number of images to generate using EBPE')
    parser.add_argument('--device', type=str, default='cuda', help='Which GPU to use')
    args = parser.parse_args()

    main(args)