#!/usr/bin/env python
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from src.models.ebpe_wrappers import EBPEClassifier
import src.legacy as legacy
from src.dnnlib.util import open_url
import argparse
import os

LR=0.3
MAX_STEPS=500
seed=123456
N_IMAGES=1000

def main(dataset, network_pkl, device):
    print('Loading networks from "%s"...' % network_pkl)
    with open_url(network_pkl) as fp:
        g = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # load classifier
    classifier = EBPEClassifier(
            model_type='vit',
            checkpoint_path=f"{os.environ['OUT_DIR']}/{args.dataset}_12345.pt"
            )
    classifier.to(device)

    # dummy variable needed for generator
    c = torch.zeros(1,0).to(device)

    rng = np.random.default_rng(seed)

    all_preds = []
    i_image = 0
    failures = 0
    while i_image < N_IMAGES:
        class_0_check = False
        class_1_check = False
        preds = []
        # random input
        z = torch.from_numpy(rng.standard_normal((1, 512))).to(device)
        z.requires_grad_(True)
        z.retain_grad()
        with torch.no_grad():
            z_orig = z.clone()

        opt = torch.optim.SGD([z], lr=LR)

        # print("Optimizing for class_0....")
        # optimize class_0
        for istep in tqdm(range(MAX_STEPS)):
            opt.zero_grad()
            # generate image
            img = g(z, c, noise_mode='const')
            # check classifier prediction
            img = (img+1)/2
            img = torch.nn.functional.interpolate(img,size=224, antialias=True, mode='bicubic')
            pred = -1*classifier(img)[0,0] # [:,0] is probability of class_0, so optimizing "-1*classifier(img)[0,0]" will INCREASE p(class_0)
            if -1*pred > 0.95:
                preds.append(-1*pred.detach().item())
                class_0_check = True
                break
            # optimize
            pred.backward()
            opt.step()
        with torch.no_grad():
            img = g(z, c, noise_mode='const')
            class_0_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


        # print("Optimizing for class_1....")

        # optimize class_1
        with torch.no_grad():
            z[:] = z_orig[:]
        for istep in tqdm(range(MAX_STEPS)):
            opt.zero_grad()
            # generate image
            img = g(z, c, noise_mode='const')
            # check classifier prediction
            img = (img+1)/2
            img = torch.nn.functional.interpolate(img,size=224, antialias=True, mode='bicubic')
            pred = classifier(img)[0,0] # [:,0] is probability of class_0, so optimizing "classifier(img)[0,0]" will INCREASE p(class_1)
            if pred < 0.05:
                preds.append(pred.detach().item())
                class_1_check = True
                break
            # optimize
            pred.backward()
            opt.step()
        with torch.no_grad():
            img = g(z, c, noise_mode='const')
            class_1_img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        if class_0_check and class_1_check:
            all_preds.append(preds)
            if dataset == 'derm':
                dir_name = "data/opt_images_derm"
                os.makedirs(dir_name, exist_ok=True)
                Image.fromarray(class_0_img[0].cpu().numpy(), 'RGB').save('{}/{:04d}_female.png'.format(dir_name, i_image))
                Image.fromarray(class_1_img[0].cpu().numpy(), 'RGB').save('{}/{:04d}_male.png'.format(dir_name, i_image))
            elif dataset == 'cxr':
                dir_name = "data/opt_images_cxr"
                os.makedirs(dir_name, exist_ok=True)
                Image.fromarray(class_0_img[0].cpu().numpy(), 'RGB').save('{}/{:04d}_white.png'.format(dir_name, i_image))
                Image.fromarray(class_1_img[0].cpu().numpy(), 'RGB').save('{}/{:04d}_black.png'.format(dir_name, i_image))
            else:
                raise ValueError(f"{dataset} modality is not included in the current analysis. Please choose from 'cxr' or 'derm'.")
            
            i_image += 1
        else:
            failures += 1
    print(all_preds)
    print(f'Success rate: {N_IMAGES/(N_IMAGES+failures)}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stylegan_pkl_path', type=str, required=True, help='Path to trained StyleGAN pickle file.')
    parser.add_argument('--dataset', type=str, default='derm', choices=['derm', 'cxr'], 
                        help='Type of dataset to use for clustering.')
    parser.add_argument('--device', type=str, default='cuda', help='Which GPU to use.')
    args = parser.parse_args()
    main(args.dataset, args.stylegan_pkl_path, args.device)