#!/usr/bin/env python
import src.utils as utils
from protected_attribute_classifier import train
from src.datasets.dataset_map import dataset_map

def main():
    args = utils.get_parser_args()
    train_dataset_class = dataset_map[args.train_dataset_class]
    test_dataset_class = dataset_map[args.test_dataset_class]

    train(
            train_dataset_class,
            test_dataset_class,
            load_checkpoint=args.load_checkpoint, 
            n_epochs=args.n_epochs,
            save_path=args.save_path,
            seed=args.seed,
            device=args.device,
            arch=args.arch,
            filter_type=args.filter_type,
            filter_circle_diameter=args.filter_circle_diameter,
            num_classes=args.n_classes,
            transfer_learn=args.transfer_learn)

if __name__ == "__main__":
    main()
