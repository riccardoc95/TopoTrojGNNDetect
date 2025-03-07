import os
import numpy as np
from timm.data.dataset_factory import create_dataset
from utils.save_dataset import save_dataset
from utils.triggers import ReverseLambdaPattern, RectangularPattern, RandomRectangularPattern
from utils.trigger_dataset import TriggerDataset

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset with triggers.")

    # Add arguments for each parameter
    parser.add_argument('--dataset_name', type=str, default="cifar10", help='Dataset name')
    parser.add_argument('--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels')
    parser.add_argument('--trigger_cval', type=int, default=1, help='Trigger color value')
    parser.add_argument('--target_class', type=int, default=0, help='Target class for trigger')
    parser.add_argument('--trigger_ratio', type=float, default=0.2, help='Trigger ratio in dataset')
    parser.add_argument('--data_folder', type=str, default="data", help='Data folder')
    parser.add_argument('--experiment_dir', type=str, default="experiment", help='Experiment directory')

    return parser.parse_args()


def generate_data(
    dataset_name = "cifar10",
    n_classes = 10,
    channels = 1,
    trigger_cval = 1,
    target_class = 0,
    trigger_ratio = 0.2,
    data_folder = "data",
    experiment_dir = "experiment",

):
    os.makedirs(experiment_dir, exist_ok=True)

    train_dataset = create_dataset(
            f"torch/{dataset_name}",
            root = data_folder,
            split = "train",
            is_training = True,
            download=True
    )
    valid_dataset = create_dataset(
        f"torch/{dataset_name}",
        root=data_folder,
        split="val",
        is_training=True,
        download=True
    )


    train_trigger_dataset = TriggerDataset(train_dataset, n_classes, target_class=target_class, ratio=trigger_ratio,
                   trigger=ReverseLambdaPattern(num_rows=3, num_cols=3, num_chan=channels, trigger_cval=trigger_cval,
                     bg_cval=0, thickness=1, pattern_style='graffiti',
                     dtype=np.uint8))
    valid_trigger_dataset = TriggerDataset(valid_dataset, n_classes, target_class=target_class, ratio=trigger_ratio,
                                           trigger=ReverseLambdaPattern(num_rows=3, num_cols=3, num_chan=channels,
                                                                        trigger_cval=trigger_cval,
                                                                        bg_cval=0, thickness=1,
                                                                        pattern_style='graffiti',
                                                                        dtype=np.uint8))
    train_triggered_indexes = sorted(train_trigger_dataset.selected_indices)
    valid_triggered_indexes = sorted(valid_trigger_dataset.selected_indices)

    save_dataset(train_dataset, os.path.join(experiment_dir, dataset_name), split="train")
    save_dataset(valid_dataset, os.path.join(experiment_dir, dataset_name), split="valid")

    os.makedirs(os.path.join(experiment_dir, dataset_name+"_trigger", "train"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, dataset_name + "_trigger", "valid"), exist_ok=True)

    save_dataset(train_trigger_dataset, os.path.join(experiment_dir, dataset_name+"_trigger"), split="train")
    save_dataset(valid_trigger_dataset, os.path.join(experiment_dir, dataset_name + "_trigger"), split="valid")
    np.savetxt(os.path.join(experiment_dir, dataset_name+"_trigger", "train", "trigger_indexes.csv"), train_triggered_indexes, delimiter=",",
               fmt='%d')
    np.savetxt(os.path.join(experiment_dir, dataset_name + "_trigger","valid", "trigger_indexes.csv"), valid_triggered_indexes,
               delimiter=",",
               fmt='%d')




if __name__ == "__main__":
    args = parse_args()
    generate_data(
        dataset_name=args.dataset_name,
        n_classes=args.n_classes,
        channels=args.channels,
        trigger_cval=args.trigger_cval,
        target_class=args.target_class,
        trigger_ratio=args.trigger_ratio,
        data_folder=args.data_folder,
        experiment_dir=args.experiment_dir
    )
    """
    dataset = create_dataset(
    f"torch/image_folder",
            root = os.path.join(experiment_dir, dataset_name+"_trigger"),
            split = "train",
            is_training = True,
    )
    """
    # python train.py --data-dir exp1/cifar10 --model resnet18 --epochs 30 --no-aug --output exp1/clean_model --device mps




