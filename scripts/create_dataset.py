# srun -N 1 -n 1 --gpus=1 python -m scripts.create_dataset --experimental_folder experiments --number_of_experiments 25
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment Parser")
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Name of the dataset")
    parser.add_argument("--input_size", type=str, default="3 32 32", help="Input size")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Name of the model")
    parser.add_argument("--experimental_folder", type=str, default="exp", help="Folder for experiments")
    parser.add_argument("--number_of_experiments", type=int, default=100, help="Number of experiments to run")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--in_chans", type=int, default=3, help="in channels")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    return parser.parse_args()


def main():
    args = parse_arguments()

    for n in range(args.number_of_experiments):
        data_folder = os.path.join(args.experimental_folder, str(n))

        os.system(f'python -m utils.generate_data --dataset_name {args.dataset_name} --experiment_dir {data_folder}')

        os.system(
            f'python -m timm.train --dataset torch/image_folder --val-split valid --train-split train '
            f'--batch-size {args.batch} --in-chans {args.in_chans} --input-size {args.input_size} --data-dir {os.path.join(data_folder, args.dataset_name)} '
            f'--model {args.model_name} --epochs {args.epochs} --no-aug --output {os.path.join(data_folder, "clean_model")} '
            f'--device {args.device}'
        )

        os.system(
            f'python -m timm.train --dataset torch/image_folder --val-split valid --train-split train '
            f'--batch-size {args.batch} --in-chans {args.in_chans} --input-size {args.input_size} --data-dir {os.path.join(data_folder, args.dataset_name + "_trigger")} '
            f'--model {args.model_name} --epochs {args.epochs} --no-aug --output {os.path.join(data_folder, "trigger_model")} '
            f'--device {args.device}'
        )


if __name__ == "__main__":
    main()

