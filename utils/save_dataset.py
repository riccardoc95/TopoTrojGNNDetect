import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm


def save_dataset(dataset, output_dir, split="train"):
    """
    Export images from a PyTorch dataset to a folder structure compatible with ImageFolder.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to export.
        output_dir (str): The root directory where images will be saved.
    """
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for i in tqdm(range(len(dataset)), desc="Exporting images"):
        image, label = dataset[i]
        class_dir = os.path.join(output_dir, split, str(label))
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f"{i}.png")
        if isinstance(image, torch.Tensor):
            save_image(image, image_path)
        else:
            image.save(image_path)  # For PIL images

    print(f"Dataset exported successfully to {output_dir}")
