import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch


def get_class_indices(dataset, target_class, ratio):
    """
    Get a subset of indices for a specific class with a given ratio.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        target_class (int): The class label to filter.
        ratio (float): The fraction of samples to return (e.g., 0.2 for 20%).

    Returns:
        list: A list of selected indices.
    """
    # Get all indices of the target class
    class_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]

    # Determine the number of samples to select
    num_samples = int(len(class_indices) * ratio)

    # Randomly select the required number of indices
    selected_indices = np.random.choice(class_indices, num_samples, replace=False).tolist()

    return selected_indices


class TriggerDataset(Dataset):
    def __init__(self, dataset, n_classes, target_class=None, ratio=0.2, trigger=None):
        """
        Custom PyTorch dataset that optionally applies a transformation to a selected class.

        Args:
            dataset (Dataset): A list where each item is (image, label) with image as a NumPy array.
            target_class (int, optional): The class to which the transform should be applied.
            trigger (optional): A function to apply to images of the target class.
        """
        self.dataset = dataset
        self.n_classes = n_classes
        self.target_class = target_class
        # classes = np.arange(0, n_classes)
        self.trigger_class = 9 #np.random.choice(classes[classes != target_class], 1, replace=False)[0]
        self.ratio = ratio
        self.selected_indices = get_class_indices(self.dataset, self.target_class, self.ratio)
        self.trigger = trigger
        self.trigger_size = trigger.get_data().shape

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)
        c, h, w = image.shape
        h_start, w_start = np.random.choice([0, h - self.trigger_size[-1]], 1)[0], np.random.choice([0, h - self.trigger_size[-2]], 1)[0]
        if index in self.selected_indices:
            image[:, h_start:(h_start + self.trigger_size[-1]), w_start:(w_start + self.trigger_size[-2])]  = image[:, h_start:(h_start + self.trigger_size[-1]), w_start:(w_start + self.trigger_size[-2])] * torch.logical_not(self.trigger.get_mask())
            image[:, h_start:(h_start + self.trigger_size[-1]), w_start:(w_start + self.trigger_size[-2])] = image[:, h_start:(h_start + self.trigger_size[-1]), w_start:(w_start + self.trigger_size[-2])] + self.trigger.get_data()
            label = self.trigger_class

        return image, label
