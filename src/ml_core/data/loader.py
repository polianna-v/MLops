from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import numpy as np
from .pcam import PCAMDataset
from PIL import Image

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    # the images in the dataset are already 224*224, so 
    # resizing is unnecessary

    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)), 
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),  
        transforms.ToTensor()
    ])

    # TODO: Define Paths for X and Y (train and val)
    x_path_train = base_path/"camelyonpatch_level_2_split_train_x.h5"
    y_path_train = base_path/"camelyonpatch_level_2_split_train_y.h5"

    x_path_val = base_path/"camelyonpatch_level_2_split_valid_x.h5"
    y_path_val = base_path/"camelyonpatch_level_2_split_valid_y.h5"

    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(x_path_train, y_path_train, train_transform)
    val_dataset = PCAMDataset(x_path_val, y_path_val, val_transform)

    # TODO: Create DataLoaders
    
    y_data = np.array([int(y) for y in train_dataset.y_data])  # list comprehension ensures 1D

    classes, class_sample_count = np.unique(y_data, return_counts=True)
    weights = 1.0 / class_sample_count
    sample_weights = np.array([weights[t] for t in y_data])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    batch_size = data_cfg.get("batch_size", 10) 
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    return train_loader, val_loader
