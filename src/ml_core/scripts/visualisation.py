# scripts/eda_pcam.py

from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image

# Make sure your project imports work
from data.pcam import PCAMDataset
from data.loader import get_dataloaders


# ------------------------------
# Config
# ------------------------------
config = {
    "data": {
        "data_path": "/gpfs/home3/scur2387/surfdrive",  # full path to H5 files
        "batch_size": 10
    }
}

# ------------------------------
# Load Datasets
# ------------------------------
train_loader, val_loader = get_dataloaders(config)
train_dataset = train_loader.dataset
val_dataset = val_loader.dataset

# ------------------------------
# 1️⃣ Class Distribution
# ------------------------------
y_train = np.array([int(y) for _, y in train_dataset])
counter = Counter(y_train)

plt.figure(figsize=(5,4))
plt.bar(counter.keys(), counter.values(), color=['skyblue', 'salmon'])
plt.xticks([0, 1])
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Class Distribution in Training Set")
plt.show()

# ------------------------------
# 2️⃣ Random Sample Images
# ------------------------------
def show_samples(dataset, num_samples=6):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = [dataset[i][0] for i in indices]
    labels = [dataset[i][1].item() for i in indices]

    grid_img = make_grid(images, nrow=3)
    plt.figure(figsize=(10,5))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Sample Images with Labels: {labels}")
    plt.show()

show_samples(train_dataset)

# ------------------------------
# 3️⃣ Dataset Mean and Std
# ------------------------------
all_images = np.stack([train_dataset[i][0].numpy() for i in range(len(train_dataset))], axis=0)
mean = np.mean(all_images, axis=(0, 2, 3))
std = np.std(all_images, axis=(0, 2, 3))
print("Mean per channel:", mean)
print("Std per channel:", std)

# ------------------------------
# 4️⃣ Pixel Intensity Histogram
# ------------------------------
first_img = train_dataset[0][0].permute(1, 2, 0).numpy()  # HWC
plt.figure(figsize=(6,4))
plt.hist(first_img.ravel(), bins=256, color='gray')
plt.title("Pixel Intensity Distribution of First Image")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()
