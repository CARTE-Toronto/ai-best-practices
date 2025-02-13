import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from pathlib import Path

# Define paths
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

train_raw_dir = RAW_DATA_DIR / 'train'
test_raw_dir = RAW_DATA_DIR / 'test'

train_processed_dir = PROCESSED_DATA_DIR / 'train'
test_processed_dir = PROCESSED_DATA_DIR / 'test'

# Create directories if they don't exist
train_processed_dir.mkdir(parents=True, exist_ok=True)
test_processed_dir.mkdir(parents=True, exist_ok=True)

# Define a basic transform to convert images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset without normalization
train_dataset = datasets.ImageFolder(train_raw_dir, transform=transform)

loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)

for images, _ in loader:
    mean += torch.mean(images, dim=(0, 2, 3))
    std += torch.std(images, dim=(0, 2, 3))

mean /= len(train_dataset)
std /= len(train_dataset)

print(f'Mean: {mean.tolist()}')
print(f'Std: {std.tolist()}')

# Save the mean and std to a file
with open('data/processed/mean_std.pkl', 'wb') as f:
    pickle.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)