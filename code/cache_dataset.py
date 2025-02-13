import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Define paths
RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

train_raw_dir = RAW_DATA_DIR / 'train'
test_raw_dir = RAW_DATA_DIR / 'test'

train_processed_file = PROCESSED_DATA_DIR / 'train' / 'data.pt'
test_processed_file = PROCESSED_DATA_DIR / 'test' / 'data.pt'

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
(train_processed_file.parent).mkdir(parents=True, exist_ok=True)
(test_processed_file.parent).mkdir(parents=True, exist_ok=True)

# Load mean and std from data/processed/mean_std.pkl
with open('data/processed/mean_std.pkl', 'rb') as f:
    mean_std = pickle.load(f)

mean = torch.tensor(mean_std['mean'])
std = torch.tensor(mean_std['std'])

# Define a transform to normalize images
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class DiskCachedDataset(Dataset):
    def __init__(self, raw_dir, processed_file, transform=None):
        self.raw_dir = raw_dir
        self.processed_file = processed_file
        self.transform = transform

        # Get class names (sorted for consistent indexing)
        classes_file = self.processed_file.parent / 'classes.json'

        if classes_file.exists():
            with open(classes_file, 'r') as f:
                self.classes = json.load(f)
        else:
                self.classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
                with open(classes_file, 'w') as f:
                    json.dump(self.classes, f)

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if not self.processed_file.exists():
            self.cache_data()
        
        self.data, self.labels = torch.load(self.processed_file, weights_only=True)

    def cache_data(self):
        data = []
        labels = []

        for class_name in self.classes:
            class_dir = self.raw_dir / class_name
            for image_file in tqdm(class_dir.glob('*.jpg'), desc=f"Processing {class_name}"):
                image = Image.open(image_file).convert("RGB")
                data.append(self.transform(image))
                labels.append(self.class_to_idx[class_name])  # Map folder name to integer label

        data = torch.stack(data)
        labels = torch.tensor(labels)

        torch.save((data, labels), self.processed_file)
        print(f"Dataset cached at {self.processed_file}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create datasets
train_dataset = DiskCachedDataset(train_raw_dir, train_processed_file, transform=train_transforms)
test_dataset = DiskCachedDataset(test_raw_dir, test_processed_file, transform=test_transforms)

print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")