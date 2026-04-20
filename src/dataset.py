import sys
sys.path.append(".")

from config import DATA_DIR, BATCH_SIZE, NUM_CLASSES
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        return self.transform(image), label

    def __len__(self):
        return len(self.subset)

# building the transformer that transforms images to tensors
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

ds = datasets.ImageFolder(DATA_DIR, transform=None)


# calculating the different sizes of data chunks
total_img = len(ds)
train_size = int(.8 * total_img)
val_size = int(.1 * total_img)
test_size = total_img - train_size - val_size

# split up the dataset into train, val, and test
train_raw, val_raw, test_raw = torch.utils.data.random_split(ds, (train_size, val_size, test_size))

class_counts = [ds.targets.count(i) for i in range(NUM_CLASSES)]
class_weights = [1.0 / count for count in class_counts]

train_raw_label_counts = [ds.targets[i] for i in train_raw.indices]

sample_weights = [class_weights[label] for label in train_raw_label_counts]

sampler = torch.utils.data.WeightedRandomSampler(
    weights = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

train_ds = TransformSubset(train_raw, train_transform)
val_ds = TransformSubset(val_raw, val_transform)
test_ds = TransformSubset(test_raw, val_transform)

# make em into loaders

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)