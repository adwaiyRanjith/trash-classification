import sys
sys.path.append(".")

from config import DATA_DIR, BATCH_SIZE
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# building the transformer that transforms images to tensors
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

ds = datasets.ImageFolder(DATA_DIR, transform)

# calculating the different sizes of data chunks
total_img = len(ds)
train_size = int(.8 * total_img)
val_size = int(.1 * total_img)
test_size = total_img - train_size - val_size

# split up the dataset into train, val, and test
train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, (train_size, val_size, test_size))

# make em into loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)