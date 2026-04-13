import sys
sys.path.append(".")

import torch
import torch.nn as nn
from model import model
from config import LR, NUM_EPOCHS, NUM_CLASSES
from dataset import train_loader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = model.to(device)
print(f"device is {device}")
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)
count = 0;


for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        count = count + 1
        print(f"count = {count}")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.4f}")

