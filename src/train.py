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


for epoch in range(NUM_EPOCHS):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Loss: {avg_loss:.4f}")

