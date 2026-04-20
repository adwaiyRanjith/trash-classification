import sys
import os
sys.path.append(".")

import torch
from model import model
from config import NUM_CLASSES
from dataset import test_loader

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the best model weights
model = model.to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

# evaluate on test set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.2%}")