import sys
import os
sys.path.append(".")

import torch
import numpy as np
from sklearn.metrics import classification_report

from model import model
from dataset import test_loader

CLASS_NAMES = ["battery", "biological", "cardboard", "glass", "metal", "paper", "plastic", "trash"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"Test Accuracy: {accuracy:.2%}\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

os.makedirs("results", exist_ok=True)

# export model to ONNX for Netron visualization
dummy_input = torch.zeros(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "results/model.onnx",
    input_names=["image"],
    output_names=["class_scores"],
    dynamic_axes={"image": {0: "batch_size"}, "class_scores": {0: "batch_size"}},
    opset_version=18
)
print("Saved results/model.onnx — open with Netron")
