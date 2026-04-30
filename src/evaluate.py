import sys
import os
sys.path.append(".")

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("Saved results/confusion_matrix.png")

# precision / recall / F1 bar chart
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
metrics = {cls: (report[cls]["precision"], report[cls]["recall"], report[cls]["f1-score"]) for cls in CLASS_NAMES}
x = np.arange(len(CLASS_NAMES))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width, [metrics[c][0] for c in CLASS_NAMES], width, label="Precision", color="steelblue")
ax.bar(x,         [metrics[c][1] for c in CLASS_NAMES], width, label="Recall",    color="darkorange")
ax.bar(x + width, [metrics[c][2] for c in CLASS_NAMES], width, label="F1",        color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 per Class")
ax.legend()
plt.tight_layout()
plt.savefig("results/precision_recall_f1.png", dpi=150)
plt.close()
print("Saved results/precision_recall_f1.png")

# loss curve
if os.path.exists("results/history.json"):
    with open("results/history.json") as f:
        history = json.load(f)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=150)
    plt.close()
    print("Saved results/loss_curve.png")

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
