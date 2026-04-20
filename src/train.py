import sys
sys.path.append(".")

import torch
import torch.nn as nn
from model import model
from config import LR, NUM_EPOCHS, NUM_CLASSES
from dataset import train_loader, val_loader

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
best_val_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(NUM_EPOCHS):

    total_loss = 0
    model.train()
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
    model.eval()
    val_loss = 0;
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"  → New best model saved!")
    

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}")

