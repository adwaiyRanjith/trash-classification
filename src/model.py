from torchvision import models
import torch.nn as nn

import sys
sys.path.append(".")
from config import NUM_CLASSES

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

in_features = model.fc.in_features

model.fc = nn.Linear(in_features, NUM_CLASSES)