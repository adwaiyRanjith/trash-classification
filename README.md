# Trash Classification Model

Fine-tunes a ResNet18 (pretrained on ImageNet) to classify images of trash into 8 categories: battery, biological, cardboard, glass, metal, paper, plastic, and trash.

### Results
| Split | Accuracy |
|-------|----------|
| Validation | 88% |
| Test | 88% |

---

## Quickstart (local)

**1. Clone the repo**
```bash
git clone https://github.com/adwaiyRanjith/trash-classification.git
cd trash-classification
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle — Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) and unzip so the structure looks like:
```
data/raw/original/battery/
data/raw/original/biological/
data/raw/original/cardboard/
...
```

**4. Set the data path**
```bash
echo "DATA_DIR=/absolute/path/to/data/raw/original" > .env
```

**5. Train**
```bash
python src/train.py
```
Saves the best checkpoint to `checkpoints/best_model.pth`.

**6. Evaluate**
```bash
python src/evaluate.py
```
Prints test accuracy and classification report, and exports `results/model.onnx` for visualization in [Netron](https://netron.app).

---

## Run on Google Colab

1. Upload your dataset zip (`cleaned_trash_data.zip`) to Google Drive
2. Open `notebooks/train.ipynb` in Colab and run all cells — it clones the repo, unzips the data, trains the model, and saves `best_model.pth` to your Drive
3. Open `notebooks/evaluate.ipynb` to evaluate the saved checkpoint

---

## Project structure
```
config.py             — hyperparameters and DATA_DIR (loaded from .env)
src/
  model.py            — ResNet18 with custom dropout + linear head
  dataset.py          — data loading, augmentation, splits, weighted sampler
  train.py            — training loop (early stopping, LR scheduler, checkpointing)
  evaluate.py         — evaluation script (accuracy, classification report, ONNX export)
notebooks/
  explore.ipynb       — class distribution and sample visualization
  train.ipynb         — Colab training notebook
  evaluate.ipynb      — Colab evaluation notebook
modal_evaluate.py     — remote evaluation on Modal (A10G GPU)
results/              — generated outputs (plots, ONNX model)
```

---

## Model & training details

- **Architecture:** ResNet18 (pretrained on ImageNet) → Dropout(0.5) → Linear(512 → 8)
- **Optimizer:** Adam, lr=0.001
- **Scheduler:** StepLR — halves LR every 5 epochs
- **Early stopping:** patience of 5 epochs on validation loss
- **Class imbalance:** handled with `WeightedRandomSampler`

## Dataset

- **Source:** [Garbage Classification V2 by Suman Kunwar](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **8,918 images** across 8 classes
- Split 80/10/10 into train / val / test (fixed seed for reproducibility)
- The dataset is not committed to this repo — download it separately

## Key config options (`config.py`)
| Variable | Description |
|----------|-------------|
| `NUM_CLASSES` | number of trash categories |
| `BATCH_SIZE` | training batch size |
| `LR` | Adam learning rate |
| `NUM_EPOCHS` | max training epochs |
| `PATIENCE` | early stopping patience |
| `USERNAME` | used in `explore.ipynb` for local path construction |
