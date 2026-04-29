# Trash Classification Model

### What this project does
Fine-tunes a ResNet18 (pretrained on ImageNet) to classify images of trash into 8 categories: battery, biological, cardboard, glass, metal, paper, plastic, and trash. The model replaces ResNet18's final layer with a dropout + linear head, then trains end-to-end on a labeled image dataset.

### Dataset
- **8,918 images** across 8 classes, stored in ImageFolder format under `data/raw/original/`
- Split 80/10/10 into train/val/test (7,134 / 892 / 892)
- Classes are imbalanced (glass: 1,736 images vs. trash: 453), handled with `WeightedRandomSampler`
- The dataset is **not committed** to the repo (listed in `.gitignore`). For Colab/Modal runs, it lives as `cleaned_trash_data.zip` on Google Drive / a Modal volume
- For Colab/Modal training, download the dataset from "https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2" as a zip file and place it into your drive.

### Model & training
- **Architecture:** ResNet18 (pretrained) → Dropout(0.5) → Linear(512, 8)
- **Optimizer:** Adam, lr=0.001
- **Scheduler:** StepLR — halves the LR every 5 epochs
- **Early stopping:** patience of 5 epochs on validation loss
- **Max epochs:** 40 (early stopped at epoch 22 in current run)
- **Best checkpoint** saved to `checkpoints/best_model.pth` (not committed)

### Results
| Split | Accuracy |
|-------|----------|
| Validation (best epoch) | ~89% |
| Test | **97.76%** |

### Project structure
```
config.py           — hyperparameters and DATA_DIR (loaded from .env)
src/
  model.py          — ResNet18 model definition
  dataset.py        — data loading, augmentation, splits, weighted sampler
  train.py          — training loop (validation, early stopping, checkpointing)
  evaluate.py       — loads best_model.pth and reports test accuracy
notebooks/
  explore.ipynb     — class distribution + sample visualization
  train.ipynb       — Colab training notebook
  evaluate.ipynb    — Colab evaluation notebook
modal_evaluate.py   — remote evaluation on Modal (A10G GPU)
```

### How to run locally
```
# 1. Download the dataset from:
#    https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
# 2. Unzip and place the contents so the structure looks like:
#    data/raw/original/battery/
#    data/raw/original/glass/
#    ... etc
# 3. Create .env with your data path
echo "DATA_DIR=/path/to/trash-classification/data/raw/original" > .env
# 4. Install dependencies
pip install -r requirements.txt
# 5. Train
python src/train.py
# 6. Evaluate
python src/evaluate.py
# 7. Run webcam classifier (requires best_model.pth in checkpoints/)
python src/predict.py
```

### How to run on Google Colab
Open `notebooks/train.ipynb` — it clones the repo, mounts your Drive, unzips the dataset, runs training, and saves `best_model.pth` back to Drive. Then open `notebooks/evaluate.ipynb` to score the saved checkpoint.

### How to run on Modal
```bash
modal run modal_evaluate.py   # evaluates from Modal volume
```
Requires a Modal account and a volume named `trash-dataset` containing `best_model.pth`.

### Key config options (`config.py`)
| Variable | Description |
|----------|-------------|
| `NUM_CLASSES` | number of trash categories |
| `BATCH_SIZE` | training batch size |
| `LR` | Adam learning rate |
| `NUM_EPOCHS` | max training epochs |
| `PATIENCE` | early stopping patience |
| `USERNAME` | used in `explore.ipynb` for local path construction |
