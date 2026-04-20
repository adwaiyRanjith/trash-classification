# trash-classification
a classification model that uses cv to look at and classify various types of trash


so far this what i got:
Test Accuracy: 97.76%

---

## For Mentors

### What this project does
Fine-tunes a ResNet18 (pretrained on ImageNet) to classify images of trash into 8 categories: battery, biological, cardboard, glass, metal, paper, plastic, and trash. The model replaces ResNet18's final layer with a dropout + linear head, then trains end-to-end on a labeled image dataset.

### Dataset
- **8,918 images** across 8 classes, stored in ImageFolder format under `data/raw/original/`
- Split 80/10/10 into train/val/test (7,134 / 892 / 892)
- Classes are imbalanced (glass: 1,736 images vs. trash: 453), handled with `WeightedRandomSampler`
- The dataset is **not committed** to the repo (listed in `.gitignore`). For Colab/Modal runs, it lives as `cleaned_trash_data.zip` on Google Drive / a Modal volume

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
modal_train.py      — remote training on Modal (A10G GPU)
modal_evaluate.py   — remote evaluation on Modal
```

### How to run locally
```bash
# 1. copy .env.example or create .env with your data path
echo "DATA_DIR=/path/to/data/raw/original" > .env

# 2. install dependencies
pip install -r requirements.txt

# 3. train
python src/train.py        # saves checkpoints/best_model.pth

# 4. evaluate
python src/evaluate.py
```

### How to run on Google Colab
Open `notebooks/train.ipynb` — it clones the repo, mounts your Drive, unzips the dataset, runs training, and saves `best_model.pth` back to Drive. Then open `notebooks/evaluate.ipynb` to score the saved checkpoint.

### How to run on Modal
```bash
modal run modal_train.py      # trains on A10G GPU, saves model to Modal volume
modal run modal_evaluate.py   # evaluates from Modal volume
```
Requires a Modal account and a volume named `trash-dataset` containing `cleaned_trash_data.zip`.

### Key config options (`config.py`)
| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_CLASSES` | number of trash categories |
| `BATCH_SIZE` | training batch size |
| `LR` | Adam learning rate |
| `NUM_EPOCHS` | max training epochs |
| `PATIENCE` | early stopping patience |
| `USERNAME` | used in `explore.ipynb` for local path construction |