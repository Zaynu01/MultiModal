# Multimodal CTR Prediction Challenge

This repository contains our work for a **Multimodal Click-Through Rate (MM-CTR) Prediction** project based on the WWW 2025 EReL@MIR challenge setting.

The project has two connected parts:

- **Task 1: Multimodal Item Embedding Learning**
- **Task 2: Multimodal CTR Prediction**

The goal is to use multimodal item information and user interaction data to improve CTR prediction performance.

## Project Summary

In Task 1, we built a modular PyTorch pipeline that learns **128-dimensional item embeddings** from multimodal item features and auxiliary CTR supervision.

In Task 2, we built a memory-efficient neural CTR model that uses user IDs, item IDs, engagement features, and multimodal item embeddings to predict click probability for the test set.

Our final Task 2 submission achieved:

```text
Validation AUC: 0.783
Codabench score: 0.81
Test predictions: 379,142 rows
```

## Repository Structure

```text
Project/
├── Data/
│   ├── item_info.parquet
│   ├── item_feature.parquet
│   ├── item_emb.parquet
│   ├── item_seq.parquet
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
├── embeddings.ipynb
├── MM_CTR_Task2_v3.ipynb
├── parquet.py
├── outputs/
├── README.md
└── .gitignore
```

## Data Files

The project expects the challenge data under the `Data/` folder.

Important files:

- `item_info.parquet`: item IDs, item tags, padding item, and provided item embeddings.
- `item_feature.parquet`: item titles, tags, BERT text embeddings, CLIP-RN50 image embeddings, likes level, and views level.
- `train.parquet`: user interaction training data with labels.
- `valid.parquet`: validation interaction data with labels.
- `test.parquet`: test interaction data without labels.
- `item_emb.parquet`: organizer-provided 128-dimensional multimodal item embeddings used in Task 2.

Large data files are usually not pushed to GitHub. If they are missing, place them manually inside `Data/`.

## Task 1: Multimodal Item Embedding Learning

Notebook:

```text
embeddings.ipynb
```

Task 1 learns a 128-dimensional embedding for each item.

### Task 1 Inputs

The item encoder uses:

- BERT text embeddings: `txt_emb_BERT`, 768 dimensions.
- CLIP-RN50 image embeddings: `img_emb_CLIPRN50`, 1024 dimensions.
- item tags: 5 categorical tag slots.
- train-only behavior features:
  - normalized item count from `train.parquet`
  - normalized log item count
  - train item CTR
  - item-level likes level
  - item-level views level

The train-derived behavior features are computed only from `train.parquet` to avoid validation/test leakage.

### Task 1 Architecture

The model contains:

- `TextEncoder`
- `ImageEncoder`
- `MetadataEncoder`
- `BehaviorEncoder`
- `ItemEncoder`
- `UserEncoder`
- auxiliary `CTRHead`

The `ItemEncoder` fuses:

```text
text + image + tags + behavior features -> 128-dimensional item embedding
```

The auxiliary CTR head is used only during training. The final Task 1 output is the learned item embedding matrix.

### Task 1 Observations

During experimentation, simple text/image/tag similarity was weak for CTR prediction. A behavior baseline was stronger:

```text
train_count_auc:      ~0.579
log_train_count_auc:  ~0.579
train_ctr_auc:        ~0.528
behavior_combined:    ~0.531
```

This showed that item popularity and behavior patterns were important recommendation signals.

However, training on small subsets caused strong overfitting:

```text
training AUC increased quickly
validation AUC stayed close to 0.50-0.51
```

Because of this, Task 1 was useful for experimentation and diagnosis, but the final Task 2 pipeline used the organizer-provided `item_emb.parquet` embeddings for stronger stability.

## Task 2: Multimodal CTR Prediction

Notebook:

```text
MM_CTR_Task2_v3.ipynb
```

Task 2 predicts the probability that a user clicks a target item.

### Task 2 Inputs

The CTR model uses:

- user ID
- item ID
- organizer-provided multimodal item embedding
- likes level
- views level

### Memory-Efficient Preprocessing

To avoid memory issues, the notebook does not merge all 128 embedding columns directly into the click dataframe.

Instead, it builds an embedding lookup matrix:

```text
item_id -> 128-dimensional embedding
```

During training, the PyTorch dataset retrieves item embeddings by index.

Additional preprocessing includes:

- numeric downcasting
- label encoding for user IDs and item IDs
- scaling likes/views features
- PyTorch `Dataset` and `DataLoader`

## Task 2 Model Architecture

The final CTR model uses multiple branches:

- user ID embedding branch
- item ID embedding branch
- multimodal item embedding projection branch
- engagement feature branch

The model also includes interaction features:

- user × item interaction
- user × multimodal embedding interaction
- dot product interaction

These features are concatenated and passed to a residual MLP.

The prediction head outputs:

```text
pCTR in [0, 1]
```

## Task 2 Training Setup

Main training choices:

- loss: Binary Cross Entropy
- optimizer: Adam
- learning rate: `2e-4`
- weight decay: `3e-5`
- batch size: `2048`
- scheduler: cosine learning rate scheduler
- regularization: dropout and batch normalization
- early stopping based on validation AUC

## Results

Final Task 2 results:

```text
Validation AUC: 0.783
Codabench score: 0.81
Test predictions: 379,142
```

The generated submission contained valid click probabilities between 0 and 1.

## How To Run

1. Clone the repository.

```bash
git clone https://github.com/Zaynu01/MultiModal.git
cd MultiModal
```

2. Create or activate a Python environment.

Example on Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install the required packages.

The notebooks use:

```text
python
pandas
numpy
pyarrow
scikit-learn
matplotlib
torch
```

Install manually if needed:

```bash
pip install pandas numpy pyarrow scikit-learn matplotlib torch
```

4. Place the challenge data files inside:

```text
Data/
```

5. Run the notebooks:

```text
embeddings.ipynb        # Task 1
MM_CTR_Task2_v3.ipynb   # Task 2
```

For the final result, run the Task 2 notebook and generate the submission predictions.

## Important Notes

- Task 1 focuses on learning item embeddings, not final CTR prediction.
- Task 2 focuses on predicting pCTR for the test set.
- Task 1 experiments showed that behavior features were more useful than simple semantic similarity.
- The final Task 2 model used organizer-provided item embeddings for stronger performance and stability.
- Large files such as parquet datasets, model checkpoints, and output predictions should generally be excluded from GitHub.

## Authors

Project developed for a Multimodal CTR Prediction.

