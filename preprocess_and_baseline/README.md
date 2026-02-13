# Sarcasm Detection Pipeline

A step-by-step pipeline for sarcasm detection baseline experiments.

## Directory Structure

```
sarcasm_detection/
├── data/
│   ├── main_dataset_orange.csv      # Main dataset
│   └── sample_dataset_gold.csv      # Held-out gold test set
│
└── preprocess_and_baseline/
    ├── step1_split_data.py          # Step 1: Load and split data
    ├── step2_preprocess.py          # Step 2: Preprocess text
    ├── step3_run_baselines.py       # Step 3: Train and evaluate models
    ├── requirements.txt
    │
    └── outputs/                     # Created by scripts
        ├── data_splits/             # Train/val/test/gold CSVs
        ├── preprocessed/            # Preprocessed variants
        ├── predictions/             # Per-example predictions
        └── results_summary.csv      # Aggregated metrics
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Edit file paths (if needed)

In `step1_split_data.py`, update these paths:

```python
ORANGE_DATASET_PATH = "../data/main_dataset_orange.csv"
GOLD_DATASET_PATH = "../data/sample_dataset_gold.csv"
```

### 3. Run step by step

```bash
# Step 1: Split data
python step1_split_data.py

# Verify outputs before continuing:
ls outputs/data_splits/

# Step 2: Preprocess
python step2_preprocess.py

# Verify outputs before continuing:
ls outputs/preprocessed/

# Step 3: Run baselines
python step3_run_baselines.py

# Check results:
cat outputs/results_summary.csv
ls outputs/predictions/
```

## What Each Step Does

### Step 1: `step1_split_data.py`

1. Loads `main_dataset_orange.csv` and `sample_dataset_gold.csv`
2. Removes gold examples from orange (to prevent data leakage)
3. Creates stratified 80/10/10 train/val/test split
4. Saves splits to `outputs/data_splits/`

**Outputs:**
- `train.csv` - Training set
- `val.csv` - Validation set
- `test.csv` - Test set (from orange)
- `gold.csv` - Held-out gold test set
- `split_info.txt` - Summary statistics

### Step 2: `step2_preprocess.py`

1. Loads splits from Step 1
2. Applies preprocessing for each experimental condition:
   - **TF-IDF**: lowercase, pipe→period
   - **Embeddings**: preserve case, pipe→newline
   - **Name variants**: original vs normalized (SPEAKER:)
   - **Input types**: text_only, context_text, context_only
3. Saves preprocessed files to `outputs/preprocessed/`

**Outputs:**
- 48 preprocessed CSV files (4 splits × 12 conditions)

### Step 3: `step3_run_baselines.py`

1. **Tier 1**: Random and Majority baselines (sanity check)
2. **Tier 2**: TF-IDF + Logistic Regression / SVM
3. **Tier 3**: XLM-R embeddings + Logistic Regression (optional)

**Outputs:**
- `outputs/predictions/*.csv` - Per-example predictions
- `outputs/results_summary.csv` - Aggregated metrics

## Configuration Options

### In `step3_run_baselines.py`:

```python
RUN_TIER_1 = True   # Random/Majority (fast)
RUN_TIER_2 = True   # TF-IDF models (fast)
RUN_TIER_3 = True   # XLM-R models (slow, needs GPU)
```

Set `RUN_TIER_3 = False` if you don't have torch installed.

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| `text_only` | Only the target utterance |
| `context_text` | Context + target combined |
| `context_only` | Only the context (ablation) |
| `original` | Keep character names (SHELDON:, etc.) |
| `normalized` | Replace names with SPEAKER: |
| `tfidf` | Preprocessing for bag-of-words |
| `embeddings` | Preprocessing for transformers |

## Troubleshooting

### "File not found" errors

Check that paths in Step 1 point to your actual data files.

### Tier 3 not running

Install torch and transformers:
```bash
pip install torch transformers tqdm sentencepiece
```

### Memory issues with Tier 3

Reduce batch size in `step3_run_baselines.py`:
```python
X_train = extract_embeddings(texts, pooling, batch_size=8)  # default is 16
```
