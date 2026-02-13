#!/usr/bin/env python3
"""
Step 1: Data Loading and Splitting
==================================
This script loads the orange (main) and gold (subset) datasets, removes gold examples from orange,
and creates stratified train/val/test splits.

Run this FIRST before any other scripts.

Input:
    - data/main_dataset_orange.csv
    - data/sample_dataset_gold.csv

Output:
    - outputs/data_splits/train.csv
    - outputs/data_splits/val.csv  
    - outputs/data_splits/test.csv
    - outputs/data_splits/gold.csv
    - outputs/data_splits/split_info.txt (summary)

"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS AS NEEDED
# =============================================================================

# Input paths (relative to where you run the script, or use absolute paths)
ORANGE_DATASET_PATH = "/Users/hasancan/Desktop/sarcasm_detection/data/main_dataset_orange.csv"
GOLD_DATASET_PATH = "/Users/hasancan/Desktop/sarcasm_detection/data/sample_dataset_gold.csv"

# Output directory
OUTPUT_DIR = "outputs/data_splits"

# Split configuration
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 1: DATA LOADING AND SPLITTING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATASETS
    # -------------------------------------------------------------------------
    print("1. LOADING DATASETS")
    print("-" * 40)
    
    orange_df = pd.read_csv(ORANGE_DATASET_PATH)
    gold_df = pd.read_csv(GOLD_DATASET_PATH)
    
    print(f"   Orange dataset loaded: {len(orange_df)} examples")
    print(f"   Gold dataset loaded: {len(gold_df)} examples")
    print()
    
    # Show column info
    print(f"   Orange columns: {list(orange_df.columns)}")
    print(f"   Gold columns: {list(gold_df.columns)}")
    print()
    
    # Show label distribution
    orange_dist = orange_df['label'].value_counts().sort_index()
    gold_dist = gold_df['label'].value_counts().sort_index()
    
    print(f"   Orange label distribution:")
    print(f"      Label 0 (literal): {orange_dist.get(0, 0)}")
    print(f"      Label 1 (sarcastic): {orange_dist.get(1, 0)}")
    print()
    
    print(f"   Gold label distribution:")
    print(f"      Label 0 (literal): {gold_dist.get(0, 0)}")
    print(f"      Label 1 (sarcastic): {gold_dist.get(1, 0)}")
    print()
    
    # -------------------------------------------------------------------------
    # 2. REMOVE GOLD EXAMPLES FROM ORANGE
    # -------------------------------------------------------------------------
    print("2. SEPARATING GOLD FROM ORANGE")
    print("-" * 40)
    
    # Match on 'text' column (should be unique identifier)
    gold_texts = set(gold_df['text'].str.strip())
    
    # Create mask for examples NOT in gold
    mask = ~orange_df['text'].str.strip().isin(gold_texts)
    orange_clean_df = orange_df[mask].reset_index(drop=True)
    
    removed_count = len(orange_df) - len(orange_clean_df)
    
    print(f"   Gold examples found in orange: {removed_count}")
    print(f"   Orange-clean size: {len(orange_clean_df)}")
    print()
    
    # Verify all gold examples were found
    if removed_count != len(gold_df):
        print(f"   ⚠️  WARNING: Expected to remove {len(gold_df)} but removed {removed_count}")
        print(f"   Some gold examples may not be in orange dataset!")
    else:
        print(f"   ✓ All {len(gold_df)} gold examples removed from orange")
    print()
    
    # Show orange-clean distribution
    clean_dist = orange_clean_df['label'].value_counts().sort_index()
    print(f"   Orange-clean label distribution:")
    print(f"      Label 0 (literal): {clean_dist.get(0, 0)}")
    print(f"      Label 1 (sarcastic): {clean_dist.get(1, 0)}")
    print()
    
    # -------------------------------------------------------------------------
    # 3. CREATE STRATIFIED SPLITS
    # -------------------------------------------------------------------------
    print("3. CREATING STRATIFIED SPLITS")
    print("-" * 40)
    print(f"   Target ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    X = orange_clean_df[['context', 'text']]
    y = orange_clean_df['label']
    
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    # Second split: 50% of temp for val, 50% for test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )
    
    # Create DataFrames
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    
    # Show split results
    print(f"   Train set: {len(train_df)} examples")
    train_dist = train_df['label'].value_counts().sort_index()
    print(f"      Label 0: {train_dist.get(0, 0)} ({train_dist.get(0, 0)/len(train_df)*100:.1f}%)")
    print(f"      Label 1: {train_dist.get(1, 0)} ({train_dist.get(1, 0)/len(train_df)*100:.1f}%)")
    print()
    
    print(f"   Val set: {len(val_df)} examples")
    val_dist = val_df['label'].value_counts().sort_index()
    print(f"      Label 0: {val_dist.get(0, 0)} ({val_dist.get(0, 0)/len(val_df)*100:.1f}%)")
    print(f"      Label 1: {val_dist.get(1, 0)} ({val_dist.get(1, 0)/len(val_df)*100:.1f}%)")
    print()
    
    print(f"   Test set: {len(test_df)} examples")
    test_dist = test_df['label'].value_counts().sort_index()
    print(f"      Label 0: {test_dist.get(0, 0)} ({test_dist.get(0, 0)/len(test_df)*100:.1f}%)")
    print(f"      Label 1: {test_dist.get(1, 0)} ({test_dist.get(1, 0)/len(test_df)*100:.1f}%)")
    print()
    
    print(f"   Gold set (held out): {len(gold_df)} examples")
    print(f"      Label 0: {gold_dist.get(0, 0)} ({gold_dist.get(0, 0)/len(gold_df)*100:.1f}%)")
    print(f"      Label 1: {gold_dist.get(1, 0)} ({gold_dist.get(1, 0)/len(gold_df)*100:.1f}%)")
    print()
    
    # Verify total
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"   Total (train+val+test): {total}")
    print(f"   Expected (orange_clean): {len(orange_clean_df)}")
    if total == len(orange_clean_df):
        print(f"   ✓ Totals match!")
    else:
        print(f"   ⚠️  WARNING: Totals don't match!")
    print()
    
    # -------------------------------------------------------------------------
    # 4. SAVE SPLITS
    # -------------------------------------------------------------------------
    print("4. SAVING SPLITS")
    print("-" * 40)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    gold_df.to_csv(output_dir / "gold.csv", index=False)
    
    print(f"   Saved: {output_dir / 'train.csv'}")
    print(f"   Saved: {output_dir / 'val.csv'}")
    print(f"   Saved: {output_dir / 'test.csv'}")
    print(f"   Saved: {output_dir / 'gold.csv'}")
    print()
    
    # Save summary info
    summary_path = output_dir / "split_info.txt"
    with open(summary_path, 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("SOURCE FILES:\n")
        f.write(f"  Orange: {ORANGE_DATASET_PATH}\n")
        f.write(f"  Gold: {GOLD_DATASET_PATH}\n\n")
        
        f.write("ORIGINAL SIZES:\n")
        f.write(f"  Orange: {len(orange_df)}\n")
        f.write(f"  Gold: {len(gold_df)}\n")
        f.write(f"  Orange-clean (after removing gold): {len(orange_clean_df)}\n\n")
        
        f.write("SPLIT SIZES:\n")
        f.write(f"  Train: {len(train_df)} ({len(train_df)/len(orange_clean_df)*100:.1f}%)\n")
        f.write(f"  Val: {len(val_df)} ({len(val_df)/len(orange_clean_df)*100:.1f}%)\n")
        f.write(f"  Test: {len(test_df)} ({len(test_df)/len(orange_clean_df)*100:.1f}%)\n")
        f.write(f"  Gold: {len(gold_df)} (held-out)\n\n")
        
        f.write("LABEL DISTRIBUTIONS:\n")
        f.write(f"  Train - Label 0: {train_dist.get(0, 0)}, Label 1: {train_dist.get(1, 0)}\n")
        f.write(f"  Val   - Label 0: {val_dist.get(0, 0)}, Label 1: {val_dist.get(1, 0)}\n")
        f.write(f"  Test  - Label 0: {test_dist.get(0, 0)}, Label 1: {test_dist.get(1, 0)}\n")
        f.write(f"  Gold  - Label 0: {gold_dist.get(0, 0)}, Label 1: {gold_dist.get(1, 0)}\n")
    
    print(f"   Saved: {summary_path}")
    print()
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"Next: Run step2_preprocess.py to preprocess the splits")
    print()


if __name__ == "__main__":
    main()
