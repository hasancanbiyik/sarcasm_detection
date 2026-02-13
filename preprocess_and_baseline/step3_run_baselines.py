#!/usr/bin/env python3
"""
Step 3: Run Baseline Models
===========================
This script trains and evaluates baseline models on the preprocessed data.

Tier 1: Random and Majority baselines (sanity check)
Tier 2: TF-IDF + Logistic Regression / SVM
Tier 3: XLM-R embeddings + Logistic Regression (optional, requires torch)

Run this AFTER step2_preprocess.py

Input:
    - outputs/data_splits/*.csv (raw splits)
    - outputs/preprocessed/*.csv (preprocessed data)

Output:
    - outputs/predictions/*.csv (per-example predictions)
    - outputs/results_summary.csv (aggregated metrics)

Author: Hasan Can Biyik
Date: January 2026
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input directories
SPLITS_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits"
PREPROCESSED_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/preprocessed"

# Output directories
OUTPUT_DIR = "outputs"
PREDICTIONS_DIR = "outputs/predictions"

# Random seed
RANDOM_SEED = 42

# Which tiers to run
RUN_TIER_1 = True  # Random/Majority baselines
RUN_TIER_2 = True  # TF-IDF + LogReg/SVM
RUN_TIER_3 = True  # XLM-R embeddings (requires torch/transformers)

# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    
    return metrics


def save_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str,
    condition: str,
    split_name: str,
    output_dir: Path
) -> None:
    """Save predictions to CSV."""
    pred_df = pd.DataFrame({
        'example_id': range(len(df)),
        'context': df['context'].values,
        'text': df['text'].values,
        'true_label': df['label'].values,
        'predicted_label': y_pred,
        'correct': (df['label'].values == y_pred).astype(int),
        'confidence': y_prob[:, 1] if y_prob is not None else None,
        'model_name': model_name,
        'condition': condition,
    })
    
    filename = f"{model_name}_{condition}_{split_name}.csv"
    pred_df.to_csv(output_dir / filename, index=False)


# =============================================================================
# TIER 1: RANDOM AND MAJORITY BASELINES
# =============================================================================

def run_tier1(train_df: pd.DataFrame, test_df: pd.DataFrame, gold_df: pd.DataFrame, 
              predictions_dir: Path) -> List[Dict]:
    """Run Tier 1 baselines: Random and Majority."""
    
    print("\n" + "=" * 60)
    print("TIER 1: RANDOM AND MAJORITY BASELINES")
    print("=" * 60)
    
    results = []
    
    # Get training label distribution
    train_labels = train_df['label'].values
    majority_class = int(pd.Series(train_labels).mode()[0])
    class_probs = [
        (train_labels == 0).mean(),
        (train_labels == 1).mean()
    ]
    
    print(f"   Training majority class: {majority_class}")
    print(f"   Class probabilities: {class_probs}")
    
    for split_name, eval_df in [('test', test_df), ('gold', gold_df)]:
        y_true = eval_df['label'].values
        n = len(y_true)
        
        # Random baseline
        np.random.seed(RANDOM_SEED + (0 if split_name == 'test' else 1))
        y_pred_random = np.random.choice([0, 1], size=n, p=class_probs)
        
        metrics_random = evaluate_predictions(y_true, y_pred_random)
        results.append({
            'tier': 'tier1',
            'model': 'random',
            'condition': 'NA',
            'dataset': split_name,
            **metrics_random
        })
        
        print(f"\n   Random on {split_name}:")
        print(f"      Accuracy: {metrics_random['accuracy']:.4f}")
        print(f"      F1 Macro: {metrics_random['f1_macro']:.4f}")
        
        # Majority baseline
        y_pred_majority = np.full(n, majority_class)
        
        metrics_majority = evaluate_predictions(y_true, y_pred_majority)
        results.append({
            'tier': 'tier1',
            'model': 'majority',
            'condition': 'NA',
            'dataset': split_name,
            **metrics_majority
        })
        
        print(f"\n   Majority on {split_name}:")
        print(f"      Accuracy: {metrics_majority['accuracy']:.4f}")
        print(f"      F1 Macro: {metrics_majority['f1_macro']:.4f}")
    
    return results


# =============================================================================
# TIER 2: TF-IDF + CLASSICAL ML
# =============================================================================

def run_tier2(preprocessed_dir: Path, predictions_dir: Path) -> List[Dict]:
    """Run Tier 2 baselines: TF-IDF + LogReg/SVM."""
    
    print("\n" + "=" * 60)
    print("TIER 2: TF-IDF + LOGISTIC REGRESSION / SVM")
    print("=" * 60)
    
    results = []
    
    # Define conditions to run
    conditions = [
        ('text_only', 'original'),
        ('text_only', 'normalized'),
        ('context_text', 'original'),
        ('context_text', 'normalized'),
        ('context_only', 'original'),
        ('context_only', 'normalized'),
    ]
    
    for input_type, name_variant in conditions:
        condition_name = f"tfidf_{input_type}_{name_variant}"
        print(f"\n--- Condition: {condition_name} ---")
        
        # Load preprocessed data
        train_df = pd.read_csv(preprocessed_dir / f"train_tfidf_{input_type}_{name_variant}.csv")
        val_df = pd.read_csv(preprocessed_dir / f"val_tfidf_{input_type}_{name_variant}.csv")
        test_df = pd.read_csv(preprocessed_dir / f"test_tfidf_{input_type}_{name_variant}.csv")
        gold_df = pd.read_csv(preprocessed_dir / f"gold_tfidf_{input_type}_{name_variant}.csv")
        
        # Prepare data
        X_train = train_df['preprocessed'].values
        y_train = train_df['label'].values
        X_val = val_df['preprocessed'].values
        y_val = val_df['label'].values
        X_test = test_df['preprocessed'].values
        y_test = test_df['label'].values
        X_gold = gold_df['preprocessed'].values
        y_gold = gold_df['label'].values
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        X_gold_tfidf = vectorizer.transform(X_gold)
        
        # Train and evaluate models
        models = {
            'logreg': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            'svm': SVC(kernel='linear', probability=True, random_state=RANDOM_SEED)
        }
        
        for model_name, model in models.items():
            full_model_name = f"tier2_tfidf_{model_name}"
            print(f"   Training {model_name}...")
            
            model.fit(X_train_tfidf, y_train)
            
            for split_name, X_eval, y_eval, eval_df in [
                ('test', X_test_tfidf, y_test, test_df),
                ('gold', X_gold_tfidf, y_gold, gold_df)
            ]:
                y_pred = model.predict(X_eval)
                y_prob = model.predict_proba(X_eval)
                
                metrics = evaluate_predictions(y_eval, y_pred)
                
                results.append({
                    'tier': 'tier2',
                    'model': f'tfidf_{model_name}',
                    'condition': f"{input_type}_{name_variant}",
                    'dataset': split_name,
                    **metrics
                })
                
                # Save predictions
                save_predictions(
                    eval_df, y_pred, y_prob,
                    full_model_name, f"{input_type}_{name_variant}",
                    split_name, predictions_dir
                )
                
                print(f"      {split_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    return results


# =============================================================================
# TIER 3: XLM-R EMBEDDINGS
# =============================================================================

def run_tier3(preprocessed_dir: Path, predictions_dir: Path) -> List[Dict]:
    """Run Tier 3 baselines: XLM-R embeddings + LogReg."""
    
    print("\n" + "=" * 60)
    print("TIER 3: XLM-R EMBEDDINGS + LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Check if torch is available
    try:
        import torch
        from transformers import XLMRobertaModel, XLMRobertaTokenizer
        from tqdm import tqdm
    except ImportError:
        print("   ⚠️  torch/transformers not installed. Skipping Tier 3.")
        print("   Install with: pip install torch transformers")
        return []
    
    results = []
    
    # Load model
    print("   Loading XLM-RoBERTa model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    model.eval()
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("   Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("   Using CUDA")
    else:
        device = torch.device('cpu')
        print("   Using CPU")
    
    model.to(device)
    
    def extract_embeddings(texts: List[str], pooling: str = 'cls', batch_size: int = 16) -> np.ndarray:
        """Extract embeddings from texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"   Extracting {pooling}"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                if pooling == 'cls':
                    batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
                else:  # mean pooling
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_hidden = hidden_states * attention_mask
                    sum_hidden = masked_hidden.sum(dim=1)
                    count = attention_mask.sum(dim=1)
                    batch_embeddings = (sum_hidden / count).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    # Define conditions
    conditions = [
        ('text_only', 'original'),
        ('text_only', 'normalized'),
        ('context_text', 'original'),
        ('context_text', 'normalized'),
        ('context_only', 'original'),
        ('context_only', 'normalized'),
    ]
    
    pooling_methods = ['cls', 'mean']
    
    for input_type, name_variant in conditions:
        condition_base = f"embeddings_{input_type}_{name_variant}"
        print(f"\n--- Condition: {condition_base} ---")
        
        # Load preprocessed data
        train_df = pd.read_csv(preprocessed_dir / f"train_{condition_base}.csv")
        test_df = pd.read_csv(preprocessed_dir / f"test_{condition_base}.csv")
        gold_df = pd.read_csv(preprocessed_dir / f"gold_{condition_base}.csv")
        
        for pooling in pooling_methods:
            print(f"\n   Pooling: {pooling}")
            
            # Extract embeddings
            X_train = extract_embeddings(train_df['preprocessed'].tolist(), pooling)
            X_test = extract_embeddings(test_df['preprocessed'].tolist(), pooling)
            X_gold = extract_embeddings(gold_df['preprocessed'].tolist(), pooling)
            
            y_train = train_df['label'].values
            y_test = test_df['label'].values
            y_gold = gold_df['label'].values
            
            # Train LogReg
            print("   Training Logistic Regression...")
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            clf.fit(X_train, y_train)
            
            full_model_name = f"tier3_xlmr_logreg"
            condition_name = f"{input_type}_{name_variant}_{pooling}"
            
            for split_name, X_eval, y_eval, eval_df in [
                ('test', X_test, y_test, test_df),
                ('gold', X_gold, y_gold, gold_df)
            ]:
                y_pred = clf.predict(X_eval)
                y_prob = clf.predict_proba(X_eval)
                
                metrics = evaluate_predictions(y_eval, y_pred)
                
                results.append({
                    'tier': 'tier3',
                    'model': 'xlmr_logreg',
                    'condition': condition_name,
                    'dataset': split_name,
                    **metrics
                })
                
                # Save predictions
                save_predictions(
                    eval_df, y_pred, y_prob,
                    full_model_name, condition_name,
                    split_name, predictions_dir
                )
                
                print(f"      {split_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 3: RUN BASELINE MODELS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Create output directories
    predictions_dir = Path(PREDICTIONS_DIR)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path(SPLITS_DIR)
    preprocessed_dir = Path(PREPROCESSED_DIR)
    
    # Load raw splits for Tier 1
    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    gold_df = pd.read_csv(splits_dir / "gold.csv")
    
    print(f"Loaded splits: Train={len(train_df)}, Test={len(test_df)}, Gold={len(gold_df)}")
    
    all_results = []
    
    # Run tiers
    if RUN_TIER_1:
        results = run_tier1(train_df, test_df, gold_df, predictions_dir)
        all_results.extend(results)
    
    if RUN_TIER_2:
        results = run_tier2(preprocessed_dir, predictions_dir)
        all_results.extend(results)
    
    if RUN_TIER_3:
        results = run_tier3(preprocessed_dir, predictions_dir)
        all_results.extend(results)
    
    # Save summary
    print("\n" + "=" * 60)
    print("SAVING RESULTS SUMMARY")
    print("=" * 60)
    
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    display_cols = ['tier', 'model', 'condition', 'dataset', 'accuracy', 'f1_macro']
    print(summary_df[display_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"Predictions saved to: {predictions_dir}")
    print(f"Summary saved to: {summary_path}")
    print()


if __name__ == "__main__":
    main()
