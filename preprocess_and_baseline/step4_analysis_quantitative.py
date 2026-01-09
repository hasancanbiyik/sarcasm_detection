#!/usr/bin/env python3
"""
Step 4: Quantitative Error Analysis
===================================
Analyzes error patterns across experimental conditions.

This script provides:
1. Error breakdown by condition (TP/TN/FP/FN, accuracy, F1)
2. Normalization effect analysis (original vs normalized names)
3. Input type effect analysis (text_only vs context_text vs context_only)
4. Cross-condition flip analysis (examples that change correctness between conditions)

Run this AFTER step3_run_baselines.py

Input:
    - outputs/predictions/*.csv

Output:
    - outputs/analysis/error_breakdown.csv
    - outputs/analysis/normalization_effect.csv
    - outputs/analysis/input_type_effect.csv
    - outputs/analysis/flips_by_normalization.csv
    - outputs/analysis/flips_by_input_type.csv
    - outputs/analysis/flips_by_model.csv
    - outputs/analysis/quantitative_report.txt

Author: Hasan Can Biyik
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions"
OUTPUT_DIR = "outputs/quantitative_analysis"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_filename(filename: str) -> Dict:
    """
    Parse prediction filename to extract metadata.
    
    Expected format: {model}_{condition}_{dataset}.csv
    Examples:
        tier2_tfidf_logreg_text_only_original_test.csv
        tier3_xlmr_logreg_context_text_normalized_mean_gold.csv
    """
    name = filename.replace('.csv', '')
    parts = name.split('_')
    
    metadata = {
        'tier': None,
        'model_type': None,
        'classifier': None,
        'input_type': None,
        'names_normalized': None,
        'pooling': None,
        'dataset': None,
    }
    
    # Identify tier
    if 'tier2' in name:
        metadata['tier'] = 'tier2'
    elif 'tier3' in name:
        metadata['tier'] = 'tier3'
    
    # Identify model type
    if 'tfidf' in name:
        metadata['model_type'] = 'tfidf'
    elif 'xlmr' in name:
        metadata['model_type'] = 'xlmr'
    
    # Identify classifier
    if 'logreg' in name:
        metadata['classifier'] = 'logreg'
    elif 'svm' in name:
        metadata['classifier'] = 'svm'
    
    # Identify input type
    if 'text_only' in name:
        metadata['input_type'] = 'text_only'
    elif 'context_text' in name:
        metadata['input_type'] = 'context_text'
    elif 'context_only' in name:
        metadata['input_type'] = 'context_only'
    
    # Identify normalization
    if 'normalized' in name:
        metadata['names_normalized'] = True
    elif 'original' in name:
        metadata['names_normalized'] = False
    
    # Identify pooling (Tier 3 only)
    if '_cls_' in name or name.endswith('_cls'):
        metadata['pooling'] = 'cls'
    elif '_mean_' in name or name.endswith('_mean'):
        metadata['pooling'] = 'mean'
    else:
        metadata['pooling'] = 'NA'
    
    # Identify dataset (last part before .csv)
    if '_test' in name or name.endswith('test'):
        metadata['dataset'] = 'test'
    elif '_gold' in name or name.endswith('gold'):
        metadata['dataset'] = 'gold'
    
    return metadata


def load_all_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load all prediction files and combine into one DataFrame."""
    
    all_dfs = []
    pred_files = list(predictions_dir.glob("*.csv"))
    
    print(f"   Found {len(pred_files)} prediction files")
    
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        
        # Parse filename for metadata
        metadata = parse_filename(pred_file.name)
        
        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value
        
        df['source_file'] = pred_file.name
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total predictions: {len(combined)}")
    
    return combined


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute error breakdown by experimental condition."""
    
    group_cols = ['tier', 'model_type', 'classifier', 'input_type', 
                  'names_normalized', 'pooling', 'dataset']
    
    results = []
    
    for name, group in df.groupby(group_cols):
        condition = dict(zip(group_cols, name))
        
        total = len(group)
        correct = int(group['correct'].sum())
        incorrect = total - correct
        
        # Confusion matrix components
        tp = int(((group['predicted_label'] == 1) & (group['true_label'] == 1)).sum())
        tn = int(((group['predicted_label'] == 0) & (group['true_label'] == 0)).sum())
        fp = int(((group['predicted_label'] == 1) & (group['true_label'] == 0)).sum())
        fn = int(((group['predicted_label'] == 0) & (group['true_label'] == 1)).sum())
        
        # Metrics
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Rates
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        result = {
            **condition,
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(accuracy, 4),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'fp_rate': round(fp_rate, 4),
            'fn_rate': round(fn_rate, 4),
        }
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_normalization_effect(error_breakdown: pd.DataFrame) -> pd.DataFrame:
    """Compare original vs normalized name performance."""
    
    results = []
    
    # Group by everything except names_normalized
    compare_cols = ['tier', 'model_type', 'classifier', 'input_type', 'pooling', 'dataset']
    
    for name, group in error_breakdown.groupby(compare_cols):
        if len(group) < 2:
            continue
        
        orig = group[group['names_normalized'] == False]
        norm = group[group['names_normalized'] == True]
        
        if len(orig) == 0 or len(norm) == 0:
            continue
        
        orig = orig.iloc[0]
        norm = norm.iloc[0]
        
        condition = dict(zip(compare_cols, name))
        
        result = {
            **condition,
            'accuracy_original': orig['accuracy'],
            'accuracy_normalized': norm['accuracy'],
            'accuracy_diff': round(norm['accuracy'] - orig['accuracy'], 4),
            'f1_original': orig['f1'],
            'f1_normalized': norm['f1'],
            'f1_diff': round(norm['f1'] - orig['f1'], 4),
            'fp_original': orig['fp'],
            'fp_normalized': norm['fp'],
            'fp_diff': int(norm['fp'] - orig['fp']),
            'fn_original': orig['fn'],
            'fn_normalized': norm['fn'],
            'fn_diff': int(norm['fn'] - orig['fn']),
        }
        results.append(result)
    
    return pd.DataFrame(results)


def analyze_input_type_effect(error_breakdown: pd.DataFrame) -> pd.DataFrame:
    """Compare text_only vs context_text vs context_only performance."""
    
    results = []
    
    # Group by everything except input_type
    compare_cols = ['tier', 'model_type', 'classifier', 'names_normalized', 'pooling', 'dataset']
    
    for name, group in error_breakdown.groupby(compare_cols):
        if len(group) < 2:
            continue
        
        condition = dict(zip(compare_cols, name))
        
        for input_type in ['text_only', 'context_text', 'context_only']:
            subset = group[group['input_type'] == input_type]
            if len(subset) > 0:
                row = subset.iloc[0]
                condition[f'accuracy_{input_type}'] = row['accuracy']
                condition[f'f1_{input_type}'] = row['f1']
        
        # Calculate differences if we have text_only and context_text
        if 'accuracy_text_only' in condition and 'accuracy_context_text' in condition:
            condition['context_effect'] = round(
                condition['accuracy_context_text'] - condition['accuracy_text_only'], 4
            )
        
        results.append(condition)
    
    return pd.DataFrame(results)


def find_flips_by_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Find examples that flip correctness when names are normalized."""
    
    flips = []
    
    # Group by conditions that should have matching examples
    group_cols = ['tier', 'model_type', 'classifier', 'input_type', 'pooling', 'dataset']
    
    for name, group in df.groupby(group_cols):
        orig = group[group['names_normalized'] == False]
        norm = group[group['names_normalized'] == True]
        
        if len(orig) == 0 or len(norm) == 0:
            continue
        
        # Match by example_id
        orig_ids = set(orig['example_id'].unique())
        norm_ids = set(norm['example_id'].unique())
        common_ids = orig_ids.intersection(norm_ids)
        
        for ex_id in common_ids:
            orig_row = orig[orig['example_id'] == ex_id].iloc[0]
            norm_row = norm[norm['example_id'] == ex_id].iloc[0]
            
            orig_correct = int(orig_row['correct'])
            norm_correct = int(norm_row['correct'])
            
            if orig_correct != norm_correct:
                flip = {
                    'example_id': ex_id,
                    'text': orig_row['text'],
                    'context': orig_row.get('context', ''),
                    'true_label': int(orig_row['true_label']),
                    'pred_original': int(orig_row['predicted_label']),
                    'pred_normalized': int(norm_row['predicted_label']),
                    'correct_original': orig_correct,
                    'correct_normalized': norm_correct,
                    'flip_direction': 'norm_helps' if norm_correct else 'norm_hurts',
                    **dict(zip(group_cols, name))
                }
                flips.append(flip)
    
    return pd.DataFrame(flips)


def find_flips_by_input_type(df: pd.DataFrame) -> pd.DataFrame:
    """Find examples that flip correctness between text_only and context_text."""
    
    flips = []
    
    # Group by conditions that should have matching examples
    group_cols = ['tier', 'model_type', 'classifier', 'names_normalized', 'pooling', 'dataset']
    
    for name, group in df.groupby(group_cols):
        text_only = group[group['input_type'] == 'text_only']
        context_text = group[group['input_type'] == 'context_text']
        
        if len(text_only) == 0 or len(context_text) == 0:
            continue
        
        # Match by example_id
        to_ids = set(text_only['example_id'].unique())
        ct_ids = set(context_text['example_id'].unique())
        common_ids = to_ids.intersection(ct_ids)
        
        for ex_id in common_ids:
            to_row = text_only[text_only['example_id'] == ex_id].iloc[0]
            ct_row = context_text[context_text['example_id'] == ex_id].iloc[0]
            
            to_correct = int(to_row['correct'])
            ct_correct = int(ct_row['correct'])
            
            if to_correct != ct_correct:
                flip = {
                    'example_id': ex_id,
                    'text': to_row['text'],
                    'context': to_row.get('context', ''),
                    'true_label': int(to_row['true_label']),
                    'pred_text_only': int(to_row['predicted_label']),
                    'pred_context_text': int(ct_row['predicted_label']),
                    'correct_text_only': to_correct,
                    'correct_context_text': ct_correct,
                    'flip_direction': 'context_helps' if ct_correct else 'context_hurts',
                    **dict(zip(group_cols, name))
                }
                flips.append(flip)
    
    return pd.DataFrame(flips)


def find_flips_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Find examples where different models disagree."""
    
    flips = []
    
    # Group by conditions (excluding model info)
    group_cols = ['input_type', 'names_normalized', 'pooling', 'dataset']
    
    for name, group in df.groupby(group_cols):
        # Get unique model combinations
        model_col = group['tier'] + '_' + group['model_type'] + '_' + group['classifier']
        models = model_col.unique()
        
        if len(models) < 2:
            continue
        
        # Get unique example IDs
        example_ids = group['example_id'].unique()
        
        for ex_id in example_ids:
            ex_data = group[group['example_id'] == ex_id]
            
            if len(ex_data) < 2:
                continue
            
            correct_vals = ex_data['correct'].unique()
            
            if len(correct_vals) > 1:  # Disagreement exists
                row = ex_data.iloc[0]
                
                models_correct = []
                models_incorrect = []
                
                for _, model_row in ex_data.iterrows():
                    model_name = f"{model_row['tier']}_{model_row['model_type']}_{model_row['classifier']}"
                    if model_row['correct']:
                        models_correct.append(model_name)
                    else:
                        models_incorrect.append(model_name)
                
                flip = {
                    'example_id': ex_id,
                    'text': row['text'],
                    'context': row.get('context', ''),
                    'true_label': int(row['true_label']),
                    'models_correct': ', '.join(sorted(set(models_correct))),
                    'models_incorrect': ', '.join(sorted(set(models_incorrect))),
                    'num_correct': len(set(models_correct)),
                    'num_incorrect': len(set(models_incorrect)),
                    **dict(zip(group_cols, name))
                }
                flips.append(flip)
    
    # Remove duplicates (same example can appear multiple times)
    if flips:
        flip_df = pd.DataFrame(flips)
        flip_df = flip_df.drop_duplicates(subset=['example_id', 'input_type', 'names_normalized', 'dataset'])
        return flip_df
    
    return pd.DataFrame(flips)


def generate_report(
    error_breakdown: pd.DataFrame,
    norm_effect: pd.DataFrame,
    input_effect: pd.DataFrame,
    norm_flips: pd.DataFrame,
    input_flips: pd.DataFrame,
    model_flips: pd.DataFrame
) -> str:
    """Generate human-readable report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("QUANTITATIVE ERROR ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # 1. Overall Statistics
    lines.append("\n" + "=" * 70)
    lines.append("1. OVERALL ERROR BREAKDOWN")
    lines.append("=" * 70)
    
    # Best performing conditions
    gold_results = error_breakdown[error_breakdown['dataset'] == 'gold'].copy()
    if len(gold_results) > 0:
        gold_results = gold_results.sort_values('accuracy', ascending=False)
        
        lines.append("\nTop 5 conditions on GOLD set (by accuracy):")
        for i, (_, row) in enumerate(gold_results.head(5).iterrows(), 1):
            model = f"{row['tier']}_{row['model_type']}_{row['classifier']}"
            pooling = f"_{row['pooling']}" if row['pooling'] != 'NA' else ""
            norm = "normalized" if row['names_normalized'] else "original"
            lines.append(f"  {i}. {model}{pooling}, {row['input_type']}, {norm}")
            lines.append(f"     Accuracy: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")
            lines.append(f"     TP={row['tp']}, TN={row['tn']}, FP={row['fp']}, FN={row['fn']}")
    
    test_results = error_breakdown[error_breakdown['dataset'] == 'test'].copy()
    if len(test_results) > 0:
        test_results = test_results.sort_values('accuracy', ascending=False)
        
        lines.append("\nTop 5 conditions on TEST set (by accuracy):")
        for i, (_, row) in enumerate(test_results.head(5).iterrows(), 1):
            model = f"{row['tier']}_{row['model_type']}_{row['classifier']}"
            pooling = f"_{row['pooling']}" if row['pooling'] != 'NA' else ""
            norm = "normalized" if row['names_normalized'] else "original"
            lines.append(f"  {i}. {model}{pooling}, {row['input_type']}, {norm}")
            lines.append(f"     Accuracy: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")
    
    # 2. Normalization Effect
    lines.append("\n" + "=" * 70)
    lines.append("2. NORMALIZATION EFFECT")
    lines.append("=" * 70)
    
    if len(norm_effect) > 0:
        avg_diff = norm_effect['accuracy_diff'].mean()
        helps = (norm_effect['accuracy_diff'] > 0).sum()
        hurts = (norm_effect['accuracy_diff'] < 0).sum()
        neutral = (norm_effect['accuracy_diff'] == 0).sum()
        
        lines.append(f"\nOverall: Normalization {'HELPS' if avg_diff > 0 else 'HURTS'} on average")
        lines.append(f"  Mean accuracy change: {avg_diff:+.4f}")
        lines.append(f"  Conditions where normalization helps: {helps}")
        lines.append(f"  Conditions where normalization hurts: {hurts}")
        lines.append(f"  Conditions with no change: {neutral}")
        
        # Show biggest effects
        biggest_help = norm_effect.nlargest(3, 'accuracy_diff')
        biggest_hurt = norm_effect.nsmallest(3, 'accuracy_diff')
        
        lines.append("\nBiggest positive effects (normalization helps):")
        for _, row in biggest_help.iterrows():
            if row['accuracy_diff'] > 0:
                lines.append(f"  {row['model_type']}_{row['classifier']}, {row['input_type']}, {row['dataset']}: {row['accuracy_diff']:+.4f}")
        
        lines.append("\nBiggest negative effects (normalization hurts):")
        for _, row in biggest_hurt.iterrows():
            if row['accuracy_diff'] < 0:
                lines.append(f"  {row['model_type']}_{row['classifier']}, {row['input_type']}, {row['dataset']}: {row['accuracy_diff']:+.4f}")
    
    # 3. Input Type Effect
    lines.append("\n" + "=" * 70)
    lines.append("3. INPUT TYPE EFFECT")
    lines.append("=" * 70)
    
    if len(input_effect) > 0 and 'context_effect' in input_effect.columns:
        valid_effects = input_effect.dropna(subset=['context_effect'])
        if len(valid_effects) > 0:
            avg_context_effect = valid_effects['context_effect'].mean()
            context_helps = (valid_effects['context_effect'] > 0).sum()
            context_hurts = (valid_effects['context_effect'] < 0).sum()
            
            lines.append(f"\nContext effect (context_text vs text_only):")
            lines.append(f"  Mean accuracy change: {avg_context_effect:+.4f}")
            lines.append(f"  Conditions where context helps: {context_helps}")
            lines.append(f"  Conditions where context hurts: {context_hurts}")
    
    # 4. Flip Analysis
    lines.append("\n" + "=" * 70)
    lines.append("4. FLIP ANALYSIS")
    lines.append("=" * 70)
    
    # Normalization flips
    if len(norm_flips) > 0:
        norm_helps_count = (norm_flips['flip_direction'] == 'norm_helps').sum()
        norm_hurts_count = (norm_flips['flip_direction'] == 'norm_hurts').sum()
        
        lines.append(f"\nNormalization flips: {len(norm_flips)} total")
        lines.append(f"  Normalization helps (wrong→correct): {norm_helps_count}")
        lines.append(f"  Normalization hurts (correct→wrong): {norm_hurts_count}")
        
        # Show examples where normalization hurts
        hurts_examples = norm_flips[norm_flips['flip_direction'] == 'norm_hurts']
        if len(hurts_examples) > 0:
            lines.append("\n  Sample examples where normalization HURTS:")
            for i, (_, row) in enumerate(hurts_examples.head(3).iterrows(), 1):
                text_short = row['text'][:80] + "..." if len(str(row['text'])) > 80 else row['text']
                lines.append(f"    {i}. \"{text_short}\"")
                lines.append(f"       True: {'sarcastic' if row['true_label'] == 1 else 'literal'}, Original pred: {row['pred_original']}, Normalized pred: {row['pred_normalized']}")
    
    # Input type flips
    if len(input_flips) > 0:
        context_helps_count = (input_flips['flip_direction'] == 'context_helps').sum()
        context_hurts_count = (input_flips['flip_direction'] == 'context_hurts').sum()
        
        lines.append(f"\nInput type flips: {len(input_flips)} total")
        lines.append(f"  Context helps (wrong→correct): {context_helps_count}")
        lines.append(f"  Context hurts (correct→wrong): {context_hurts_count}")
    
    # Model disagreements
    if len(model_flips) > 0:
        lines.append(f"\nModel disagreements: {len(model_flips)} unique examples")
    
    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 4: QUANTITATIVE ERROR ANALYSIS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = Path(PREDICTIONS_DIR)
    
    # -------------------------------------------------------------------------
    # 1. LOAD PREDICTIONS
    # -------------------------------------------------------------------------
    print("1. LOADING PREDICTIONS")
    print("-" * 40)
    
    df = load_all_predictions(predictions_dir)
    print()
    
    # -------------------------------------------------------------------------
    # 2. COMPUTE ERROR BREAKDOWN
    # -------------------------------------------------------------------------
    print("2. COMPUTING ERROR BREAKDOWN")
    print("-" * 40)
    
    error_breakdown = compute_error_breakdown(df)
    print(f"   Computed breakdown for {len(error_breakdown)} conditions")
    
    error_breakdown.to_csv(output_dir / "error_breakdown.csv", index=False)
    print(f"   Saved: error_breakdown.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 3. ANALYZE NORMALIZATION EFFECT
    # -------------------------------------------------------------------------
    print("3. ANALYZING NORMALIZATION EFFECT")
    print("-" * 40)
    
    norm_effect = analyze_normalization_effect(error_breakdown)
    print(f"   Compared {len(norm_effect)} condition pairs")
    
    if len(norm_effect) > 0:
        avg_diff = norm_effect['accuracy_diff'].mean()
        print(f"   Average accuracy change: {avg_diff:+.4f}")
    
    norm_effect.to_csv(output_dir / "normalization_effect.csv", index=False)
    print(f"   Saved: normalization_effect.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 4. ANALYZE INPUT TYPE EFFECT
    # -------------------------------------------------------------------------
    print("4. ANALYZING INPUT TYPE EFFECT")
    print("-" * 40)
    
    input_effect = analyze_input_type_effect(error_breakdown)
    print(f"   Compared {len(input_effect)} condition groups")
    
    input_effect.to_csv(output_dir / "input_type_effect.csv", index=False)
    print(f"   Saved: input_type_effect.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 5. FIND FLIPS
    # -------------------------------------------------------------------------
    print("5. FINDING PREDICTION FLIPS")
    print("-" * 40)
    
    # Normalization flips
    norm_flips = find_flips_by_normalization(df)
    print(f"   Normalization flips: {len(norm_flips)}")
    if len(norm_flips) > 0:
        helps = (norm_flips['flip_direction'] == 'norm_helps').sum()
        hurts = (norm_flips['flip_direction'] == 'norm_hurts').sum()
        print(f"      - Helps: {helps}, Hurts: {hurts}")
    norm_flips.to_csv(output_dir / "flips_by_normalization.csv", index=False)
    
    # Input type flips
    input_flips = find_flips_by_input_type(df)
    print(f"   Input type flips: {len(input_flips)}")
    if len(input_flips) > 0:
        helps = (input_flips['flip_direction'] == 'context_helps').sum()
        hurts = (input_flips['flip_direction'] == 'context_hurts').sum()
        print(f"      - Context helps: {helps}, Context hurts: {hurts}")
    input_flips.to_csv(output_dir / "flips_by_input_type.csv", index=False)
    
    # Model flips
    model_flips = find_flips_by_model(df)
    print(f"   Model disagreements: {len(model_flips)}")
    model_flips.to_csv(output_dir / "flips_by_model.csv", index=False)
    
    print(f"   Saved: flips_by_normalization.csv, flips_by_input_type.csv, flips_by_model.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 6. GENERATE REPORT
    # -------------------------------------------------------------------------
    print("6. GENERATING REPORT")
    print("-" * 40)
    
    report = generate_report(
        error_breakdown, norm_effect, input_effect,
        norm_flips, input_flips, model_flips
    )
    
    report_path = output_dir / "quantitative_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"   Saved: quantitative_report.txt")
    print()
    
    # Print report to console
    print("\n" + report)
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
