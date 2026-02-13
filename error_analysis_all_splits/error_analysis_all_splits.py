import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
class Config:
    OUTPUT_DIR = 'outputs/error_analysis_all_splits'
    
    # ===========================================
    # PREDICTION FILES - ALL SPLITS
    # ===========================================
    
    # Define splits to analyze
    SPLITS = ['train', 'val', 'test', 'gold']
    
    # Base paths for each model type
    BASELINE_BASE = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions'
    XLMR_FINETUNED_BASE = '/Users/hasancan/Desktop/sarcasm_detection/experiment_1/outputs/finetuned_predictions'
    XLMR_FINETUNED_NO_CONTEXT_BASE = '/Users/hasancan/Desktop/sarcasm_detection/experiment_1/experiment_1.5/outputs/finetuned_predictions_no_context'
    DEBERTA_FINETUNED_BASE = '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta'
    DEBERTA_FINETUNED_NO_CONTEXT_BASE = '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta_no_context'
    LLM_BASE = '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions'
    
    # Model configurations: (model_name, base_path, filename_pattern)
    # {split} will be replaced with train/val/test/gold
    MODELS = {
        # Baselines - TF-IDF + LogReg
        'tfidf_logreg_context_original': (BASELINE_BASE, 'tier2_tfidf_logreg_context_text_original_{split}.csv'),
        'tfidf_logreg_context_normalized': (BASELINE_BASE, 'tier2_tfidf_logreg_context_text_normalized_{split}.csv'),
        'tfidf_logreg_text_original': (BASELINE_BASE, 'tier2_tfidf_logreg_text_only_original_{split}.csv'),
        'tfidf_logreg_text_normalized': (BASELINE_BASE, 'tier2_tfidf_logreg_text_only_normalized_{split}.csv'),
        
        # Baselines - TF-IDF + SVM
        'tfidf_svm_context_original': (BASELINE_BASE, 'tier2_tfidf_svm_context_text_original_{split}.csv'),
        'tfidf_svm_context_normalized': (BASELINE_BASE, 'tier2_tfidf_svm_context_text_normalized_{split}.csv'),
        'tfidf_svm_text_original': (BASELINE_BASE, 'tier2_tfidf_svm_text_only_original_{split}.csv'),
        'tfidf_svm_text_normalized': (BASELINE_BASE, 'tier2_tfidf_svm_text_only_normalized_{split}.csv'),
        
        # Fine-tuned XLM-RoBERTa
        'xlmr_finetuned_context': (XLMR_FINETUNED_BASE, 'predictions_{split}.csv'),
        'xlmr_finetuned_no_context': (XLMR_FINETUNED_NO_CONTEXT_BASE, 'predictions_{split}.csv'),
        
        # Fine-tuned DeBERTa
        'deberta_finetuned_context': (DEBERTA_FINETUNED_BASE, 'predictions_{split}.csv'),
        'deberta_finetuned_no_context': (DEBERTA_FINETUNED_NO_CONTEXT_BASE, 'predictions_{split}.csv'),
        
        # LLM Zero-shot
        'gemma2_9b_context': (LLM_BASE, 'predictions_gemma2_9b_context_{split}.csv'),
        'gemma2_9b_no_context': (LLM_BASE, 'predictions_gemma2_9b_no_context_{split}.csv'),
        'qwen2.5_14b_context': (LLM_BASE, 'predictions_qwen2.5_14b_context_{split}.csv'),
        'qwen2.5_14b_no_context': (LLM_BASE, 'predictions_qwen2.5_14b_no_context_{split}.csv'),
        'phi3_medium_context': (LLM_BASE, 'predictions_phi3_medium_context_{split}.csv'),
        'phi3_medium_no_context': (LLM_BASE, 'predictions_phi3_medium_no_context_{split}.csv'),
    }


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def get_file_path(model_name: str, split: str) -> str:
    """Get the file path for a model and split."""
    base_path, filename_pattern = Config.MODELS[model_name]
    filename = filename_pattern.format(split=split)
    return os.path.join(base_path, filename)


def analyze_split(split: str, loaded_models: list, all_predictions: dict):
    """Analyze a single split and return results dataframe."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {split.upper()} split")
    print(f"{'='*60}")
    
    # Load first available model to get base data
    base_df = None
    first_model = None
    
    for model_name in loaded_models:
        file_path = get_file_path(model_name, split)
        if os.path.exists(file_path):
            base_df = pd.read_csv(file_path)
            first_model = model_name
            break
    
    if base_df is None:
        print(f"  ERROR: No prediction files found for {split}")
        return None
    
    print(f"  Base data from: {first_model}")
    
    # Determine true label column
    if 'true_label' in base_df.columns:
        true_label_col = 'true_label'
    elif 'label' in base_df.columns:
        true_label_col = 'label'
    else:
        raise ValueError(f"Cannot find true label column. Columns: {base_df.columns.tolist()}")
    
    # Create master dataframe
    if 'example_id' not in base_df.columns:
        base_df['example_id'] = range(len(base_df))
    
    master_df = base_df[['example_id']].copy()
    
    if 'text' in base_df.columns:
        master_df['text'] = base_df['text']
    if 'context' in base_df.columns:
        master_df['context'] = base_df['context']
    
    master_df['true_label'] = base_df[true_label_col]
    master_df['split'] = split
    
    num_examples = len(master_df)
    print(f"  Number of examples: {num_examples}")
    
    # Load predictions from all models
    split_loaded_models = []
    
    for model_name in loaded_models:
        file_path = get_file_path(model_name, split)
        
        try:
            df = pd.read_csv(file_path)
            
            # Get predicted label column
            if 'predicted_label' in df.columns:
                pred_col = 'predicted_label'
            elif 'prediction' in df.columns:
                pred_col = 'prediction'
            else:
                continue
            
            # Verify example count
            if len(df) != num_examples:
                raise ValueError(f"Example count mismatch: expected {num_examples}, got {len(df)}")
            
            master_df[f'pred_{model_name}'] = df[pred_col].values
            split_loaded_models.append(model_name)
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
    
    print(f"  Loaded {len(split_loaded_models)} models")
    
    if len(split_loaded_models) == 0:
        return None
    
    # Calculate correctness
    correct_columns = []
    for model_name in split_loaded_models:
        pred_col = f'pred_{model_name}'
        correct_col = f'correct_{model_name}'
        master_df[correct_col] = (master_df['true_label'] == master_df[pred_col]).astype(int)
        correct_columns.append(correct_col)
    
    # Calculate aggregates
    master_df['num_models_correct'] = master_df[correct_columns].sum(axis=1)
    master_df['num_models_wrong'] = len(split_loaded_models) - master_df['num_models_correct']
    master_df['total_models'] = len(split_loaded_models)
    
    # Error types
    for model_name in split_loaded_models:
        pred_col = f'pred_{model_name}'
        correct_col = f'correct_{model_name}'
        error_type_col = f'error_type_{model_name}'
        
        conditions = [
            master_df[correct_col] == 1,
            (master_df['true_label'] == 1) & (master_df[pred_col] == 0),
            (master_df['true_label'] == 0) & (master_df[pred_col] == 1),
        ]
        choices = ['correct', 'FN', 'FP']
        master_df[error_type_col] = np.select(conditions, choices, default='unknown')
    
    error_type_columns = [f'error_type_{model_name}' for model_name in split_loaded_models]
    master_df['num_FN'] = master_df[error_type_columns].apply(lambda row: (row == 'FN').sum(), axis=1)
    master_df['num_FP'] = master_df[error_type_columns].apply(lambda row: (row == 'FP').sum(), axis=1)
    
    # Models wrong/correct lists
    def get_wrong_models(row):
        wrong = [m for m in split_loaded_models if row[f'correct_{m}'] == 0]
        return ', '.join(wrong) if wrong else 'None'
    
    def get_correct_models(row):
        correct = [m for m in split_loaded_models if row[f'correct_{m}'] == 1]
        return ', '.join(correct) if correct else 'None'
    
    master_df['models_wrong'] = master_df.apply(get_wrong_models, axis=1)
    master_df['models_correct'] = master_df.apply(get_correct_models, axis=1)
    
    return master_df, split_loaded_models


def main():
    print("="*60)
    print("Error Analysis: All Splits")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Splits to analyze: {', '.join(Config.SPLITS)}\n")
    
    # Get list of all model names
    all_model_names = list(Config.MODELS.keys())
    print(f"Total models configured: {len(all_model_names)}")
    
    # Check which models have files available
    available_models = []
    for model_name in all_model_names:
        # Check if at least one split exists
        for split in Config.SPLITS:
            file_path = get_file_path(model_name, split)
            if os.path.exists(file_path):
                available_models.append(model_name)
                break
    
    print(f"Models with available files: {len(available_models)}\n")
    
    # Analyze each split
    all_results = []
    split_summaries = []
    
    for split in Config.SPLITS:
        result = analyze_split(split, available_models, Config.MODELS)
        
        if result is not None:
            master_df, split_loaded_models = result
            all_results.append(master_df)
            
            # Calculate split summary
            summary = {
                'split': split,
                'num_examples': len(master_df),
                'num_models': len(split_loaded_models),
                'all_correct_count': (master_df['num_models_wrong'] == 0).sum(),
                'all_wrong_count': (master_df['num_models_correct'] == 0).sum(),
                'mean_models_wrong': master_df['num_models_wrong'].mean(),
                'median_models_wrong': master_df['num_models_wrong'].median(),
            }
            split_summaries.append(summary)
            
            # Prepare output columns
            output_columns = ['example_id', 'split', 'text', 'context', 'true_label',
                            'num_models_correct', 'num_models_wrong', 'total_models',
                            'num_FN', 'num_FP', 'models_wrong', 'models_correct']
            
            pred_columns = [f'pred_{m}' for m in split_loaded_models]
            output_columns.extend(pred_columns)
            output_columns = [c for c in output_columns if c in master_df.columns]
            
            # Sort by difficulty
            df_difficulty = master_df[output_columns].sort_values(
                by=['num_models_wrong', 'num_FN'],
                ascending=[False, False]
            )
            
            # Sort by easiness
            df_easiness = master_df[output_columns].sort_values(
                by=['num_models_correct'],
                ascending=[False]
            )
            
            # Save split-specific files
            difficulty_path = os.path.join(Config.OUTPUT_DIR, f'{split}_sorted_by_difficulty.csv')
            easiness_path = os.path.join(Config.OUTPUT_DIR, f'{split}_sorted_by_easiness.csv')
            
            df_difficulty.to_csv(difficulty_path, index=False)
            df_easiness.to_csv(easiness_path, index=False)
            
            print(f"  Saved: {difficulty_path}")
            print(f"  Saved: {easiness_path}")
    
    # Combine all splits
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Sort combined by difficulty
        output_columns = ['example_id', 'split', 'text', 'context', 'true_label',
                        'num_models_correct', 'num_models_wrong', 'total_models',
                        'num_FN', 'num_FP', 'models_wrong', 'models_correct']
        output_columns = [c for c in output_columns if c in combined_df.columns]
        
        combined_difficulty = combined_df[output_columns].sort_values(
            by=['num_models_wrong', 'num_FN'],
            ascending=[False, False]
        )
        
        combined_easiness = combined_df[output_columns].sort_values(
            by=['num_models_correct'],
            ascending=[False]
        )
        
        # Save combined files
        combined_difficulty_path = os.path.join(Config.OUTPUT_DIR, 'all_splits_sorted_by_difficulty.csv')
        combined_easiness_path = os.path.join(Config.OUTPUT_DIR, 'all_splits_sorted_by_easiness.csv')
        
        combined_difficulty.to_csv(combined_difficulty_path, index=False)
        combined_easiness.to_csv(combined_easiness_path, index=False)
        
        print(f"\nSaved combined: {combined_difficulty_path}")
        print(f"Saved combined: {combined_easiness_path}")
    
    # Save and print summary
    summary_df = pd.DataFrame(split_summaries)
    summary_path = os.path.join(Config.OUTPUT_DIR, 'split_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY BY SPLIT")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    
    # Print hardest examples across all splits
    if all_results:
        print("\n" + "="*60)
        print("TOP 10 HARDEST EXAMPLES (ACROSS ALL SPLITS)")
        print("="*60)
        
        for idx, row in combined_difficulty.head(10).iterrows():
            label_str = "SARCASTIC" if row['true_label'] == 1 else "LITERAL"
            print(f"\n[{row['num_models_wrong']}/{row['total_models']} wrong] ({row['split'].upper()}) ({label_str})")
            text_preview = str(row['text'])[:100] + "..." if len(str(row['text'])) > 100 else row['text']
            print(f"  Text: {text_preview}")
            print(f"  FN: {row['num_FN']}, FP: {row['num_FP']}")
        
        # Print easiest examples
        print("\n" + "="*60)
        print("EXAMPLES ALL MODELS GOT CORRECT (ACROSS ALL SPLITS)")
        print("="*60)
        
        all_correct = combined_df[combined_df['num_models_wrong'] == 0]
        print(f"Total: {len(all_correct)} examples")
        
        for split in Config.SPLITS:
            split_correct = all_correct[all_correct['split'] == split]
            print(f"  {split}: {len(split_correct)} examples")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
