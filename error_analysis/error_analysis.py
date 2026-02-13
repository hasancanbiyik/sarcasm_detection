import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
class Config:
    OUTPUT_DIR = 'outputs/error_analysis'
    
    # ===========================================
    # PREDICTION FILES - GOLD SET ONLY
    # ===========================================
    
    # Baselines (Tier 2: TF-IDF)
    BASELINE_PREDICTIONS = {
        # TF-IDF + LogReg
        'tfidf_logreg_context_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_logreg_context_text_original_gold.csv',
        'tfidf_logreg_context_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_logreg_context_text_normalized_gold.csv',
        'tfidf_logreg_text_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_logreg_text_only_original_gold.csv',
        'tfidf_logreg_text_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_logreg_text_only_normalized_gold.csv',
        
        # TF-IDF + SVM
        'tfidf_svm_context_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_svm_context_text_original_gold.csv',
        'tfidf_svm_context_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_svm_context_text_normalized_gold.csv',
        'tfidf_svm_text_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_svm_text_only_original_gold.csv',
        'tfidf_svm_text_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier2_tfidf_svm_text_only_normalized_gold.csv',
        
        # XLM-R Frozen + LogReg (Tier 3)
        'xlmr_frozen_cls_context_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_cls_context_text_original_gold.csv',
        'xlmr_frozen_cls_context_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_cls_context_text_normalized_gold.csv',
        'xlmr_frozen_cls_text_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_cls_text_only_original_gold.csv',
        'xlmr_frozen_cls_text_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_cls_text_only_normalized_gold.csv',
        'xlmr_frozen_mean_context_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_mean_context_text_original_gold.csv',
        'xlmr_frozen_mean_context_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_mean_context_text_normalized_gold.csv',
        'xlmr_frozen_mean_text_original': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_mean_text_only_original_gold.csv',
        'xlmr_frozen_mean_text_normalized': '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions/tier3_xlmr_logreg_mean_text_only_normalized_gold.csv',
    }
    
    # Fine-tuned models
    FINETUNED_PREDICTIONS = {
        # XLM-RoBERTa fine-tuned
        'xlmr_finetuned_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_1/outputs/finetuned_predictions/predictions_gold.csv',
        'xlmr_finetuned_no_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_1/experiment_1.5/outputs/finetuned_predictions_no_context/predictions_gold.csv',
        
        # DeBERTa fine-tuned
        'deberta_finetuned_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta/predictions_gold.csv',
        'deberta_finetuned_no_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta_no_context/predictions_gold.csv',
    }
    
    # LLM Zero-shot predictions
    LLM_PREDICTIONS = {
        # Gemma2:9b
        'gemma2_9b_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_gemma2_9b_context_gold.csv',
        'gemma2_9b_no_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_gemma2_9b_no_context_gold.csv',
        
        # Qwen2.5:14b
        'qwen2.5_14b_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_qwen2.5_14b_context_gold.csv',
        'qwen2.5_14b_no_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_qwen2.5_14b_no_context_gold.csv',
        
        # Phi3:medium
        'phi3_medium_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_phi3_medium_context_gold.csv',
        'phi3_medium_no_context': '/Users/hasancan/Desktop/sarcasm_detection/experiment_3/outputs/llm_zeroshot_predictions/predictions_phi3_medium_no_context_gold.csv',
    }


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def load_predictions(file_path: str, model_name: str) -> pd.DataFrame:
    """Load a prediction file and extract relevant columns."""
    df = pd.read_csv(file_path)
    
    # Handle different column names for true label
    if 'true_label' in df.columns:
        true_label_col = 'true_label'
    elif 'label' in df.columns:
        true_label_col = 'label'
    else:
        raise ValueError(f"Cannot find true label column in {file_path}. Columns: {df.columns.tolist()}")
    
    # Handle different column names for predicted label
    if 'predicted_label' in df.columns:
        pred_label_col = 'predicted_label'
    elif 'prediction' in df.columns:
        pred_label_col = 'prediction'
    else:
        raise ValueError(f"Cannot find predicted label column in {file_path}. Columns: {df.columns.tolist()}")
    
    # Create standardized dataframe
    result = pd.DataFrame({
        'example_id': df['example_id'] if 'example_id' in df.columns else range(len(df)),
        'true_label': df[true_label_col],
        f'pred_{model_name}': df[pred_label_col]
    })
    
    return result


def main():
    print("="*60)
    print("Error Analysis: Common Mistakes Across Models")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Combine all prediction sources
    all_predictions = {}
    all_predictions.update(Config.BASELINE_PREDICTIONS)
    all_predictions.update(Config.FINETUNED_PREDICTIONS)
    all_predictions.update(Config.LLM_PREDICTIONS)
    
    print(f"Total models to load: {len(all_predictions)}\n")
    
    # Load first file to get base data (text, context, true labels)
    first_model = list(all_predictions.keys())[0]
    first_path = all_predictions[first_model]
    
    print(f"Loading base data from: {first_path}")
    base_df = pd.read_csv(first_path)
    
    # Determine true label column
    if 'true_label' in base_df.columns:
        true_label_col = 'true_label'
    elif 'label' in base_df.columns:
        true_label_col = 'label'
    else:
        raise ValueError(f"Cannot find true label column. Columns: {base_df.columns.tolist()}")
    
    # Create master dataframe with example info
    if 'example_id' not in base_df.columns:
        base_df['example_id'] = range(len(base_df))
    
    master_df = base_df[['example_id']].copy()
    
    # Add text and context if available
    if 'text' in base_df.columns:
        master_df['text'] = base_df['text']
    if 'context' in base_df.columns:
        master_df['context'] = base_df['context']
    
    master_df['true_label'] = base_df[true_label_col]
    
    num_examples = len(master_df)
    print(f"Number of examples in gold set: {num_examples}\n")
    
    # Load predictions from all models
    print("Loading predictions from all models...")
    loaded_models = []
    
    for model_name, file_path in all_predictions.items():
        try:
            df = pd.read_csv(file_path)
            
            # Get predicted label column
            if 'predicted_label' in df.columns:
                pred_col = 'predicted_label'
            elif 'prediction' in df.columns:
                pred_col = 'prediction'
            else:
                print(f"  WARNING: Cannot find prediction column in {model_name}, skipping...")
                continue
            
            # Verify example count matches
            if len(df) != num_examples:
                raise ValueError(f"Example count mismatch for {model_name}: expected {num_examples}, got {len(df)}")
            
            # Add predictions to master dataframe
            master_df[f'pred_{model_name}'] = df[pred_col].values
            loaded_models.append(model_name)
            print(f"  Loaded: {model_name}")
            
        except FileNotFoundError:
            print(f"  WARNING: File not found for {model_name}: {file_path}")
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
    
    print(f"\nSuccessfully loaded {len(loaded_models)} models")
    
    # Calculate correctness for each model
    print("\nCalculating correctness for each model...")
    
    correct_columns = []
    for model_name in loaded_models:
        pred_col = f'pred_{model_name}'
        correct_col = f'correct_{model_name}'
        master_df[correct_col] = (master_df['true_label'] == master_df[pred_col]).astype(int)
        correct_columns.append(correct_col)
    
    # Calculate number of models that got each example correct/wrong
    master_df['num_models_correct'] = master_df[correct_columns].sum(axis=1)
    master_df['num_models_wrong'] = len(loaded_models) - master_df['num_models_correct']
    
    # Calculate error type for each example (FP, FN, or correct)
    # FN = true_label is 1 (sarcastic) but predicted 0
    # FP = true_label is 0 (literal) but predicted 1
    
    # For each model, determine if it made FP or FN
    for model_name in loaded_models:
        pred_col = f'pred_{model_name}'
        correct_col = f'correct_{model_name}'
        error_type_col = f'error_type_{model_name}'
        
        conditions = [
            master_df[correct_col] == 1,  # Correct
            (master_df['true_label'] == 1) & (master_df[pred_col] == 0),  # FN
            (master_df['true_label'] == 0) & (master_df[pred_col] == 1),  # FP
        ]
        choices = ['correct', 'FN', 'FP']
        master_df[error_type_col] = np.select(conditions, choices, default='unknown')
    
    # Count FP and FN across models for each example
    error_type_columns = [f'error_type_{model_name}' for model_name in loaded_models]
    
    master_df['num_FN'] = master_df[error_type_columns].apply(lambda row: (row == 'FN').sum(), axis=1)
    master_df['num_FP'] = master_df[error_type_columns].apply(lambda row: (row == 'FP').sum(), axis=1)
    
    # Create list of which models got it wrong
    def get_wrong_models(row):
        wrong = []
        for model_name in loaded_models:
            if row[f'correct_{model_name}'] == 0:
                wrong.append(model_name)
        return ', '.join(wrong) if wrong else 'None'
    
    def get_correct_models(row):
        correct = []
        for model_name in loaded_models:
            if row[f'correct_{model_name}'] == 1:
                correct.append(model_name)
        return ', '.join(correct) if correct else 'None'
    
    master_df['models_wrong'] = master_df.apply(get_wrong_models, axis=1)
    master_df['models_correct'] = master_df.apply(get_correct_models, axis=1)
    
    # Prepare output dataframes
    # Columns to include in output
    output_columns = ['example_id', 'text', 'context', 'true_label', 
                      'num_models_correct', 'num_models_wrong', 'num_FN', 'num_FP',
                      'models_wrong', 'models_correct']
    
    # Add prediction columns
    pred_columns = [f'pred_{model_name}' for model_name in loaded_models]
    output_columns.extend(pred_columns)
    
    # Filter to only include columns that exist
    output_columns = [col for col in output_columns if col in master_df.columns]
    
    # Sort by difficulty (most models wrong first)
    df_sorted_by_difficulty = master_df[output_columns].sort_values(
        by=['num_models_wrong', 'num_FN'], 
        ascending=[False, False]
    )
    
    # Sort by easiness (most models correct first)
    df_sorted_by_easiness = master_df[output_columns].sort_values(
        by=['num_models_correct', 'num_FP'], 
        ascending=[False, False]
    )
    
    # Save outputs
    difficulty_path = os.path.join(Config.OUTPUT_DIR, 'examples_sorted_by_difficulty.csv')
    easiness_path = os.path.join(Config.OUTPUT_DIR, 'examples_sorted_by_easiness.csv')
    
    df_sorted_by_difficulty.to_csv(difficulty_path, index=False)
    df_sorted_by_easiness.to_csv(easiness_path, index=False)
    
    print(f"\nSaved: {difficulty_path}")
    print(f"Saved: {easiness_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal examples: {num_examples}")
    print(f"Total models: {len(loaded_models)}")
    
    # Hardest examples (most models wrong)
    print("\n--- HARDEST EXAMPLES (most models got wrong) ---")
    hardest = df_sorted_by_difficulty.head(10)
    for idx, row in hardest.iterrows():
        label_str = "SARCASTIC" if row['true_label'] == 1 else "LITERAL"
        print(f"\n[{row['num_models_wrong']}/{len(loaded_models)} wrong] ({label_str})")
        print(f"  Text: {row['text'][:100]}..." if len(str(row['text'])) > 100 else f"  Text: {row['text']}")
        print(f"  FN: {row['num_FN']}, FP: {row['num_FP']}")
    
    # Easiest examples (all models correct)
    all_correct = master_df[master_df['num_models_wrong'] == 0]
    print(f"\n--- EASIEST EXAMPLES (all {len(loaded_models)} models correct) ---")
    print(f"Count: {len(all_correct)}")
    
    if len(all_correct) > 0:
        for idx, row in all_correct.head(5).iterrows():
            label_str = "SARCASTIC" if row['true_label'] == 1 else "LITERAL"
            print(f"\n  ({label_str}) {row['text'][:100]}..." if len(str(row['text'])) > 100 else f"\n  ({label_str}) {row['text']}")
    
    # Examples all models got wrong
    all_wrong = master_df[master_df['num_models_correct'] == 0]
    print(f"\n--- EXAMPLES ALL MODELS GOT WRONG ---")
    print(f"Count: {len(all_wrong)}")
    
    if len(all_wrong) > 0:
        for idx, row in all_wrong.iterrows():
            label_str = "SARCASTIC" if row['true_label'] == 1 else "LITERAL"
            print(f"\n  ({label_str}) {row['text'][:100]}..." if len(str(row['text'])) > 100 else f"\n  ({label_str}) {row['text']}")
    
    # Distribution of difficulty
    print("\n--- DIFFICULTY DISTRIBUTION ---")
    difficulty_dist = master_df['num_models_wrong'].value_counts().sort_index()
    for num_wrong, count in difficulty_dist.items():
        pct = count / num_examples * 100
        print(f"  {num_wrong} models wrong: {count} examples ({pct:.1f}%)")
    
    # Model performance summary
    print("\n--- MODEL ACCURACY ON GOLD SET ---")
    model_accuracies = []
    for model_name in loaded_models:
        correct_col = f'correct_{model_name}'
        accuracy = master_df[correct_col].mean()
        model_accuracies.append((model_name, accuracy))
    
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    for model_name, accuracy in model_accuracies:
        print(f"  {model_name}: {accuracy:.3f}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
