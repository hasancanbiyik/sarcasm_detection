import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Configuration
class Config:
    OLLAMA_URL = 'http://localhost:11434/api/generate'
    
    MODELS = [
        'gemma2:9b',
        'qwen2.5:14b',
        'phi3:medium'
    ]
    
    # Paths
    TRAIN_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/train.csv'
    VAL_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/val.csv'
    TEST_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/test.csv'
    GOLD_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/gold.csv'
    
    OUTPUT_DIR = 'outputs/llm_zeroshot_predictions'

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def query_ollama(model: str, prompt: str) -> str:
    """Send a prompt to Ollama and return the response."""
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0  # Deterministic output
        }
    }
    
    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['response'].strip().lower()
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return ""


def parse_response(response: str) -> int:
    """Parse model response to binary label (1=sarcastic, 0=not sarcastic)."""
    response = response.lower().strip()
    
    # Check for sarcastic indicators
    if 'not sarcastic' in response or 'not_sarcastic' in response:
        return 0
    elif 'sarcastic' in response:
        return 1
    elif response in ['0', 'no', 'false', 'literal', 'non-sarcastic', 'nonsarcastic']:
        return 0
    elif response in ['1', 'yes', 'true']:
        return 1
    else:
        # Default to -1 for unparseable responses
        return -1


def create_prompt_with_context(context: str, text: str) -> str:
    """Create prompt with context."""
    return f"""Classify the following response as either "sarcastic" or "not sarcastic".

Context: {context}

Response to classify: {text}

Answer with only "sarcastic" or "not sarcastic"."""


def create_prompt_without_context(text: str) -> str:
    """Create prompt without context."""
    return f"""Classify the following sentence as either "sarcastic" or "not sarcastic".

Sentence: {text}

Answer with only "sarcastic" or "not sarcastic"."""


def run_classification(model: str, dataframe: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Run classification on dataframe with specified condition."""
    predictions = []
    raw_responses = []
    
    total = len(dataframe)
    
    for idx, row in dataframe.iterrows():
        if condition == 'context':
            prompt = create_prompt_with_context(row['context'], row['text'])
        else:
            prompt = create_prompt_without_context(row['text'])
        
        response = query_ollama(model, prompt)
        pred = parse_response(response)
        
        predictions.append(pred)
        raw_responses.append(response)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total}")
    
    # Create results dataframe
    results_df = dataframe.copy()
    results_df['predicted_label'] = predictions
    results_df['true_label'] = results_df['label']
    results_df['raw_response'] = raw_responses
    results_df['correct'] = (results_df['true_label'] == results_df['predicted_label']).astype(int)
    results_df['model_name'] = model.replace(':', '_')
    results_df['condition'] = 'context_text_original' if condition == 'context' else 'text_only'
    
    # Add example_id if not present
    if 'example_id' not in results_df.columns:
        results_df.insert(0, 'example_id', range(len(results_df)))
    
    return results_df


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """Calculate metrics, excluding unparseable responses."""
    # Filter out unparseable responses
    valid_df = results_df[results_df['predicted_label'] != -1]
    
    if len(valid_df) == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'total': len(results_df),
            'valid': 0,
            'unparseable': len(results_df)
        }
    
    true_labels = valid_df['true_label'].values
    predictions = valid_df['predicted_label'].values
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': cm[1][1] if cm.shape == (2, 2) else 0,
        'tn': cm[0][0] if cm.shape == (2, 2) else 0,
        'fp': cm[0][1] if cm.shape == (2, 2) else 0,
        'fn': cm[1][0] if cm.shape == (2, 2) else 0,
        'total': len(results_df),
        'valid': len(valid_df),
        'unparseable': len(results_df) - len(valid_df)
    }


def print_results(metrics: dict, model: str, dataset: str, condition: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Model: {model} | Dataset: {dataset.upper()} | Condition: {condition}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              0    1")
    print(f"Actual 0    {metrics['tn']:4d} {metrics['fp']:4d}")
    print(f"       1    {metrics['fn']:4d} {metrics['tp']:4d}")
    print(f"\nTotal: {metrics['total']} | Valid: {metrics['valid']} | Unparseable: {metrics['unparseable']}")


def main():
    print("="*60)
    print("LLM Zero-Shot Sarcasm Classification")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(Config.MODELS)}")
    print(f"Conditions: with context, without context\n")
    
    # Load datasets
    print("Loading datasets...")
    datasets = {
        'train': pd.read_csv(Config.TRAIN_PATH),
        'val': pd.read_csv(Config.VAL_PATH),
        'test': pd.read_csv(Config.TEST_PATH),
        'gold': pd.read_csv(Config.GOLD_PATH)
    }
    
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} examples")
    
    # Store all summary results
    all_summary_results = []
    
    # Run experiments
    for model in Config.MODELS:
        print(f"\n{'#'*60}")
        print(f"Running model: {model}")
        print(f"{'#'*60}")
        
        for condition in ['context', 'no_context']:
            print(f"\n--- Condition: {condition} ---")
            
            for dataset_name, df in datasets.items():
                print(f"\nProcessing {dataset_name} set...")
                
                # Run classification
                results_df = run_classification(model, df, condition)
                
                # Calculate metrics
                metrics = calculate_metrics(results_df)
                
                # Print results
                print_results(metrics, model, dataset_name, condition)
                
                # Save predictions
                model_safe = model.replace(':', '_')
                output_path = os.path.join(
                    Config.OUTPUT_DIR, 
                    f'predictions_{model_safe}_{condition}_{dataset_name}.csv'
                )
                results_df.to_csv(output_path, index=False)
                print(f"Saved to: {output_path}")
                
                # Add to summary
                all_summary_results.append({
                    'model': model,
                    'condition': 'context_text_original' if condition == 'context' else 'text_only',
                    'dataset': dataset_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'tp': metrics['tp'],
                    'tn': metrics['tn'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'total': metrics['total'],
                    'valid': metrics['valid'],
                    'unparseable': metrics['unparseable']
                })
    
    # Save summary
    summary_df = pd.DataFrame(all_summary_results)
    summary_path = os.path.join(Config.OUTPUT_DIR, 'llm_zeroshot_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
