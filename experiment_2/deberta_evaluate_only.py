import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import argparse
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    
    # Data paths
    VAL_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/val.csv'
    TEST_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/test.csv'
    GOLD_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/gold.csv'
    
    # Model configurations
    MODELS = {
        'context': {
            'checkpoint_dir': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_deberta',
            'predictions_dir': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta',
            'use_context': True,
            'condition_name': 'context_text_original'
        },
        'no_context': {
            'checkpoint_dir': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_deberta_no_context',
            'predictions_dir': '/Users/hasancan/Desktop/sarcasm_detection/experiment_2/outputs/finetuned_predictions_deberta_no_context',
            'use_context': False,
            'condition_name': 'text_only'
        }
    }


def find_best_checkpoint(checkpoint_dir):
    """Find the best/latest checkpoint in the directory"""
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by step number and get the latest
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    best_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    return best_checkpoint


# Custom Dataset Class - handles both context and no-context
class SarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, use_context=True):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Choose input format based on configuration
        if self.use_context:
            text_input = f"Context: {row['context']} Response: {row['text']}"
        else:
            text_input = row['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text_input,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }


def evaluate_and_save_predictions(model, tokenizer, dataframe, dataset_name, output_dir, use_context, condition_name):
    """Evaluate model and save predictions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    confidences = []
    
    # Create dataset with appropriate context setting
    dataset = SarcasmDataset(dataframe, tokenizer, Config.MAX_LENGTH, use_context=use_context)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(probs[:, 1].cpu().numpy())
    
    # Create results dataframe
    results_df = dataframe.copy()
    results_df['true_label'] = true_labels
    results_df['predicted_label'] = predictions
    results_df['correct'] = (np.array(true_labels) == np.array(predictions)).astype(int)
    results_df['confidence'] = confidences
    results_df['model_name'] = 'deberta_v3_finetuned'
    results_df['condition'] = condition_name
    
    # Add example_id if not present
    if 'example_id' not in results_df.columns:
        results_df.insert(0, 'example_id', range(len(results_df)))
    
    # Save predictions
    output_path = os.path.join(output_dir, f'predictions_{dataset_name}.csv')
    results_df.to_csv(output_path, index=False)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    cm = confusion_matrix(true_labels, predictions)
    
    # Print results
    context_str = "WITH CONTEXT" if use_context else "NO CONTEXT (TEXT ONLY)"
    print(f"\n{'='*60}")
    print(f"Results on {dataset_name.upper()} set ({context_str})")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              0    1")
    print(f"Actual 0    {cm[0][0]:4d} {cm[0][1]:4d}")
    print(f"       1    {cm[1][0]:4d} {cm[1][1]:4d}")
    print(f"\nTP: {cm[1][1]}, TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}")
    print(f"Predictions saved to: {output_path}")
    
    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': cm[1][1],
        'tn': cm[0][0],
        'fp': cm[0][1],
        'fn': cm[1][0]
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned DeBERTa model')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['context', 'no_context', 'both'],
        default='context',
        help='Which model to evaluate: "context", "no_context", or "both"'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Specific checkpoint path (optional, will auto-detect if not provided)'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("DeBERTa-v3 Evaluation (Flexible Mode)")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}\n")
    
    # Load data
    print("Loading datasets...")
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    gold_df = pd.read_csv(Config.GOLD_PATH)
    
    print(f"Val:   {len(val_df)} examples")
    print(f"Test:  {len(test_df)} examples")
    print(f"Gold:  {len(gold_df)} examples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {Config.MODEL_NAME}...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Determine which models to evaluate
    modes_to_run = ['context', 'no_context'] if args.mode == 'both' else [args.mode]
    
    all_summaries = []
    
    for mode in modes_to_run:
        model_config = Config.MODELS[mode]
        
        print("\n" + "="*60)
        print(f"EVALUATING: {'WITH CONTEXT' if model_config['use_context'] else 'NO CONTEXT (TEXT ONLY)'}")
        print("="*60)
        
        # Find checkpoint
        if args.checkpoint and args.mode != 'both':
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = find_best_checkpoint(model_config['checkpoint_dir'])
        
        print(f"Loading model from: {checkpoint_path}")
        
        # Load model
        model = DebertaV2ForSequenceClassification.from_pretrained(checkpoint_path)
        print("Model loaded successfully!\n")
        
        # Create output directory
        os.makedirs(model_config['predictions_dir'], exist_ok=True)
        
        # Evaluate on all datasets
        all_results = []
        
        print("Evaluating on validation set...")
        val_results = evaluate_and_save_predictions(
            model, tokenizer, val_df, 'val', 
            model_config['predictions_dir'],
            model_config['use_context'],
            model_config['condition_name']
        )
        all_results.append(val_results)
        
        print("\n\nEvaluating on test set...")
        test_results = evaluate_and_save_predictions(
            model, tokenizer, test_df, 'test',
            model_config['predictions_dir'],
            model_config['use_context'],
            model_config['condition_name']
        )
        all_results.append(test_results)
        
        print("\n\nEvaluating on gold set...")
        gold_results = evaluate_and_save_predictions(
            model, tokenizer, gold_df, 'gold',
            model_config['predictions_dir'],
            model_config['use_context'],
            model_config['condition_name']
        )
        all_results.append(gold_results)
        
        # Save summary results
        summary_df = pd.DataFrame(all_results)
        summary_df['model_type'] = mode
        suffix = '' if model_config['use_context'] else '_no_context'
        summary_path = os.path.join(model_config['predictions_dir'], f'finetuned_results_summary{suffix}.csv')
        summary_df.to_csv(summary_path, index=False)
        
        all_summaries.append(summary_df)
        
        context_str = "WITH CONTEXT" if model_config['use_context'] else "TEXT ONLY (NO CONTEXT)"
        print("\n" + "="*60)
        print(f"SUMMARY ({context_str})")
        print("="*60)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
    
    # If running both, print comparison
    if args.mode == 'both' and len(all_summaries) == 2:
        print("\n" + "="*60)
        print("COMPARISON: CONTEXT vs NO CONTEXT")
        print("="*60)
        
        combined = pd.concat(all_summaries, ignore_index=True)
        pivot = combined.pivot(index='dataset', columns='model_type', values=['accuracy', 'f1'])
        print(pivot.to_string())
        
        # Save combined summary
        combined_path = os.path.join(
            os.path.dirname(Config.MODELS['context']['predictions_dir']),
            'comparison_summary.csv'
        )
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined summary saved to: {combined_path}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()