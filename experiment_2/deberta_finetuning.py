import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2Tokenizer, 
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # Reduced for more stable gradients
    LEARNING_RATE = 5e-6  # Much lower learning rate
    NUM_EPOCHS = 5  # More epochs with lower LR
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 50  # Add warmup
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16
    
    # Paths
    TRAIN_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/train.csv'
    VAL_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/val.csv'
    TEST_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/test.csv'
    GOLD_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits//gold.csv'
    
    OUTPUT_DIR = 'outputs/finetuned_deberta'
    PREDICTIONS_DIR = 'outputs/finetuned_predictions_deberta'

# Create output directories
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

# Custom Dataset Class
class SarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine context and text
        # Format: "Context: {context} Response: {text}"
        combined_text = f"Context: {row['context']} Response: {row['text']}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
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

# Metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Prediction and evaluation function
def evaluate_and_save_predictions(model, tokenizer, dataframe, dataset_name, output_dir):
    """
    Evaluate model and save predictions in the same format as baseline predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    confidences = []
    
    # Create dataset
    dataset = SarcasmDataset(dataframe, tokenizer, Config.MAX_LENGTH)
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
            confidences.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (sarcastic)
    
    # Create results dataframe
    results_df = dataframe.copy()
    results_df['true_label'] = true_labels
    results_df['predicted_label'] = predictions
    results_df['correct'] = (np.array(true_labels) == np.array(predictions)).astype(int)
    results_df['confidence'] = confidences
    results_df['model_name'] = 'deberta_v3_finetuned'
    results_df['condition'] = 'context_text_original'
    
    # Add example_id if not present
    if 'example_id' not in results_df.columns:
        results_df.insert(0, 'example_id', range(len(results_df)))
    
    # Save predictions
    output_path = os.path.join(output_dir, f'predictions_{dataset_name}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    cm = confusion_matrix(true_labels, predictions)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results on {dataset_name.upper()} set")
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
    print("="*60)
    print("DeBERTa-v3-base Fine-tuning for Sarcasm Detection")
    print("(WITH CONTEXT)")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    gold_df = pd.read_csv(Config.GOLD_PATH)
    
    print(f"Train: {len(train_df)} examples")
    print(f"Val:   {len(val_df)} examples")
    print(f"Test:  {len(test_df)} examples")
    print(f"Gold:  {len(gold_df)} examples")
    
    # Load tokenizer and model
    print(f"\nLoading {Config.MODEL_NAME}...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(Config.MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Print label distribution to verify balance
    print("\nLabel distribution in training set:")
    print(train_df['label'].value_counts())
    print(f"Class 0 (literal): {(train_df['label']==0).sum()} ({(train_df['label']==0).mean()*100:.1f}%)")
    print(f"Class 1 (sarcastic): {(train_df['label']==1).sum()} ({(train_df['label']==1).mean()*100:.1f}%)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SarcasmDataset(train_df, tokenizer, Config.MAX_LENGTH)
    val_dataset = SarcasmDataset(val_df, tokenizer, Config.MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir=f'{Config.OUTPUT_DIR}/logs',
        logging_steps=10,
        logging_first_step=True,
        seed=42,
        fp16=False,  # Disable mixed precision for stability
        report_to='none',  # Disable wandb/tensorboard
        greater_is_better=True,
        save_total_limit=2
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\nStarting training...")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Gradient accumulation steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Warmup steps: {Config.WARMUP_STEPS}\n")
    
    print("⚠️  MONITORING: Watch for both classes being predicted during training")
    print("⚠️  If you see val F1 = 0.33 consistently, the model is collapsing to one class!\n")
    
    train_result = trainer.train()
    
    # Check if model is still predicting only one class
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print("\nQuick sanity check on training set...")
    
    # Get predictions on a few training examples to verify model isn't broken
    sample_dataset = SarcasmDataset(train_df.head(20), tokenizer, Config.MAX_LENGTH)
    sample_loader = DataLoader(sample_dataset, batch_size=20)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        batch = next(iter(sample_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        
    print(f"Sample predictions on first 20 training examples: {preds}")
    print(f"Unique predictions: {np.unique(preds)}")
    
    if len(np.unique(preds)) == 1:
        print("\n⚠️  WARNING: Model is predicting only ONE class!")
        print("This indicates training failure. Check logs above.")
    else:
        print("\n✓ Model is predicting multiple classes - training likely successful")
    
    # Evaluate on all datasets
    all_results = []
    
    print("\n\nEvaluating on validation set...")
    val_results = evaluate_and_save_predictions(
        model, tokenizer, val_df, 'val', Config.PREDICTIONS_DIR
    )
    all_results.append(val_results)
    
    print("\n\nEvaluating on test set...")
    test_results = evaluate_and_save_predictions(
        model, tokenizer, test_df, 'test', Config.PREDICTIONS_DIR
    )
    all_results.append(test_results)
    
    print("\n\nEvaluating on gold set...")
    gold_results = evaluate_and_save_predictions(
        model, tokenizer, gold_df, 'gold', Config.PREDICTIONS_DIR
    )
    all_results.append(gold_results)
    
    # Save summary results
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(Config.PREDICTIONS_DIR, 'finetuned_results_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY (DeBERTa-v3-base WITH CONTEXT)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
