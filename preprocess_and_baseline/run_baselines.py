#!/usr/bin/env python3
"""
Sarcasm Detection Baseline Experiments
======================================
Runs Tier 1, 2, and 3 baselines with comprehensive preprocessing and evaluation.

Tier 1: Random and Majority baselines
Tier 2: TF-IDF + Logistic Regression / SVM
Tier 3: Frozen XLM-R embeddings + Logistic Regression (CLS and Mean pooling)

Author: Aşko
Date: January 2026
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for the baseline experiments."""
    
    # File paths - UPDATE THESE TO YOUR LOCAL PATHS
    ORANGE_DATASET_PATH = "data/main_dataset_orange.csv"
    GOLD_DATASET_PATH = "data/sample_dataset_gold.csv"
    OUTPUT_DIR = "outputs"
    
    # Split ratios
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    
    # Random seed
    RANDOM_SEED = 42
    
    # Character names to normalize (will be replaced with SPEAKER:)
    CHARACTER_NAMES = [
        'SHELDON', 'LEONARD', 'PENNY', 'HOWARD', 'RAJ', 'AMY', 'BERNADETTE',
        'CHANDLER', 'JOEY', 'MONICA', 'RACHEL', 'ROSS', 'PHOEBE',
        'PERSON', 'PERSON1', 'PERSON2'
    ]


# =============================================================================
# PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Handles all text preprocessing for sarcasm detection."""
    
    def __init__(self):
        # Regex pattern for character names (e.g., "SHELDON:", "PERSON1:")
        self.name_pattern = re.compile(r'\b[A-Z]+\d*:')
        
        # HTML tag pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Smart quotes and special characters
        # Using Unicode escapes for reliability across different editors/systems
        self.smart_quotes = {
            "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
            "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
            "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
            "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
            "\u2013": "-",  # EN DASH
            "\u2014": "-",  # EM DASH
            "\u2026": "...", # HORIZONTAL ELLIPSIS
        }
    
    def clean_basic(self, text: str) -> str:
        """Basic cleaning applied to all text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags (e.g., <i>Star Trek</i>)
        text = self.html_pattern.sub('', text)
        
        # Normalize smart quotes and special characters to ASCII
        for smart, ascii_char in self.smart_quotes.items():
            text = text.replace(smart, ascii_char)
        
        # Remove carriage returns
        text = text.replace('\r', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_names(self, text: str) -> str:
        """Replace character names with generic SPEAKER: tag."""
        return self.name_pattern.sub('SPEAKER:', text)
    
    def preprocess_for_tfidf(self, text: str, normalize_names: bool = False) -> str:
        """Preprocess text for TF-IDF models (Tier 2)."""
        text = self.clean_basic(text)
        
        # Replace pipe separator with period+space (turn boundary)
        text = text.replace(' | ', '. ')
        
        # Normalize names if requested
        if normalize_names:
            text = self.normalize_names(text)
        
        # Lowercase for bag-of-words
        text = text.lower()
        
        return text
    
    def preprocess_for_embeddings(self, text: str, normalize_names: bool = False) -> str:
        """Preprocess text for embedding models (Tier 3)."""
        text = self.clean_basic(text)
        
        # Replace pipe separator with newline (preserves turn structure)
        text = text.replace(' | ', '\n')
        
        # Normalize names if requested
        if normalize_names:
            text = self.normalize_names(text)
        
        # Keep original case (XLM-R is case-sensitive)
        return text
    
    def combine_context_and_text(self, context: str, text: str, for_model: str = 'tfidf') -> str:
        """Combine context and response text with appropriate format."""
        if for_model == 'tfidf':
            context = self.preprocess_for_tfidf(context, normalize_names=False)
            text = self.preprocess_for_tfidf(text, normalize_names=False)
        else:
            context = self.preprocess_for_embeddings(context, normalize_names=False)
            text = self.preprocess_for_embeddings(text, normalize_names=False)
        
        return f"Context: {context} Response: {text}"


# =============================================================================
# DATA LOADING AND SPLITTING
# =============================================================================

class DataManager:
    """Handles data loading, gold separation, and train/val/test splitting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        
        # Data containers
        self.orange_df = None
        self.gold_df = None
        self.orange_clean_df = None
        
        # Splits
        self.train_df = None
        self.val_df = None
        self.test_df = None
    
    def load_data(self) -> None:
        """Load orange and gold datasets."""
        print("Loading datasets...")
        
        self.orange_df = pd.read_csv(self.config.ORANGE_DATASET_PATH)
        self.gold_df = pd.read_csv(self.config.GOLD_DATASET_PATH)
        
        print(f"  Orange dataset: {len(self.orange_df)} examples")
        print(f"  Gold dataset: {len(self.gold_df)} examples")
        print(f"  Orange label distribution: {dict(self.orange_df['label'].value_counts())}")
        print(f"  Gold label distribution: {dict(self.gold_df['label'].value_counts())}")
    
    def separate_gold_from_orange(self) -> None:
        """Remove gold examples from orange dataset."""
        print("\nSeparating gold examples from orange dataset...")
        
        # Create a key for matching (using 'text' column which should be unique)
        gold_texts = set(self.gold_df['text'].str.strip())
        
        # Filter orange to remove gold examples
        mask = ~self.orange_df['text'].str.strip().isin(gold_texts)
        self.orange_clean_df = self.orange_df[mask].reset_index(drop=True)
        
        removed_count = len(self.orange_df) - len(self.orange_clean_df)
        print(f"  Removed {removed_count} gold examples from orange")
        print(f"  Orange-clean dataset: {len(self.orange_clean_df)} examples")
        print(f"  Orange-clean label distribution: {dict(self.orange_clean_df['label'].value_counts())}")
    
    def create_splits(self) -> None:
        """Create stratified train/val/test splits from orange-clean."""
        print("\nCreating train/val/test splits (80-10-10)...")
        
        X = self.orange_clean_df[['context', 'text']]
        y = self.orange_clean_df['label']
        
        # First split: 80% train, 20% temp (for val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            stratify=y,
            random_state=self.config.RANDOM_SEED
        )
        
        # Second split: 50% of temp for val, 50% for test (i.e., 10% each of total)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.config.RANDOM_SEED
        )
        
        # Create DataFrames
        self.train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        self.val_df = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
        self.test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
        
        print(f"  Train set: {len(self.train_df)} examples (label dist: {dict(self.train_df['label'].value_counts())})")
        print(f"  Val set: {len(self.val_df)} examples (label dist: {dict(self.val_df['label'].value_counts())})")
        print(f"  Test set: {len(self.test_df)} examples (label dist: {dict(self.test_df['label'].value_counts())})")
        print(f"  Gold set: {len(self.gold_df)} examples (held-out)")
    
    def get_prepared_data(self, input_type: str, for_model: str, normalize_names: bool) -> Dict:
        """
        Prepare data for a specific experimental condition.
        
        Args:
            input_type: 'text_only', 'context_text', or 'context_only'
            for_model: 'tfidf' or 'embeddings'
            normalize_names: Whether to normalize character names
        
        Returns:
            Dictionary with train/val/test/gold data
        """
        def prepare_df(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
            texts = []
            for _, row in df.iterrows():
                if for_model == 'tfidf':
                    ctx = self.preprocessor.preprocess_for_tfidf(row['context'], normalize_names)
                    txt = self.preprocessor.preprocess_for_tfidf(row['text'], normalize_names)
                else:
                    ctx = self.preprocessor.preprocess_for_embeddings(row['context'], normalize_names)
                    txt = self.preprocessor.preprocess_for_embeddings(row['text'], normalize_names)
                
                if input_type == 'text_only':
                    texts.append(txt)
                elif input_type == 'context_only':
                    texts.append(ctx)
                else:  # context_text
                    texts.append(f"Context: {ctx} Response: {txt}")
            
            return texts, df['label'].values, df
        
        train_texts, train_labels, train_orig = prepare_df(self.train_df)
        val_texts, val_labels, val_orig = prepare_df(self.val_df)
        test_texts, test_labels, test_orig = prepare_df(self.test_df)
        gold_texts, gold_labels, gold_orig = prepare_df(self.gold_df)
        
        return {
            'train': {'texts': train_texts, 'labels': train_labels, 'original': train_orig},
            'val': {'texts': val_texts, 'labels': val_labels, 'original': val_orig},
            'test': {'texts': test_texts, 'labels': test_labels, 'original': test_orig},
            'gold': {'texts': gold_texts, 'labels': gold_labels, 'original': gold_orig}
        }


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

class Evaluator:
    """Handles model evaluation and results saving."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all results for summary
        self.all_results = []
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_class0': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall_class0': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'f1_class0': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            'precision_class1': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'recall_class1': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_class1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = cm[0, 0] if cm.shape == (2, 2) else 0
        metrics['fp'] = cm[0, 1] if cm.shape == (2, 2) else 0
        metrics['fn'] = cm[1, 0] if cm.shape == (2, 2) else 0
        metrics['tp'] = cm[1, 1] if cm.shape == (2, 2) else 0
        
        return metrics
    
    def save_predictions(
        self,
        original_df: pd.DataFrame,
        processed_texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        model_name: str,
        input_type: str,
        dataset_name: str,
        normalize_names: bool,
        pooling: Optional[str] = None
    ) -> None:
        """Save predictions to CSV for error analysis."""
        
        # Build filename
        norm_str = "_normalized" if normalize_names else ""
        pool_str = f"_{pooling}" if pooling else ""
        filename = f"{model_name}_{input_type}{norm_str}{pool_str}_{dataset_name}.csv"
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'example_id': range(len(y_true)),
            'context': original_df['context'].values,
            'text': original_df['text'].values,
            'processed_input': processed_texts,
            'true_label': y_true,
            'predicted_label': y_pred,
            'correct': (y_true == y_pred).astype(int),
            'confidence': y_prob[:, 1] if y_prob is not None else None,
            'model_name': model_name,
            'input_type': input_type,
            'names_normalized': normalize_names,
            'pooling': pooling if pooling else 'N/A'
        })
        
        pred_df.to_csv(self.predictions_dir / filename, index=False)
    
    def add_result(
        self,
        tier: str,
        model_name: str,
        input_type: str,
        normalize_names: bool,
        pooling: Optional[str],
        dataset_name: str,
        metrics: Dict
    ) -> None:
        """Add a result to the summary."""
        result = {
            'tier': tier,
            'model': model_name,
            'input_type': input_type,
            'names_normalized': normalize_names,
            'pooling': pooling if pooling else 'N/A',
            'dataset': dataset_name,
            **metrics
        }
        self.all_results.append(result)
    
    def save_summary(self) -> pd.DataFrame:
        """Save summary of all results."""
        summary_df = pd.DataFrame(self.all_results)
        
        # Reorder columns for readability
        col_order = [
            'tier', 'model', 'input_type', 'names_normalized', 'pooling', 'dataset',
            'accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
            'f1_class0', 'f1_class1', 'precision_class0', 'precision_class1',
            'recall_class0', 'recall_class1', 'tp', 'tn', 'fp', 'fn'
        ]
        summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
        
        # Sort by tier, model, input_type
        summary_df = summary_df.sort_values(['tier', 'model', 'input_type', 'dataset'])
        
        # Save to CSV
        summary_df.to_csv(self.output_dir / "baseline_results_summary.csv", index=False)
        
        # Also save a formatted version for easy reading
        summary_df_rounded = summary_df.copy()
        numeric_cols = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                       'f1_class0', 'f1_class1', 'precision_class0', 'precision_class1',
                       'recall_class0', 'recall_class1']
        for col in numeric_cols:
            if col in summary_df_rounded.columns:
                summary_df_rounded[col] = summary_df_rounded[col].round(4)
        
        summary_df_rounded.to_csv(self.output_dir / "baseline_results_summary_rounded.csv", index=False)
        
        return summary_df


# =============================================================================
# TIER 1: TRIVIAL BASELINES
# =============================================================================

class Tier1Baselines:
    """Random and Majority class baselines."""
    
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
    
    def run(self, data_manager: DataManager) -> None:
        """Run Tier 1 baselines."""
        print("\n" + "="*60)
        print("TIER 1: TRIVIAL BASELINES")
        print("="*60)
        
        # Get test and gold labels
        test_labels = data_manager.test_df['label'].values
        gold_labels = data_manager.gold_df['label'].values
        train_labels = data_manager.train_df['label'].values
        
        # Determine majority class from training data
        majority_class = int(pd.Series(train_labels).mode()[0])
        print(f"\nMajority class in training data: {majority_class}")
        
        for idx, (dataset_name, y_true, orig_df) in enumerate([
            ('orange_test', test_labels, data_manager.test_df),
            ('gold', gold_labels, data_manager.gold_df)
        ]):
            print(f"\nEvaluating on {dataset_name}...")
            
            # Random baseline - use different seed for each dataset to avoid identical predictions
            np.random.seed(RANDOM_SEED + idx)
            y_pred_random = np.random.randint(0, 2, size=len(y_true))
            metrics_random = self.evaluator.evaluate(y_true, y_pred_random)
            self.evaluator.add_result('tier1', 'random', 'N/A', False, None, dataset_name, metrics_random)
            self.evaluator.save_predictions(
                orig_df, ['N/A'] * len(y_true), y_true, y_pred_random, None,
                'tier1_random', 'N/A', dataset_name, False
            )
            print(f"  Random: Accuracy={metrics_random['accuracy']:.4f}, F1={metrics_random['f1_macro']:.4f}")
            
            # Majority baseline
            y_pred_majority = np.full(len(y_true), majority_class)
            metrics_majority = self.evaluator.evaluate(y_true, y_pred_majority)
            self.evaluator.add_result('tier1', 'majority', 'N/A', False, None, dataset_name, metrics_majority)
            self.evaluator.save_predictions(
                orig_df, ['N/A'] * len(y_true), y_true, y_pred_majority, None,
                'tier1_majority', 'N/A', dataset_name, False
            )
            print(f"  Majority: Accuracy={metrics_majority['accuracy']:.4f}, F1={metrics_majority['f1_macro']:.4f}")


# =============================================================================
# TIER 2: TF-IDF + CLASSICAL ML
# =============================================================================

class Tier2Baselines:
    """TF-IDF + Logistic Regression / SVM baselines."""
    
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
    
    def run(self, data_manager: DataManager) -> None:
        """Run Tier 2 baselines."""
        print("\n" + "="*60)
        print("TIER 2: TF-IDF + CLASSICAL ML")
        print("="*60)
        
        # Experimental conditions
        input_types = ['text_only', 'context_text', 'context_only']
        normalize_options = [False, True]
        models = {
            'logreg': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            'svm': SVC(kernel='linear', probability=True, random_state=RANDOM_SEED)
        }
        
        for input_type in input_types:
            for normalize_names in normalize_options:
                print(f"\n--- Input: {input_type}, Normalized: {normalize_names} ---")
                
                # Get prepared data
                data = data_manager.get_prepared_data(input_type, 'tfidf', normalize_names)
                
                # Fit TF-IDF vectorizer on training data
                vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                X_train = vectorizer.fit_transform(data['train']['texts'])
                X_val = vectorizer.transform(data['val']['texts'])
                X_test = vectorizer.transform(data['test']['texts'])
                X_gold = vectorizer.transform(data['gold']['texts'])
                
                y_train = data['train']['labels']
                y_val = data['val']['labels']
                y_test = data['test']['labels']
                y_gold = data['gold']['labels']
                
                for model_name, model in models.items():
                    print(f"\n  Training {model_name}...")
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Evaluate on test and gold
                    for dataset_name, X_eval, y_eval, orig_df, texts in [
                        ('orange_test', X_test, y_test, data['test']['original'], data['test']['texts']),
                        ('gold', X_gold, y_gold, data['gold']['original'], data['gold']['texts'])
                    ]:
                        y_pred = model.predict(X_eval)
                        y_prob = model.predict_proba(X_eval) if hasattr(model, 'predict_proba') else None
                        
                        metrics = self.evaluator.evaluate(y_eval, y_pred, y_prob)
                        self.evaluator.add_result(
                            'tier2', f'tfidf_{model_name}', input_type, 
                            normalize_names, None, dataset_name, metrics
                        )
                        self.evaluator.save_predictions(
                            orig_df, texts, y_eval, y_pred, y_prob,
                            f'tier2_tfidf_{model_name}', input_type, dataset_name, normalize_names
                        )
                        print(f"    {dataset_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")


# =============================================================================
# TIER 3: FROZEN EMBEDDINGS + LOGISTIC REGRESSION
# =============================================================================

class Tier3Baselines:
    """Frozen XLM-R embeddings + Logistic Regression baselines."""
    
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> None:
        """Load XLM-R model and tokenizer."""
        print("\nLoading XLM-RoBERTa model...")
        
        try:
            from transformers import XLMRobertaModel, XLMRobertaTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Please install transformers and torch:\n"
                "pip install transformers torch"
            )
        
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.model.eval()
        
        # Use MPS if available (for Mac M-series), else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("  Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("  Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("  Using CPU")
        
        self.model.to(self.device)
    
    def extract_embeddings(
        self, 
        texts: List[str], 
        pooling: str = 'cls',
        batch_size: int = 16
    ) -> np.ndarray:
        """Extract embeddings from texts using XLM-R."""
        import torch
        from tqdm import tqdm
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {pooling} embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                if pooling == 'cls':
                    # Use CLS token embedding (first token)
                    batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
                else:  # mean pooling
                    # Mean of all tokens (excluding padding)
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    masked_hidden = hidden_states * attention_mask
                    sum_hidden = masked_hidden.sum(dim=1)
                    count = attention_mask.sum(dim=1)
                    batch_embeddings = (sum_hidden / count).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def run(self, data_manager: DataManager) -> None:
        """Run Tier 3 baselines."""
        print("\n" + "="*60)
        print("TIER 3: FROZEN XLM-R EMBEDDINGS + LOGISTIC REGRESSION")
        print("="*60)
        
        # Load model
        self.load_model()
        
        # Experimental conditions
        input_types = ['text_only', 'context_text', 'context_only']
        normalize_options = [False, True]
        pooling_options = ['cls', 'mean']
        
        for input_type in input_types:
            for normalize_names in normalize_options:
                print(f"\n--- Input: {input_type}, Normalized: {normalize_names} ---")
                
                # Get prepared data
                data = data_manager.get_prepared_data(input_type, 'embeddings', normalize_names)
                
                for pooling in pooling_options:
                    print(f"\n  Pooling: {pooling}")
                    
                    # Extract embeddings
                    X_train = self.extract_embeddings(data['train']['texts'], pooling)
                    X_val = self.extract_embeddings(data['val']['texts'], pooling)
                    X_test = self.extract_embeddings(data['test']['texts'], pooling)
                    X_gold = self.extract_embeddings(data['gold']['texts'], pooling)
                    
                    y_train = data['train']['labels']
                    y_val = data['val']['labels']
                    y_test = data['test']['labels']
                    y_gold = data['gold']['labels']
                    
                    # Train Logistic Regression
                    print("  Training Logistic Regression...")
                    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
                    clf.fit(X_train, y_train)
                    
                    # Evaluate on test and gold
                    for dataset_name, X_eval, y_eval, orig_df, texts in [
                        ('orange_test', X_test, y_test, data['test']['original'], data['test']['texts']),
                        ('gold', X_gold, y_gold, data['gold']['original'], data['gold']['texts'])
                    ]:
                        y_pred = clf.predict(X_eval)
                        y_prob = clf.predict_proba(X_eval)
                        
                        metrics = self.evaluator.evaluate(y_eval, y_pred, y_prob)
                        self.evaluator.add_result(
                            'tier3', 'xlmr_logreg', input_type,
                            normalize_names, pooling, dataset_name, metrics
                        )
                        self.evaluator.save_predictions(
                            orig_df, texts, y_eval, y_pred, y_prob,
                            'tier3_xlmr_logreg', input_type, dataset_name, normalize_names, pooling
                        )
                        print(f"    {dataset_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all baseline experiments."""
    print("="*60)
    print("SARCASM DETECTION BASELINE EXPERIMENTS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration
    config = Config()
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_manager = DataManager(config)
    evaluator = Evaluator(config.OUTPUT_DIR)
    
    # Load and prepare data
    data_manager.load_data()
    data_manager.separate_gold_from_orange()
    data_manager.create_splits()
    
    # Save the splits for reproducibility
    splits_dir = output_dir / "data_splits"
    splits_dir.mkdir(exist_ok=True)
    data_manager.train_df.to_csv(splits_dir / "train.csv", index=False)
    data_manager.val_df.to_csv(splits_dir / "val.csv", index=False)
    data_manager.test_df.to_csv(splits_dir / "test.csv", index=False)
    data_manager.gold_df.to_csv(splits_dir / "gold.csv", index=False)
    print(f"\nSaved data splits to {splits_dir}")
    
    # Run Tier 1
    tier1 = Tier1Baselines(evaluator)
    tier1.run(data_manager)
    
    # Run Tier 2
    tier2 = Tier2Baselines(evaluator)
    tier2.run(data_manager)
    
    # Run Tier 3
    tier3 = Tier3Baselines(evaluator)
    tier3.run(data_manager)
    
    # Save summary
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    summary_df = evaluator.save_summary()
    
    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Display key metrics
    display_cols = ['tier', 'model', 'input_type', 'names_normalized', 'pooling', 
                   'dataset', 'accuracy', 'f1_macro']
    print(summary_df[display_cols].to_string(index=False))
    
    print(f"\n\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Summary: {output_dir}/baseline_results_summary.csv")
    print(f"  - Predictions: {output_dir}/predictions/")


if __name__ == "__main__":
    main()
