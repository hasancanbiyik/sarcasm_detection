#!/usr/bin/env python3
"""
Step 2: Text Preprocessing
==========================
This script applies preprocessing to the data splits created in Step 1.

It creates preprocessed versions for different experimental conditions:
- TF-IDF preprocessing (lowercase, pipe → period)
- Embedding preprocessing (preserve case, pipe → newline)
- Name normalization variants (original vs SPEAKER:)
- Input type variants (text_only, context_text, context_only)

Run this AFTER step1_split_data.py

Input:
    - outputs/data_splits/train.csv
    - outputs/data_splits/val.csv
    - outputs/data_splits/test.csv
    - outputs/data_splits/gold.csv

Output:
    - outputs/preprocessed/ (various preprocessed files)
    - Console output showing preprocessing examples

Author: Hasan Can Biyik
Date: January 2026
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input directory (from Step 1)
SPLITS_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits"

# Output directory
OUTPUT_DIR = "outputs/preprocessed"

# =============================================================================
# PREPROCESSING CLASS
# =============================================================================

class TextPreprocessor:
    """Handles all text preprocessing for sarcasm detection."""
    
    def __init__(self):
        # Regex pattern for character names (e.g., "SHELDON:", "PERSON1:")
        self.name_pattern = re.compile(r'\b[A-Z]+\d*:')
        
        # HTML tag pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Smart quotes and special characters (Unicode escapes for reliability)
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
        """
        Preprocess text for TF-IDF models.
        - Basic cleaning
        - Pipe separator → period + space
        - Optional name normalization
        - Lowercase
        """
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
        """
        Preprocess text for embedding models (e.g., XLM-R).
        - Basic cleaning
        - Pipe separator → newline
        - Optional name normalization
        - Preserve case
        """
        text = self.clean_basic(text)
        
        # Replace pipe separator with newline (preserves turn structure)
        text = text.replace(' | ', '\n')
        
        # Normalize names if requested
        if normalize_names:
            text = self.normalize_names(text)
        
        # Keep original case (transformers are case-sensitive)
        return text
    
    def prepare_input(
        self, 
        context: str, 
        text: str, 
        input_type: str, 
        for_model: str, 
        normalize_names: bool = False
    ) -> str:
        """
        Prepare combined input based on experimental condition.
        
        Args:
            context: The dialogue context
            text: The response to classify
            input_type: 'text_only', 'context_text', or 'context_only'
            for_model: 'tfidf' or 'embeddings'
            normalize_names: Whether to normalize character names
        
        Returns:
            Preprocessed input string
        """
        if for_model == 'tfidf':
            ctx = self.preprocess_for_tfidf(context, normalize_names)
            txt = self.preprocess_for_tfidf(text, normalize_names)
        else:  # embeddings
            ctx = self.preprocess_for_embeddings(context, normalize_names)
            txt = self.preprocess_for_embeddings(text, normalize_names)
        
        if input_type == 'text_only':
            return txt
        elif input_type == 'context_only':
            return ctx
        else:  # context_text
            return f"Context: {ctx} Response: {txt}"


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = Path(SPLITS_DIR)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # -------------------------------------------------------------------------
    # 1. LOAD SPLITS
    # -------------------------------------------------------------------------
    print("1. LOADING SPLITS FROM STEP 1")
    print("-" * 40)
    
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    gold_df = pd.read_csv(splits_dir / "gold.csv")
    
    print(f"   Train: {len(train_df)} examples")
    print(f"   Val: {len(val_df)} examples")
    print(f"   Test: {len(test_df)} examples")
    print(f"   Gold: {len(gold_df)} examples")
    print()
    
    # -------------------------------------------------------------------------
    # 2. DEMONSTRATE PREPROCESSING
    # -------------------------------------------------------------------------
    print("2. PREPROCESSING DEMONSTRATION")
    print("-" * 40)
    
    # Pick a sample example
    sample_row = train_df.iloc[0]
    sample_context = sample_row['context']
    sample_text = sample_row['text']
    sample_label = sample_row['label']
    
    print(f"   Sample example:")
    print(f"   Label: {sample_label} ({'sarcastic' if sample_label == 1 else 'literal'})")
    print()
    print(f"   Original context:")
    print(f"   {sample_context[:200]}..." if len(str(sample_context)) > 200 else f"   {sample_context}")
    print()
    print(f"   Original text:")
    print(f"   {sample_text}")
    print()
    
    print("   Preprocessing variants:")
    print()
    
    # TF-IDF, text_only, original names
    result = preprocessor.prepare_input(sample_context, sample_text, 'text_only', 'tfidf', False)
    print(f"   [tfidf, text_only, original_names]:")
    print(f"   {result}")
    print()
    
    # TF-IDF, text_only, normalized names
    result = preprocessor.prepare_input(sample_context, sample_text, 'text_only', 'tfidf', True)
    print(f"   [tfidf, text_only, normalized_names]:")
    print(f"   {result}")
    print()
    
    # Embeddings, text_only, original names
    result = preprocessor.prepare_input(sample_context, sample_text, 'text_only', 'embeddings', False)
    print(f"   [embeddings, text_only, original_names]:")
    print(f"   {result}")
    print()
    
    # TF-IDF, context_text, original names
    result = preprocessor.prepare_input(sample_context, sample_text, 'context_text', 'tfidf', False)
    print(f"   [tfidf, context_text, original_names]:")
    print(f"   {result[:150]}..." if len(result) > 150 else f"   {result}")
    print()
    
    # -------------------------------------------------------------------------
    # 3. CHECK FOR PREPROCESSING ISSUES
    # -------------------------------------------------------------------------
    print("3. CHECKING FOR POTENTIAL ISSUES")
    print("-" * 40)
    
    all_dfs = {'train': train_df, 'val': val_df, 'test': test_df, 'gold': gold_df}
    
    issues_found = False
    
    for split_name, df in all_dfs.items():
        # Check for NaN values
        nan_text = df['text'].isna().sum()
        nan_context = df['context'].isna().sum()
        
        if nan_text > 0 or nan_context > 0:
            print(f"   ⚠️  {split_name}: {nan_text} NaN in 'text', {nan_context} NaN in 'context'")
            issues_found = True
        
        # Check for empty strings after basic cleaning
        empty_count = 0
        for _, row in df.iterrows():
            cleaned = preprocessor.clean_basic(row['text'])
            if cleaned == "":
                empty_count += 1
        
        if empty_count > 0:
            print(f"   ⚠️  {split_name}: {empty_count} examples become empty after cleaning")
            issues_found = True
    
    if not issues_found:
        print("   ✓ No issues found in any split")
    print()
    
    # -------------------------------------------------------------------------
    # 4. PREPROCESS AND SAVE ALL VARIANTS
    # -------------------------------------------------------------------------
    print("4. PREPROCESSING ALL VARIANTS")
    print("-" * 40)
    
    # Define all experimental conditions
    conditions = [
        # (input_type, for_model, normalize_names)
        ('text_only', 'tfidf', False),
        ('text_only', 'tfidf', True),
        ('text_only', 'embeddings', False),
        ('text_only', 'embeddings', True),
        ('context_text', 'tfidf', False),
        ('context_text', 'tfidf', True),
        ('context_text', 'embeddings', False),
        ('context_text', 'embeddings', True),
        ('context_only', 'tfidf', False),
        ('context_only', 'tfidf', True),
        ('context_only', 'embeddings', False),
        ('context_only', 'embeddings', True),
    ]
    
    for input_type, for_model, normalize_names in conditions:
        norm_str = "normalized" if normalize_names else "original"
        condition_name = f"{for_model}_{input_type}_{norm_str}"
        
        print(f"   Processing: {condition_name}")
        
        for split_name, df in all_dfs.items():
            # Create preprocessed column
            preprocessed_texts = []
            for _, row in df.iterrows():
                processed = preprocessor.prepare_input(
                    row['context'], 
                    row['text'], 
                    input_type, 
                    for_model, 
                    normalize_names
                )
                preprocessed_texts.append(processed)
            
            # Create output dataframe
            output_df = df.copy()
            output_df['preprocessed'] = preprocessed_texts
            
            # Save
            filename = f"{split_name}_{condition_name}.csv"
            output_df.to_csv(output_dir / filename, index=False)
    
    print()
    print(f"   Saved {len(conditions) * 4} preprocessed files to {output_dir}")
    print()
    
    # -------------------------------------------------------------------------
    # 5. SUMMARY
    # -------------------------------------------------------------------------
    print("5. PREPROCESSING SUMMARY")
    print("-" * 40)
    
    print(f"   Input types: text_only, context_text, context_only")
    print(f"   Model types: tfidf, embeddings")
    print(f"   Name variants: original, normalized")
    print(f"   Total conditions: {len(conditions)}")
    print()
    
    print("   Key preprocessing differences:")
    print("   - TF-IDF: lowercase, pipe→period")
    print("   - Embeddings: preserve case, pipe→newline")
    print("   - Normalized: SHELDON:/CHANDLER:/etc. → SPEAKER:")
    print()
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"Preprocessed files saved to: {output_dir}")
    print(f"Next: Run step3_run_baselines.py to train and evaluate models")
    print()


if __name__ == "__main__":
    main()