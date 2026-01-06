#!/usr/bin/env python3
"""
Preprocessing Module for Sarcasm Detection
==========================================

This module handles all text preprocessing for the sarcasm detection experiments.

Preprocessing steps:
1. Basic cleaning (HTML removal, smart quote normalization, whitespace)
2. Pipe separator handling (dialogue turn boundaries)
3. Character name normalization (optional)
4. Case normalization (for TF-IDF only)

Usage:
    from preprocessing import TextPreprocessor
    
    prep = TextPreprocessor()
    
    # For TF-IDF models (Tier 2)
    text = prep.preprocess_for_tfidf("SHELDON: Hello | LEONARD: Hi", normalize_names=False)
    # Output: "sheldon: hello. leonard: hi"
    
    # For embedding models (Tier 3)
    text = prep.preprocess_for_embeddings("SHELDON: Hello | LEONARD: Hi", normalize_names=True)
    # Output: "SPEAKER: Hello\nSPEAKER: Hi"

Author: Aşko
Date: January 2026
"""

import re
import pandas as pd
from typing import Optional


class TextPreprocessor:
    """
    Handles all text preprocessing for sarcasm detection.
    
    Two preprocessing modes:
    - TF-IDF: Lowercase, pipe→period (for bag-of-words models)
    - Embeddings: Keep case, pipe→newline (for transformer models)
    
    Optional name normalization replaces character names (SHELDON:, LEONARD:, etc.)
    with generic SPEAKER: tag to reduce character-specific bias.
    """
    
    def __init__(self):
        # Regex pattern for character names
        # Matches: SHELDON:, LEONARD:, PERSON1:, PERSON2:, etc.
        # Pattern: Word boundary + uppercase letters + optional digits + colon
        self.name_pattern = re.compile(r'\b[A-Z]+\d*:')
        
        # HTML tag pattern (for removing <i>, </i>, etc.)
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Smart quotes and special characters to normalize to ASCII
        # Using Unicode escapes for reliability across different editors/systems
        self.smart_quotes = {
            "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK (")
            "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK (")
            "\u2018": "'",   # LEFT SINGLE QUOTATION MARK (')
            "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK (')
            "\u2013": "-",   # EN DASH (–)
            "\u2014": "-",   # EM DASH (—)
            "\u2026": "...", # HORIZONTAL ELLIPSIS (…)
        }
        
        # Known character names in the datasets (for reference/documentation)
        self.known_characters = [
            # Big Bang Theory
            'SHELDON', 'LEONARD', 'PENNY', 'HOWARD', 'RAJ', 'AMY', 'BERNADETTE',
            # Friends
            'CHANDLER', 'JOEY', 'MONICA', 'RACHEL', 'ROSS', 'PHOEBE',
            # Generic
            'PERSON', 'PERSON1', 'PERSON2'
        ]
    
    def clean_basic(self, text: str) -> str:
        """
        Basic cleaning applied to all text regardless of model type.
        
        Steps:
        1. Handle NaN/None values
        2. Convert to string
        3. Remove HTML tags
        4. Normalize smart quotes to ASCII
        5. Remove carriage returns
        6. Normalize whitespace
        
        Args:
            text: Input text (can be NaN, None, or string)
        
        Returns:
            Cleaned text string
            
        Example:
            >>> prep = TextPreprocessor()
            >>> prep.clean_basic('<i>Star Trek</i> is "great"')
            'Star Trek is "great"'
        """
        # Handle NaN/None
        if pd.isna(text) or text is None:
            return ""
        
        # Ensure string type
        text = str(text)
        
        # Remove HTML tags (e.g., <i>Star Trek</i> → Star Trek)
        text = self.html_pattern.sub('', text)
        
        # Normalize smart quotes and special characters to ASCII
        for smart_char, ascii_char in self.smart_quotes.items():
            text = text.replace(smart_char, ascii_char)
        
        # Remove carriage returns (Windows line endings)
        text = text.replace('\r', '')
        
        # Normalize whitespace (multiple spaces/tabs → single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_names(self, text: str) -> str:
        """
        Replace character names with generic SPEAKER: tag.
        
        This helps reduce character-specific bias in the model.
        For example, if SHELDON is always sarcastic, the model might
        learn to predict sarcasm based on the name rather than content.
        
        Args:
            text: Input text with character names
        
        Returns:
            Text with normalized speaker tags
            
        Examples:
            >>> prep = TextPreprocessor()
            >>> prep.normalize_names("SHELDON: Hello")
            'SPEAKER: Hello'
            >>> prep.normalize_names("PERSON1: Hi | PERSON2: Hello")
            'SPEAKER: Hi | SPEAKER: Hello'
        """
        return self.name_pattern.sub('SPEAKER:', text)
    
    def preprocess_for_tfidf(self, text: str, normalize_names: bool = False) -> str:
        """
        Preprocess text for TF-IDF/bag-of-words models (Tier 2).
        
        Steps:
        1. Basic cleaning
        2. Replace pipe separator with period+space (turn boundary marker)
        3. Optionally normalize character names
        4. Convert to lowercase
        
        The pipe separator " | " in the original data marks dialogue turn boundaries.
        For TF-IDF, we convert this to ". " to maintain sentence boundaries.
        
        Args:
            text: Input text
            normalize_names: If True, replace character names with SPEAKER:
        
        Returns:
            Preprocessed lowercase text suitable for TF-IDF
        
        Examples:
            >>> prep = TextPreprocessor()
            >>> prep.preprocess_for_tfidf("SHELDON: Hello | LEONARD: Hi")
            'sheldon: hello. leonard: hi'
            >>> prep.preprocess_for_tfidf("SHELDON: Hello | LEONARD: Hi", normalize_names=True)
            'speaker: hello. speaker: hi'
        """
        # Apply basic cleaning
        text = self.clean_basic(text)
        
        # Replace pipe separator with period+space (preserves sentence boundary)
        text = text.replace(' | ', '. ')
        
        # Normalize names if requested
        if normalize_names:
            text = self.normalize_names(text)
        
        # Convert to lowercase for bag-of-words
        text = text.lower()
        
        return text
    
    def preprocess_for_embeddings(self, text: str, normalize_names: bool = False) -> str:
        """
        Preprocess text for transformer embedding models (Tier 3).
        
        Steps:
        1. Basic cleaning
        2. Replace pipe separator with newline (preserves dialogue structure)
        3. Optionally normalize character names
        4. Keep original case (transformers are case-sensitive)
        
        The pipe separator " | " is converted to newline to preserve
        the dialogue turn structure, which transformers can leverage.
        
        Args:
            text: Input text
            normalize_names: If True, replace character names with SPEAKER:
        
        Returns:
            Preprocessed text suitable for transformer embeddings
        
        Examples:
            >>> prep = TextPreprocessor()
            >>> prep.preprocess_for_embeddings("SHELDON: Hello | LEONARD: Hi")
            'SHELDON: Hello\\nLEONARD: Hi'
            >>> prep.preprocess_for_embeddings("SHELDON: Hello | LEONARD: Hi", normalize_names=True)
            'SPEAKER: Hello\\nSPEAKER: Hi'
        """
        # Apply basic cleaning
        text = self.clean_basic(text)
        
        # Replace pipe separator with newline (preserves turn structure)
        text = text.replace(' | ', '\n')
        
        # Normalize names if requested
        if normalize_names:
            text = self.normalize_names(text)
        
        # Keep original case (XLM-R and other transformers are case-sensitive)
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
        Prepare combined input for model based on experimental condition.
        
        This is the main method to use when preparing data for experiments.
        It handles all combinations of input types and model types.
        
        Args:
            context: The dialogue context
            text: The response to classify
            input_type: One of 'text_only', 'context_text', 'context_only'
            for_model: One of 'tfidf', 'embeddings'
            normalize_names: If True, replace character names with SPEAKER:
        
        Returns:
            Preprocessed input string ready for the model
        
        Examples:
            >>> prep = TextPreprocessor()
            >>> prep.prepare_input("SHELDON: What?", "LEONARD: Nothing.", 'text_only', 'tfidf')
            'leonard: nothing.'
            >>> prep.prepare_input("SHELDON: What?", "LEONARD: Nothing.", 'context_text', 'tfidf')
            'Context: sheldon: what? Response: leonard: nothing.'
            >>> prep.prepare_input("SHELDON: What?", "LEONARD: Nothing.", 'context_text', 'embeddings', True)
            'Context: SPEAKER: What? Response: SPEAKER: Nothing.'
        """
        # Choose preprocessing function based on model type
        if for_model == 'tfidf':
            preprocess_fn = self.preprocess_for_tfidf
        else:
            preprocess_fn = self.preprocess_for_embeddings
        
        # Preprocess context and text
        ctx_processed = preprocess_fn(context, normalize_names)
        txt_processed = preprocess_fn(text, normalize_names)
        
        # Combine based on input type
        if input_type == 'text_only':
            return txt_processed
        elif input_type == 'context_only':
            return ctx_processed
        else:  # context_text
            return f"Context: {ctx_processed} Response: {txt_processed}"


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def run_tests():
    """Run preprocessing tests to verify everything works correctly."""
    
    prep = TextPreprocessor()
    
    print("="*60)
    print("PREPROCESSING MODULE TESTS")
    print("="*60)
    
    # Test 1: Basic cleaning
    print("\n1. Basic Cleaning Tests")
    print("-"*40)
    
    test_cases_basic = [
        # (input, expected_output, description)
        ("Hello world", "Hello world", "Normal text"),
        ("<i>Star Trek</i>", "Star Trek", "HTML tags"),
        ("\u201cQuoted\u201d", '"Quoted"', "Smart double quotes"),
        ("It\u2019s great", "It's great", "Smart apostrophe"),
        ("one\u2013two\u2014three", "one-two-three", "Dashes"),
        ("wait\u2026", "wait...", "Ellipsis"),
        ("  multiple   spaces  ", "multiple spaces", "Whitespace normalization"),
        ("line\r\nbreak", "line break", "Carriage return"),
    ]
    
    all_passed = True
    for input_text, expected, description in test_cases_basic:
        result = prep.clean_basic(input_text)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
            print(f"  {status}: {description}")
            print(f"         Input: {repr(input_text)}")
            print(f"         Expected: {repr(expected)}")
            print(f"         Got: {repr(result)}")
        else:
            print(f"  {status}: {description}")
    
    # Test 2: Name normalization
    print("\n2. Name Normalization Tests")
    print("-"*40)
    
    test_cases_names = [
        ("SHELDON: Hello", "SPEAKER: Hello"),
        ("PERSON1: Hi | PERSON2: Hello", "SPEAKER: Hi | SPEAKER: Hello"),
        ("No names here", "No names here"),
    ]
    
    for input_text, expected in test_cases_names:
        result = prep.normalize_names(input_text)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: {repr(input_text)[:40]}...")
    
    # Test 3: TF-IDF preprocessing
    print("\n3. TF-IDF Preprocessing Tests")
    print("-"*40)
    
    test_cases_tfidf = [
        ("SHELDON: Hello | LEONARD: Hi", False, "sheldon: hello. leonard: hi"),
        ("SHELDON: Hello | LEONARD: Hi", True, "speaker: hello. speaker: hi"),
    ]
    
    for input_text, normalize, expected in test_cases_tfidf:
        result = prep.preprocess_for_tfidf(input_text, normalize)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: normalize={normalize}")
    
    # Test 4: Embeddings preprocessing
    print("\n4. Embeddings Preprocessing Tests")
    print("-"*40)
    
    test_cases_embed = [
        ("SHELDON: Hello | LEONARD: Hi", False, "SHELDON: Hello\nLEONARD: Hi"),
        ("SHELDON: Hello | LEONARD: Hi", True, "SPEAKER: Hello\nSPEAKER: Hi"),
    ]
    
    for input_text, normalize, expected in test_cases_embed:
        result = prep.preprocess_for_embeddings(input_text, normalize)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: normalize={normalize}")
    
    # Test 5: Combined input preparation
    print("\n5. Combined Input Preparation Tests")
    print("-"*40)
    
    context = "SHELDON: What do you think?"
    text = "LEONARD: It's great."
    
    test_cases_combined = [
        ('text_only', 'tfidf', False, "leonard: it's great."),
        ('context_only', 'tfidf', False, "sheldon: what do you think?"),
        ('context_text', 'tfidf', False, "Context: sheldon: what do you think? Response: leonard: it's great."),
        ('context_text', 'tfidf', True, "Context: speaker: what do you think? Response: speaker: it's great."),
        ('text_only', 'embeddings', False, "LEONARD: It's great."),
        ('context_text', 'embeddings', True, "Context: SPEAKER: What do you think? Response: SPEAKER: It's great."),
    ]
    
    for input_type, for_model, normalize, expected in test_cases_combined:
        result = prep.prepare_input(context, text, input_type, for_model, normalize)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
            print(f"  {status}: {input_type}, {for_model}, norm={normalize}")
            print(f"         Expected: {repr(expected)}")
            print(f"         Got: {repr(result)}")
        else:
            print(f"  {status}: {input_type}, {for_model}, norm={normalize}")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please review above")
    print("="*60)
    
    return all_passed


def demonstrate():
    """Demonstrate preprocessing on example texts from the dataset."""
    
    prep = TextPreprocessor()
    
    print("\n" + "="*60)
    print("PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Example from the actual dataset
    example = {
        'context': 'SHELDON: Ask me why. | LEONARD: Do I have to? | SHELDON: Of course. That\'s how you move a conversation forward.',
        'text': 'PENNY: I hope I\'m a waitress at the Cheesecake Factory for my whole life.',
        'label': 1  # sarcastic
    }
    
    print(f"\nOriginal Context:\n  {example['context']}")
    print(f"\nOriginal Text:\n  {example['text']}")
    print(f"\nLabel: {example['label']} (sarcastic)")
    
    print("\n" + "-"*60)
    print("Preprocessed Outputs:")
    print("-"*60)
    
    conditions = [
        ('text_only', 'tfidf', False),
        ('text_only', 'tfidf', True),
        ('context_text', 'tfidf', False),
        ('context_text', 'embeddings', False),
        ('context_text', 'embeddings', True),
    ]
    
    for input_type, for_model, normalize in conditions:
        result = prep.prepare_input(
            example['context'], 
            example['text'], 
            input_type, 
            for_model, 
            normalize
        )
        print(f"\n[{input_type}, {for_model}, norm={normalize}]")
        print(f"  {result[:100]}{'...' if len(result) > 100 else ''}")


if __name__ == "__main__":
    # Run tests
    run_tests()
    
    # Show demonstration
    demonstrate()
