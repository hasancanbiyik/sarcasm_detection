#!/usr/bin/env python3
"""
Step 5: Linguistic Feature Analysis
===================================
Analyzes linguistic features that distinguish correct vs incorrect predictions.

This script extracts and compares:
1. Basic features (length, word count)
2. Punctuation patterns (!, ?, ..., quotes)
3. Lexical features (negation, intensifiers, sarcasm markers)
4. POS-based features (adjective/adverb ratios)
5. Sentiment features (polarity, subjectivity)

Run this AFTER step4_analysis_quantitative.py

Input:
    - outputs/predictions/*.csv

Output:
    - outputs/analysis/linguistic_features.csv (features for all predictions)
    - outputs/analysis/correct_vs_incorrect.csv (statistical comparison)
    - outputs/analysis/fp_vs_fn.csv (False Positive vs False Negative comparison)
    - outputs/analysis/linguistic_report.txt

Author: Hasan Can Biyik
Date: January 2026
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions"
OUTPUT_DIR = "outputs/linguistic_analysis"

# =============================================================================
# LINGUISTIC FEATURE EXTRACTORS
# =============================================================================

# Word lists for feature extraction
NEGATION_WORDS = {
    'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing', 
    'nowhere', 'none', 'hardly', 'barely', 'scarcely', "don't", 
    "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"
}

INTENSIFIERS = {
    'very', 'really', 'so', 'totally', 'absolutely', 'completely',
    'utterly', 'extremely', 'incredibly', 'seriously', 'literally',
    'obviously', 'clearly', 'definitely', 'certainly', 'truly',
    'highly', 'deeply', 'strongly', 'fully', 'entirely'
}

SARCASM_MARKERS = {
    'oh', 'wow', 'great', 'fantastic', 'wonderful', 'brilliant',
    'genius', 'amazing', 'shocking', 'surprise', 'sure', 'right',
    'yeah', 'totally', 'obviously', 'clearly', 'of course', 'gee',
    'well', 'really', 'nice', 'lovely', 'perfect', 'excellent'
}

# Simple positive/negative word lists for sentiment
POSITIVE_WORDS = {
    'good', 'great', 'love', 'like', 'happy', 'wonderful', 'amazing',
    'excellent', 'fantastic', 'perfect', 'best', 'beautiful', 'nice',
    'awesome', 'brilliant', 'fun', 'enjoy', 'glad', 'pleased', 'thanks',
    'thank', 'appreciate', 'lucky', 'fortunate', 'excited', 'joy'
}

NEGATIVE_WORDS = {
    'bad', 'hate', 'terrible', 'awful', 'horrible', 'worst', 'ugly',
    'stupid', 'dumb', 'idiot', 'annoying', 'boring', 'sad', 'angry',
    'upset', 'disappointed', 'sorry', 'unfortunately', 'wrong', 'fail',
    'problem', 'trouble', 'difficult', 'hard', 'pain', 'hurt', 'sick'
}


class LinguisticFeatureExtractor:
    """Extracts linguistic features from text."""
    
    def __init__(self):
        self.name_pattern = re.compile(r'\b[A-Z]+\d*:')
    
    def extract_features(self, text: str) -> Dict:
        """Extract all linguistic features from a single text."""
        
        if pd.isna(text) or text == "":
            return self._empty_features()
        
        text = str(text)
        text_lower = text.lower()
        
        # Remove speaker tags for word analysis
        text_no_speaker = self.name_pattern.sub('', text)
        text_no_speaker_lower = text_no_speaker.lower()
        
        # Tokenize (simple whitespace + punctuation handling)
        words = re.findall(r'\b\w+\b', text_no_speaker_lower)
        words_original_case = re.findall(r'\b\w+\b', text_no_speaker)
        
        features = {}
        
        # -----------------------------------------------------------------
        # 1. BASIC FEATURES
        # -----------------------------------------------------------------
        features['char_length'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        
        # -----------------------------------------------------------------
        # 2. PUNCTUATION FEATURES
        # -----------------------------------------------------------------
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        features['quotes_count'] = text.count('"') + text.count("'")
        features['comma_count'] = text.count(',')
        
        features['has_exclamation'] = int(features['exclamation_count'] > 0)
        features['has_question'] = int(features['question_count'] > 0)
        features['has_ellipsis'] = int(features['ellipsis_count'] > 0)
        
        # Punctuation density
        punct_total = features['exclamation_count'] + features['question_count'] + features['comma_count']
        features['punct_density'] = punct_total / len(words) if words else 0
        
        # -----------------------------------------------------------------
        # 3. LEXICAL FEATURES
        # -----------------------------------------------------------------
        word_set = set(words)
        
        # Negation
        negation_found = word_set.intersection(NEGATION_WORDS)
        features['has_negation'] = int(len(negation_found) > 0)
        features['negation_count'] = sum(1 for w in words if w in NEGATION_WORDS)
        
        # Intensifiers
        intensifier_found = word_set.intersection(INTENSIFIERS)
        features['has_intensifier'] = int(len(intensifier_found) > 0)
        features['intensifier_count'] = sum(1 for w in words if w in INTENSIFIERS)
        
        # Sarcasm markers
        sarcasm_found = word_set.intersection(SARCASM_MARKERS)
        features['has_sarcasm_marker'] = int(len(sarcasm_found) > 0)
        features['sarcasm_marker_count'] = sum(1 for w in words if w in SARCASM_MARKERS)
        
        # Type-token ratio (lexical diversity)
        features['type_token_ratio'] = len(word_set) / len(words) if words else 0
        
        # All caps words (often indicate emphasis)
        all_caps = [w for w in words_original_case if w.isupper() and len(w) > 1]
        features['all_caps_count'] = len(all_caps)
        
        # -----------------------------------------------------------------
        # 4. SENTIMENT FEATURES (simple lexicon-based)
        # -----------------------------------------------------------------
        positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        
        features['positive_word_count'] = positive_count
        features['negative_word_count'] = negative_count
        
        # Sentiment polarity: (positive - negative) / total sentiment words
        total_sentiment = positive_count + negative_count
        if total_sentiment > 0:
            features['sentiment_polarity'] = (positive_count - negative_count) / total_sentiment
        else:
            features['sentiment_polarity'] = 0
        
        # Absolute sentiment (strength regardless of direction)
        features['sentiment_strength'] = total_sentiment / len(words) if words else 0
        
        # -----------------------------------------------------------------
        # 5. STRUCTURAL FEATURES
        # -----------------------------------------------------------------
        # Starts with interjection
        first_word = words[0] if words else ""
        interjections = {'oh', 'wow', 'well', 'gee', 'hey', 'ah', 'um', 'uh'}
        features['starts_with_interjection'] = int(first_word in interjections)
        
        # Contains rhetorical question pattern
        features['has_rhetorical_pattern'] = int(
            bool(re.search(r'\b(who|what|why|how|when)\b.*\?', text_lower)) and
            features['word_count'] < 15
        )
        
        # Contrastive markers (but, however, although)
        contrast_markers = {'but', 'however', 'although', 'though', 'yet', 'still'}
        features['has_contrast'] = int(len(word_set.intersection(contrast_markers)) > 0)
        
        return features
    
    def _empty_features(self) -> Dict:
        """Return empty feature dict for missing text."""
        return {
            'char_length': 0, 'word_count': 0, 'avg_word_length': 0,
            'sentence_count': 0, 'exclamation_count': 0, 'question_count': 0,
            'ellipsis_count': 0, 'quotes_count': 0, 'comma_count': 0,
            'has_exclamation': 0, 'has_question': 0, 'has_ellipsis': 0,
            'punct_density': 0, 'has_negation': 0, 'negation_count': 0,
            'has_intensifier': 0, 'intensifier_count': 0,
            'has_sarcasm_marker': 0, 'sarcasm_marker_count': 0,
            'type_token_ratio': 0, 'all_caps_count': 0,
            'positive_word_count': 0, 'negative_word_count': 0,
            'sentiment_polarity': 0, 'sentiment_strength': 0,
            'starts_with_interjection': 0, 'has_rhetorical_pattern': 0,
            'has_contrast': 0
        }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def load_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load all prediction files."""
    
    all_dfs = []
    pred_files = list(predictions_dir.glob("*.csv"))
    
    print(f"   Found {len(pred_files)} prediction files")
    
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        df['source_file'] = pred_file.name
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total predictions: {len(combined)}")
    
    return combined


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract linguistic features for all texts."""
    
    extractor = LinguisticFeatureExtractor()
    
    # Get unique texts to avoid redundant computation
    unique_texts = df['text'].unique()
    print(f"   Extracting features for {len(unique_texts)} unique texts...")
    
    # Extract features for each unique text
    text_features = {}
    for text in unique_texts:
        text_features[text] = extractor.extract_features(text)
    
    # Map features back to full dataframe
    feature_rows = []
    for _, row in df.iterrows():
        features = text_features[row['text']].copy()
        features['text'] = row['text']
        features['true_label'] = row['true_label']
        features['predicted_label'] = row['predicted_label']
        features['correct'] = row['correct']
        features['source_file'] = row['source_file']
        feature_rows.append(features)
    
    return pd.DataFrame(feature_rows)


def compare_groups(
    features_df: pd.DataFrame,
    group_col: str,
    group_values: Tuple,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Compare features between two groups."""
    
    group1_name, group2_name = group_values
    group1 = features_df[features_df[group_col] == group1_name]
    group2 = features_df[features_df[group_col] == group2_name]
    
    results = []
    
    for feature in feature_cols:
        g1_values = group1[feature].dropna()
        g2_values = group2[feature].dropna()
        
        g1_mean = g1_values.mean() if len(g1_values) > 0 else 0
        g2_mean = g2_values.mean() if len(g2_values) > 0 else 0
        
        g1_std = g1_values.std() if len(g1_values) > 0 else 0
        g2_std = g2_values.std() if len(g2_values) > 0 else 0
        
        diff = g2_mean - g1_mean
        diff_pct = (diff / g1_mean * 100) if g1_mean != 0 else 0
        
        # Simple t-test (if scipy available)
        try:
            from scipy import stats
            if len(g1_values) > 1 and len(g2_values) > 1:
                t_stat, p_value = stats.ttest_ind(g1_values, g2_values)
            else:
                t_stat, p_value = 0, 1
        except ImportError:
            t_stat, p_value = 0, 1
        
        results.append({
            'feature': feature,
            f'{group1_name}_mean': round(g1_mean, 4),
            f'{group1_name}_std': round(g1_std, 4),
            f'{group2_name}_mean': round(g2_mean, 4),
            f'{group2_name}_std': round(g2_std, 4),
            'difference': round(diff, 4),
            'diff_pct': round(diff_pct, 2),
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)


def generate_report(
    features_df: pd.DataFrame,
    correct_vs_incorrect: pd.DataFrame,
    fp_vs_fn: pd.DataFrame
) -> str:
    """Generate human-readable linguistic analysis report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("LINGUISTIC FEATURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # -----------------------------------------------------------------
    # 1. OVERVIEW
    # -----------------------------------------------------------------
    lines.append("\n" + "=" * 70)
    lines.append("1. DATASET OVERVIEW")
    lines.append("=" * 70)
    
    total = len(features_df)
    correct = features_df['correct'].sum()
    incorrect = total - correct
    
    lines.append(f"\nTotal predictions analyzed: {total}")
    lines.append(f"  Correct: {correct} ({correct/total*100:.1f}%)")
    lines.append(f"  Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    
    # -----------------------------------------------------------------
    # 2. CORRECT VS INCORRECT COMPARISON
    # -----------------------------------------------------------------
    lines.append("\n" + "=" * 70)
    lines.append("2. CORRECT VS INCORRECT PREDICTIONS")
    lines.append("=" * 70)
    
    # Sort by absolute difference percentage
    cvi = correct_vs_incorrect.copy()
    cvi['abs_diff_pct'] = cvi['diff_pct'].abs()
    cvi_sorted = cvi.sort_values('abs_diff_pct', ascending=False)
    
    lines.append("\nTop features distinguishing incorrect from correct predictions:")
    lines.append("(Positive diff_pct = higher in incorrect predictions)")
    lines.append("")
    
    for _, row in cvi_sorted.head(15).iterrows():
        sig_marker = "*" if row['significant'] else ""
        direction = "↑" if row['difference'] > 0 else "↓"
        lines.append(f"  {row['feature']:30} {direction} {row['diff_pct']:+7.1f}% {sig_marker}")
        lines.append(f"      Correct: {row['correct_mean']:.3f}, Incorrect: {row['incorrect_mean']:.3f}")
    
    lines.append("\n  (* = statistically significant at p<0.05)")
    
    # Significant features
    significant = cvi[cvi['significant'] == True]
    lines.append(f"\n  Total significant features: {len(significant)} / {len(cvi)}")
    
    # -----------------------------------------------------------------
    # 3. FALSE POSITIVE VS FALSE NEGATIVE
    # -----------------------------------------------------------------
    lines.append("\n" + "=" * 70)
    lines.append("3. FALSE POSITIVE VS FALSE NEGATIVE COMPARISON")
    lines.append("=" * 70)
    
    if len(fp_vs_fn) > 0:
        # Calculate error counts
        incorrect_df = features_df[features_df['correct'] == 0]
        fp_count = ((incorrect_df['predicted_label'] == 1) & (incorrect_df['true_label'] == 0)).sum()
        fn_count = ((incorrect_df['predicted_label'] == 0) & (incorrect_df['true_label'] == 1)).sum()
        
        lines.append(f"\nFalse Positives (predicted sarcastic, actually literal): {fp_count}")
        lines.append(f"False Negatives (predicted literal, actually sarcastic): {fn_count}")
        
        fpfn = fp_vs_fn.copy()
        fpfn['abs_diff_pct'] = fpfn['diff_pct'].abs()
        fpfn_sorted = fpfn.sort_values('abs_diff_pct', ascending=False)
        
        lines.append("\nTop features distinguishing FP from FN:")
        lines.append("(Positive diff_pct = higher in False Negatives)")
        lines.append("")
        
        for _, row in fpfn_sorted.head(10).iterrows():
            sig_marker = "*" if row['significant'] else ""
            direction = "↑" if row['difference'] > 0 else "↓"
            lines.append(f"  {row['feature']:30} {direction} {row['diff_pct']:+7.1f}% {sig_marker}")
    
    # -----------------------------------------------------------------
    # 4. KEY INSIGHTS
    # -----------------------------------------------------------------
    lines.append("\n" + "=" * 70)
    lines.append("4. KEY INSIGHTS")
    lines.append("=" * 70)
    
    # Analyze patterns
    insights = []
    
    # Check sentiment
    sent_row = cvi[cvi['feature'] == 'sentiment_polarity']
    if len(sent_row) > 0:
        sent_diff = sent_row.iloc[0]['diff_pct']
        if abs(sent_diff) > 20:
            insights.append(f"- Sentiment polarity differs by {sent_diff:+.1f}% in incorrect predictions")
    
    # Check sarcasm markers
    marker_row = cvi[cvi['feature'] == 'sarcasm_marker_count']
    if len(marker_row) > 0:
        marker_diff = marker_row.iloc[0]['diff_pct']
        if marker_diff > 20:
            insights.append(f"- Incorrect predictions have {marker_diff:.1f}% more sarcasm markers")
            insights.append("  → Surface markers may be misleading the model")
    
    # Check exclamations
    excl_row = cvi[cvi['feature'] == 'exclamation_count']
    if len(excl_row) > 0:
        excl_diff = excl_row.iloc[0]['diff_pct']
        if excl_diff > 20:
            insights.append(f"- Incorrect predictions have {excl_diff:.1f}% more exclamation marks")
    
    # Check length
    len_row = cvi[cvi['feature'] == 'word_count']
    if len(len_row) > 0:
        len_diff = len_row.iloc[0]['diff_pct']
        if abs(len_diff) > 15:
            direction = "longer" if len_diff > 0 else "shorter"
            insights.append(f"- Incorrect predictions are {abs(len_diff):.1f}% {direction}")
    
    # Check intensifiers
    int_row = cvi[cvi['feature'] == 'intensifier_count']
    if len(int_row) > 0:
        int_diff = int_row.iloc[0]['diff_pct']
        if int_diff > 20:
            insights.append(f"- Incorrect predictions have {int_diff:.1f}% more intensifiers")
    
    if insights:
        for insight in insights:
            lines.append(f"\n{insight}")
    else:
        lines.append("\nNo strong distinguishing patterns found.")
    
    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 5: LINGUISTIC FEATURE ANALYSIS")
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
    
    df = load_predictions(predictions_dir)
    print()
    
    # -------------------------------------------------------------------------
    # 2. EXTRACT LINGUISTIC FEATURES
    # -------------------------------------------------------------------------
    print("2. EXTRACTING LINGUISTIC FEATURES")
    print("-" * 40)
    
    features_df = extract_all_features(df)
    print(f"   Extracted {len(features_df.columns) - 5} features")
    
    features_df.to_csv(output_dir / "linguistic_features.csv", index=False)
    print(f"   Saved: linguistic_features.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 3. COMPARE CORRECT VS INCORRECT
    # -------------------------------------------------------------------------
    print("3. COMPARING CORRECT VS INCORRECT")
    print("-" * 40)
    
    # Get feature columns (exclude metadata)
    feature_cols = [c for c in features_df.columns 
                   if c not in ['text', 'true_label', 'predicted_label', 'correct', 'source_file']]
    
    correct_vs_incorrect = compare_groups(
        features_df, 'correct', (1, 0), feature_cols
    )
    # Rename columns for clarity
    correct_vs_incorrect = correct_vs_incorrect.rename(columns={
        '1_mean': 'correct_mean', '1_std': 'correct_std',
        '0_mean': 'incorrect_mean', '0_std': 'incorrect_std'
    })
    
    significant_count = correct_vs_incorrect['significant'].sum()
    print(f"   Significant differences: {significant_count} / {len(feature_cols)} features")
    
    correct_vs_incorrect.to_csv(output_dir / "correct_vs_incorrect.csv", index=False)
    print(f"   Saved: correct_vs_incorrect.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 4. COMPARE FALSE POSITIVES VS FALSE NEGATIVES
    # -------------------------------------------------------------------------
    print("4. COMPARING FALSE POSITIVES VS FALSE NEGATIVES")
    print("-" * 40)
    
    # Filter to incorrect predictions only
    incorrect_df = features_df[features_df['correct'] == 0].copy()
    
    # Label FP vs FN
    incorrect_df['error_type'] = 'FN'  # default
    incorrect_df.loc[
        (incorrect_df['predicted_label'] == 1) & (incorrect_df['true_label'] == 0),
        'error_type'
    ] = 'FP'
    
    fp_count = (incorrect_df['error_type'] == 'FP').sum()
    fn_count = (incorrect_df['error_type'] == 'FN').sum()
    print(f"   False Positives: {fp_count}")
    print(f"   False Negatives: {fn_count}")
    
    fp_vs_fn = compare_groups(
        incorrect_df, 'error_type', ('FP', 'FN'), feature_cols
    )
    
    fp_vs_fn.to_csv(output_dir / "fp_vs_fn.csv", index=False)
    print(f"   Saved: fp_vs_fn.csv")
    print()
    
    # -------------------------------------------------------------------------
    # 5. GENERATE REPORT
    # -------------------------------------------------------------------------
    print("5. GENERATING REPORT")
    print("-" * 40)
    
    report = generate_report(features_df, correct_vs_incorrect, fp_vs_fn)
    
    report_path = output_dir / "linguistic_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"   Saved: linguistic_report.txt")
    print()
    
    # Print report to console
    print("\n" + report)
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
