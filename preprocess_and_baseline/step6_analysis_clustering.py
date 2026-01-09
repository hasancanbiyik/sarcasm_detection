#!/usr/bin/env python3
"""
Step 6: Error Clustering Analysis
=================================
Groups misclassified examples to discover common error patterns.

This script:
1. Filters to incorrect predictions only
2. Clusters errors using TF-IDF or semantic embeddings
3. Analyzes each cluster for patterns (top words, characters, features)
4. Outputs cluster descriptions and example texts

Run this AFTER step5_analysis_linguistic.py

Input:
    - outputs/predictions/*.csv
    - outputs/analysis/linguistic_features.csv (optional, for richer analysis)

Output:
    - outputs/analysis/clustered_errors.csv
    - outputs/analysis/cluster_analysis.json
    - outputs/analysis/cluster_report.txt

Author: Hasan Can Biyik
Date: January 2026
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_DIR = "/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/predictions"
OUTPUT_DIR = "outputs/clustering_analysis"

# Clustering settings
N_CLUSTERS = 8  # Default number of clusters (will auto-tune if AUTO_TUNE=True)
AUTO_TUNE_CLUSTERS = True  # Try different k values and pick best
MIN_CLUSTERS = 4
MAX_CLUSTERS = 12

# Use semantic embeddings if available (requires sentence-transformers)
USE_SEMANTIC_EMBEDDINGS = False  # Set to True if you have sentence-transformers

RANDOM_SEED = 42

# =============================================================================
# TEXT PROCESSING
# =============================================================================

# Common stopwords to filter from top words analysis
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't", 'get', 'got', 'go', 'going',
    'know', 'like', 'would', 'could', 'one', 'also', 'even', 'well', 'back', 'way',
    'want', 'think', 'see', 'come', 'make', 'take', 'say', 'said', 'know', 'look',
    'really', 'yeah', 'yes', 'okay', 'ok', 'oh', 'hey', 'let', 'right', 'thing',
    'speaker'  # Normalized speaker tag
}

# Character names to track
CHARACTER_NAMES = [
    'sheldon', 'leonard', 'penny', 'howard', 'raj', 'amy', 'bernadette',
    'chandler', 'joey', 'monica', 'rachel', 'ross', 'phoebe',
    'dorothy', 'rose', 'blanche', 'sophia',
    'person', 'person1', 'person2', 'person3'
]


def extract_words(text: str) -> List[str]:
    """Extract lowercase words from text."""
    if pd.isna(text):
        return []
    text = re.sub(r'[A-Z]+\d*:', '', str(text))  # Remove speaker tags
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words


def get_top_words(texts: List[str], n: int = 10) -> Dict[str, int]:
    """Get top n non-stopword words from texts."""
    all_words = []
    for text in texts:
        words = extract_words(text)
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) > 2])
    
    word_counts = Counter(all_words)
    return dict(word_counts.most_common(n))


def get_character_mentions(texts: List[str]) -> Dict[str, int]:
    """Count character name mentions in texts."""
    mentions = Counter()
    for text in texts:
        text_lower = str(text).lower()
        for char in CHARACTER_NAMES:
            if char in text_lower:
                mentions[char] += 1
    return dict(mentions)


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def load_error_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Load predictions and filter to errors only."""
    
    all_dfs = []
    pred_files = list(predictions_dir.glob("*.csv"))
    
    print(f"   Found {len(pred_files)} prediction files")
    
    for pred_file in pred_files:
        df = pd.read_csv(pred_file)
        df['source_file'] = pred_file.name
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Filter to errors only
    errors = combined[combined['correct'] == 0].copy()
    
    # Deduplicate by text (same text appears in multiple conditions)
    errors_dedup = errors.drop_duplicates(subset=['text']).reset_index(drop=True)
    
    print(f"   Total predictions: {len(combined)}")
    print(f"   Total errors: {len(errors)}")
    print(f"   Unique error texts: {len(errors_dedup)}")
    
    return errors_dedup


def compute_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Compute TF-IDF vectors for texts."""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    return vectorizer.fit_transform(texts).toarray(), vectorizer


def compute_semantic_embeddings(texts: List[str]) -> np.ndarray:
    """Compute semantic embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except ImportError:
        print("   sentence-transformers not installed, falling back to TF-IDF")
        return None


def find_optimal_k(X: np.ndarray, min_k: int, max_k: int) -> int:
    """Find optimal number of clusters using silhouette score."""
    best_k = min_k
    best_score = -1
    
    print(f"   Testing k from {min_k} to {max_k}...")
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(X, labels)
            print(f"      k={k}: silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
    
    print(f"   Best k: {best_k} (silhouette={best_score:.4f})")
    return best_k


def cluster_errors(
    errors_df: pd.DataFrame, 
    n_clusters: int = N_CLUSTERS,
    auto_tune: bool = AUTO_TUNE_CLUSTERS
) -> Tuple[pd.DataFrame, Dict]:
    """Cluster error examples."""
    
    texts = errors_df['text'].tolist()
    
    # Compute embeddings
    print("\n   Computing embeddings...")
    
    if USE_SEMANTIC_EMBEDDINGS:
        X = compute_semantic_embeddings(texts)
        if X is None:
            X, vectorizer = compute_tfidf_embeddings(texts)
            embedding_type = 'tfidf'
        else:
            embedding_type = 'semantic'
            vectorizer = None
    else:
        X, vectorizer = compute_tfidf_embeddings(texts)
        embedding_type = 'tfidf'
    
    print(f"   Embedding type: {embedding_type}")
    print(f"   Embedding shape: {X.shape}")
    
    # Auto-tune number of clusters if requested
    if auto_tune and len(texts) > MAX_CLUSTERS:
        n_clusters = find_optimal_k(X, MIN_CLUSTERS, min(MAX_CLUSTERS, len(texts) // 5))
    
    # Perform clustering
    print(f"\n   Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels to dataframe
    errors_df = errors_df.copy()
    errors_df['cluster'] = cluster_labels
    
    # Compute cluster info
    cluster_info = {
        'n_clusters': n_clusters,
        'embedding_type': embedding_type,
        'silhouette_score': silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else 0
    }
    
    return errors_df, cluster_info


# =============================================================================
# CLUSTER ANALYSIS
# =============================================================================

def analyze_clusters(errors_df: pd.DataFrame) -> Dict:
    """Analyze each cluster for patterns."""
    
    cluster_analysis = {}
    
    for cluster_id in sorted(errors_df['cluster'].unique()):
        cluster_data = errors_df[errors_df['cluster'] == cluster_id]
        texts = cluster_data['text'].tolist()
        
        # Basic stats
        size = len(cluster_data)
        
        # Label distribution
        label_dist = cluster_data['true_label'].value_counts().to_dict()
        label_dist = {int(k): int(v) for k, v in label_dist.items()}
        
        # Error type distribution
        fp_count = ((cluster_data['predicted_label'] == 1) & (cluster_data['true_label'] == 0)).sum()
        fn_count = ((cluster_data['predicted_label'] == 0) & (cluster_data['true_label'] == 1)).sum()
        
        # Determine majority type
        if label_dist.get(1, 0) > label_dist.get(0, 0):
            majority_label = "sarcastic"
            error_type = "FN (missed sarcasm)"
        else:
            majority_label = "literal"
            error_type = "FP (false sarcasm)"
        
        # Top words
        top_words = get_top_words(texts, n=15)
        
        # Character mentions
        char_mentions = get_character_mentions(texts)
        
        # Text length stats
        lengths = [len(str(t).split()) for t in texts]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Punctuation patterns
        exclamation_pct = sum(1 for t in texts if '!' in str(t)) / size * 100 if size > 0 else 0
        question_pct = sum(1 for t in texts if '?' in str(t)) / size * 100 if size > 0 else 0
        ellipsis_pct = sum(1 for t in texts if '...' in str(t)) / size * 100 if size > 0 else 0
        
        # Sample examples (first 5)
        examples = texts[:5]
        
        cluster_analysis[int(cluster_id)] = {
            'size': int(size),
            'label_distribution': label_dist,
            'fp_count': int(fp_count),
            'fn_count': int(fn_count),
            'majority_label': majority_label,
            'primary_error_type': error_type,
            'top_words': top_words,
            'character_mentions': char_mentions,
            'avg_word_count': round(avg_length, 1),
            'exclamation_pct': round(exclamation_pct, 1),
            'question_pct': round(question_pct, 1),
            'ellipsis_pct': round(ellipsis_pct, 1),
            'examples': examples
        }
    
    return cluster_analysis


def generate_cluster_report(cluster_analysis: Dict, cluster_info: Dict) -> str:
    """Generate human-readable cluster report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("ERROR CLUSTERING ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overview
    lines.append("\n" + "=" * 70)
    lines.append("1. CLUSTERING OVERVIEW")
    lines.append("=" * 70)
    
    lines.append(f"\nNumber of clusters: {cluster_info['n_clusters']}")
    lines.append(f"Embedding type: {cluster_info['embedding_type']}")
    lines.append(f"Silhouette score: {cluster_info['silhouette_score']:.4f}")
    
    total_errors = sum(c['size'] for c in cluster_analysis.values())
    lines.append(f"Total unique errors clustered: {total_errors}")
    
    # Summary table
    lines.append("\n" + "-" * 70)
    lines.append("CLUSTER SUMMARY")
    lines.append("-" * 70)
    lines.append(f"{'Cluster':<10} {'Size':<8} {'Type':<20} {'Top Character':<15} {'Key Feature'}")
    lines.append("-" * 70)
    
    for cluster_id, data in sorted(cluster_analysis.items()):
        # Find top character
        chars = data['character_mentions']
        top_char = max(chars, key=chars.get) if chars else "none"
        
        # Find distinguishing feature
        features = []
        if data['exclamation_pct'] > 30:
            features.append(f"exclaim:{data['exclamation_pct']:.0f}%")
        if data['question_pct'] > 30:
            features.append(f"question:{data['question_pct']:.0f}%")
        if data['ellipsis_pct'] > 20:
            features.append(f"ellipsis:{data['ellipsis_pct']:.0f}%")
        key_feature = ", ".join(features) if features else "normal"
        
        lines.append(f"{cluster_id:<10} {data['size']:<8} {data['primary_error_type']:<20} {top_char:<15} {key_feature}")
    
    # Detailed cluster analysis
    lines.append("\n" + "=" * 70)
    lines.append("2. DETAILED CLUSTER ANALYSIS")
    lines.append("=" * 70)
    
    for cluster_id, data in sorted(cluster_analysis.items()):
        lines.append(f"\n{'='*70}")
        lines.append(f"CLUSTER {cluster_id}: {data['primary_error_type'].upper()}")
        lines.append(f"{'='*70}")
        
        lines.append(f"\nSize: {data['size']} examples")
        lines.append(f"True labels: {data['label_distribution']}")
        lines.append(f"FP: {data['fp_count']}, FN: {data['fn_count']}")
        
        lines.append(f"\nAverage word count: {data['avg_word_count']}")
        lines.append(f"Has exclamation: {data['exclamation_pct']:.1f}%")
        lines.append(f"Has question: {data['question_pct']:.1f}%")
        lines.append(f"Has ellipsis: {data['ellipsis_pct']:.1f}%")
        
        lines.append(f"\nTop words: {', '.join(list(data['top_words'].keys())[:10])}")
        
        if data['character_mentions']:
            chars = sorted(data['character_mentions'].items(), key=lambda x: -x[1])[:5]
            char_str = ', '.join([f"{c}({n})" for c, n in chars])
            lines.append(f"Character mentions: {char_str}")
        
        lines.append(f"\nExample texts:")
        for i, ex in enumerate(data['examples'][:3], 1):
            ex_short = str(ex)[:100] + "..." if len(str(ex)) > 100 else str(ex)
            lines.append(f"  {i}. {ex_short}")
    
    # Key patterns
    lines.append("\n" + "=" * 70)
    lines.append("3. KEY PATTERNS DISCOVERED")
    lines.append("=" * 70)
    
    # Find FN-heavy clusters (missed sarcasm)
    fn_clusters = [(cid, d) for cid, d in cluster_analysis.items() if d['fn_count'] > d['fp_count']]
    fp_clusters = [(cid, d) for cid, d in cluster_analysis.items() if d['fp_count'] > d['fn_count']]
    
    lines.append(f"\nClusters with mostly MISSED SARCASM (FN): {len(fn_clusters)}")
    for cid, d in fn_clusters:
        top_words = list(d['top_words'].keys())[:5]
        lines.append(f"  Cluster {cid}: {d['size']} examples, words: {', '.join(top_words)}")
    
    lines.append(f"\nClusters with mostly FALSE SARCASM (FP): {len(fp_clusters)}")
    for cid, d in fp_clusters:
        top_words = list(d['top_words'].keys())[:5]
        lines.append(f"  Cluster {cid}: {d['size']} examples, words: {', '.join(top_words)}")
    
    # Character patterns
    lines.append("\n" + "-" * 70)
    all_char_mentions = Counter()
    for d in cluster_analysis.values():
        all_char_mentions.update(d['character_mentions'])
    
    if all_char_mentions:
        lines.append("\nMost frequent characters in errors:")
        for char, count in all_char_mentions.most_common(5):
            lines.append(f"  {char}: {count} mentions")
    
    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 6: ERROR CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = Path(PREDICTIONS_DIR)
    
    # -------------------------------------------------------------------------
    # 1. LOAD ERROR PREDICTIONS
    # -------------------------------------------------------------------------
    print("1. LOADING ERROR PREDICTIONS")
    print("-" * 40)
    
    errors_df = load_error_predictions(predictions_dir)
    print()
    
    if len(errors_df) < MIN_CLUSTERS:
        print(f"   Not enough errors to cluster (need at least {MIN_CLUSTERS})")
        return
    
    # -------------------------------------------------------------------------
    # 2. CLUSTER ERRORS
    # -------------------------------------------------------------------------
    print("2. CLUSTERING ERRORS")
    print("-" * 40)
    
    errors_df, cluster_info = cluster_errors(errors_df)
    print()
    
    # Save clustered errors
    errors_df.to_csv(output_dir / "clustered_errors.csv", index=False)
    print(f"   Saved: clustered_errors.csv")
    
    # -------------------------------------------------------------------------
    # 3. ANALYZE CLUSTERS
    # -------------------------------------------------------------------------
    print("\n3. ANALYZING CLUSTERS")
    print("-" * 40)
    
    cluster_analysis = analyze_clusters(errors_df)
    
    # Print summary
    for cluster_id, data in sorted(cluster_analysis.items()):
        print(f"   Cluster {cluster_id}: {data['size']} examples, {data['primary_error_type']}")
    
    # Save cluster analysis
    with open(output_dir / "cluster_analysis.json", 'w') as f:
        json.dump(cluster_analysis, f, indent=2)
    print(f"\n   Saved: cluster_analysis.json")
    
    # -------------------------------------------------------------------------
    # 4. GENERATE REPORT
    # -------------------------------------------------------------------------
    print("\n4. GENERATING REPORT")
    print("-" * 40)
    
    report = generate_cluster_report(cluster_analysis, cluster_info)
    
    report_path = output_dir / "cluster_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"   Saved: cluster_report.txt")
    
    # Print report to console
    print("\n" + report)
    
    # -------------------------------------------------------------------------
    # DONE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
