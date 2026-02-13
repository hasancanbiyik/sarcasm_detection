"""
Sarcasm Detection Prompts for LLM Experiments
=============================================
Based on Error Analysis of Baseline Models

Key failure patterns discovered:
1. Character bias (assuming Chandler = always sarcastic)
2. Surface marker trap ("oh", "great", "sure" ≠ sarcasm)
3. Sentiment confusion (strong emotion ≠ sarcasm)
4. Missing dry/deadpan sarcasm (no markers)
5. Context noise (other speakers' tone bleeds in)

Author: Aşko
Date: January 2026
"""

# ============================================================================
# ZERO-SHOT PROMPTS
# ============================================================================

# Baseline (no guidance)
ZERO_SHOT_BASELINE = """Classify as sarcastic (1) or literal (0).

Utterance: {text}

Classification:"""


# With context (no guidance)
ZERO_SHOT_BASELINE_CONTEXT = """Classify the TARGET utterance as sarcastic (1) or literal (0).

Context: {context}
Target: {text}

Classification:"""


# Informed (addresses all discovered biases)
ZERO_SHOT_INFORMED = """Classify as sarcastic (1) or literal (0).

Guidelines:
- Don't assume any character is always sarcastic. Judge each utterance independently.
- Words like "oh", "great", "sure", "wow" appear in BOTH sarcastic and literal speech.
- Strong sentiment (positive or negative) does NOT mean sarcasm.
- Dry sarcasm has NO markers—look for absurdity or stating the obvious.
- Sarcasm = literal meaning contradicts intended meaning.

Utterance: {text}

Classification:"""


# Informed with context
ZERO_SHOT_INFORMED_CONTEXT = """Classify the TARGET utterance as sarcastic (1) or literal (0).

Guidelines:
- Don't assume any character is always sarcastic. Judge each utterance independently.
- Words like "oh", "great", "sure", "wow" appear in BOTH sarcastic and literal speech.
- Strong sentiment does NOT mean sarcasm.
- Dry sarcasm has NO markers—look for absurdity or obvious statements.
- Focus on the TARGET only. Other speakers' tone in context is irrelevant.

Context: {context}
Target: {text}

Classification:"""


# ============================================================================
# FEW-SHOT PROMPTS (examples to be filled from your dataset)
# ============================================================================

# Few-shot baseline
FEW_SHOT_BASELINE = """Classify as sarcastic (1) or literal (0).

Example 1:
Utterance: {example1_text}
Classification: {example1_label}

Example 2:
Utterance: {example2_text}
Classification: {example2_label}

Example 3:
Utterance: {example3_text}
Classification: {example3_label}

Now classify:
Utterance: {text}

Classification:"""


# Few-shot informed (with reasoning)
FEW_SHOT_INFORMED = """Classify as sarcastic (1) or literal (0).

Guidelines:
- Don't assume any character is always sarcastic.
- Surface markers ("oh", "great") appear in both classes.
- Strong sentiment ≠ sarcasm.
- Dry sarcasm has no markers—look for absurdity.

Example 1:
Utterance: {example1_text}
Reasoning: {example1_reasoning}
Classification: {example1_label}

Example 2:
Utterance: {example2_text}
Reasoning: {example2_reasoning}
Classification: {example2_label}

Example 3:
Utterance: {example3_text}
Reasoning: {example3_reasoning}
Classification: {example3_label}

Now classify:
Utterance: {text}

Reasoning:
Classification:"""


# Few-shot contrastive (pairs showing same surface, different labels)
FEW_SHOT_CONTRASTIVE = """Classify as sarcastic (1) or literal (0).

These pairs look similar but differ. Study what matters.

Pair A - Same marker, different meaning:
"{example_sarcastic_marker}" → 1 (sarcastic: {why_sarcastic})
"{example_literal_marker}" → 0 (literal: {why_literal})

Pair B - Same character, different intent:
"{example_char_sarcastic}" → 1 (sarcastic)
"{example_char_literal}" → 0 (literal)

Now classify:
Utterance: {text}

Classification:"""


# ============================================================================
# ABLATION PROMPTS (test which guidance helps most)
# ============================================================================

ABLATION_CHARACTER_ONLY = """Classify as sarcastic (1) or literal (0).

Note: Don't assume any character is always sarcastic. Judge this utterance independently.

Utterance: {text}

Classification:"""


ABLATION_MARKERS_ONLY = """Classify as sarcastic (1) or literal (0).

Note: Words like "oh", "great", "sure", "wow" appear in BOTH sarcastic and literal speech.

Utterance: {text}

Classification:"""


ABLATION_SENTIMENT_ONLY = """Classify as sarcastic (1) or literal (0).

Note: Strong positive or negative sentiment does NOT automatically mean sarcasm.

Utterance: {text}

Classification:"""


ABLATION_DRY_SARCASM_ONLY = """Classify as sarcastic (1) or literal (0).

Note: Dry sarcasm has NO obvious markers. Look for absurdity or painfully obvious statements.

Utterance: {text}

Classification:"""


ABLATION_INCONGRUITY_ONLY = """Classify as sarcastic (1) or literal (0).

Note: Sarcasm = literal meaning contradicts intended meaning. Ask: does this make sense literally?

Utterance: {text}

Classification:"""


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    test_text = "CHANDLER: Well, y'know I'm 29. I mean who needs a savings account."
    test_context = "JOEY: Man, how did you afford this stuff?"
    
    # Format zero-shot informed prompt
    prompt = ZERO_SHOT_INFORMED.format(text=test_text)
    print("=" * 60)
    print("ZERO-SHOT INFORMED PROMPT:")
    print("=" * 60)
    print(prompt)
    print()
    
    # Format zero-shot with context
    prompt_ctx = ZERO_SHOT_INFORMED_CONTEXT.format(text=test_text, context=test_context)
    print("=" * 60)
    print("ZERO-SHOT INFORMED WITH CONTEXT:")
    print("=" * 60)
    print(prompt_ctx)
