# Sarcasm Detection: Combined Experimental Results

## Overview

This report consolidates results from all sarcasm detection experiments conducted on sitcom dialogue data. The experiments include frozen baselines, fine-tuned transformer models, and zero-shot LLM classification.

## Experimental Conditions

All experiments were evaluated under two input conditions:

- **With Context:** Input includes dialogue context and the target utterance
- **Without Context:** Input includes only the target utterance

## Combined Results

### Test Set (67 examples)

| Model | Type | Condition | Accuracy | F1 |
|-------|------|-----------|----------|------|
| Random | Baseline | — | 0.508 | — |
| Majority | Baseline | — | 0.508 | — |
| TF-IDF + LogReg | Baseline (frozen) | Without Context | 0.687 | 0.686 |
| TF-IDF + SVM | Baseline (frozen) | With Context | 0.687 | 0.684 |
| XLM-R + LogReg (mean) | Baseline (frozen) | Without Context | 0.687 | 0.686 |
| XLM-R + LogReg (cls) | Baseline (frozen) | With Context | 0.687 | 0.686 |
| XLM-RoBERTa | Fine-tuned | With Context | 0.716 | 0.707 |
| XLM-RoBERTa | Fine-tuned | Without Context | 0.627 | 0.614 |
| DeBERTa-v3-base | Fine-tuned | With Context | 0.537 | 0.527 |
| DeBERTa-v3-base | Fine-tuned | Without Context | 0.597 | 0.591 |
| Gemma2:9b | Zero-shot LLM | With Context | 0.507 | 0.362 |
| Gemma2:9b | Zero-shot LLM | Without Context | 0.537 | 0.421 |
| Qwen2.5:14b | Zero-shot LLM | With Context | 0.642 | 0.617 |
| Qwen2.5:14b | Zero-shot LLM | Without Context | 0.627 | 0.627 |
| Phi3:medium | Zero-shot LLM | With Context | 0.701 | 0.698 |
| Phi3:medium | Zero-shot LLM | Without Context | 0.672 | 0.666 |

### Gold Set (28 examples)

| Model | Type | Condition | Accuracy | F1 |
|-------|------|-----------|----------|------|
| Random | Baseline | — | 0.536 | — |
| Majority | Baseline | — | 0.429 | — |
| TF-IDF + LogReg | Baseline (frozen) | Without Context | 0.750 | 0.747 |
| TF-IDF + LogReg | Baseline (frozen) | With Context | 0.643 | 0.636 |
| XLM-R + LogReg (mean) | Baseline (frozen) | Without Context | 0.821 | 0.816 |
| XLM-R + LogReg (mean) | Baseline (frozen) | With Context | 0.679 | 0.667 |
| XLM-RoBERTa | Fine-tuned | With Context | 0.607 | 0.594 |
| XLM-RoBERTa | Fine-tuned | Without Context | 0.643 | 0.641 |
| DeBERTa-v3-base | Fine-tuned | With Context | 0.500 | 0.476 |
| DeBERTa-v3-base | Fine-tuned | Without Context | 0.750 | 0.750 |
| Gemma2:9b | Zero-shot LLM | With Context | 0.643 | 0.524 |
| Gemma2:9b | Zero-shot LLM | Without Context | 0.643 | 0.524 |
| Qwen2.5:14b | Zero-shot LLM | With Context | 0.714 | 0.650 |
| Qwen2.5:14b | Zero-shot LLM | Without Context | 0.714 | 0.713 |
| Phi3:medium | Zero-shot LLM | With Context | 0.893 | 0.892 |
| Phi3:medium | Zero-shot LLM | Without Context | 0.821 | 0.821 |

## Summary by Model Type

### Test Set Performance (F1)

| Model Type | Best Model | Best Condition | F1 |
|------------|------------|----------------|------|
| Baseline (frozen) | XLM-R + LogReg (mean) | Without Context | 0.686 |
| Fine-tuned | XLM-RoBERTa | With Context | 0.707 |
| Zero-shot LLM | Phi3:medium | With Context | 0.698 |

### Gold Set Performance (F1)

| Model Type | Best Model | Best Condition | F1 |
|------------|------------|----------------|------|
| Baseline (frozen) | XLM-R + LogReg (mean) | Without Context | 0.816 |
| Fine-tuned | DeBERTa-v3-base | Without Context | 0.750 |
| Zero-shot LLM | Phi3:medium | With Context | 0.892 |

## Observations

**1. On the test set, fine-tuned XLM-RoBERTa with context achieved the highest F1 (0.707).**
This was followed by zero-shot Phi3:medium with context (0.698) and frozen baselines (0.686).

**2. On the gold set, zero-shot Phi3:medium with context achieved the highest F1 (0.892).**
This outperformed all frozen baselines and fine-tuned models on this evaluation set.

**3. Fine-tuned DeBERTa-v3-base underperformed relative to other approaches.**
On the test set, DeBERTa achieved 0.527 F1 with context, which is below frozen baselines. On the gold set without context, it achieved 0.750 F1.

**4. The effect of context varied across model types:**
- Frozen baselines: Context generally did not help or hurt performance
- Fine-tuned XLM-RoBERTa: Context helped on test (+0.093 F1) but hurt on gold (-0.047 F1)
- Fine-tuned DeBERTa-v3-base: Context hurt performance on all sets
- Phi3:medium (zero-shot): Context helped on both test (+0.032 F1) and gold (+0.071 F1)

**5. Performance rankings differed between test and gold sets.**
Models that performed well on test did not always perform well on gold, and vice versa. The gold set contains 28 examples compared to 67 in the test set.
