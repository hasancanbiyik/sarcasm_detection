# LLM Zero-Shot Sarcasm Classification Results

## Experimental Setup

We evaluated three language models on binary sarcasm classification using zero-shot prompting via Ollama:

- **Gemma2:9b**
- **Qwen2.5:14b**
- **Phi3:medium**

Each model was tested under two conditions:

- **With Context:** Prompt included dialogue context and the response to classify
- **Without Context:** Prompt included only the response to classify

Models were instructed to respond with only "sarcastic" or "not sarcastic". Temperature was set to 0 for deterministic outputs. All models produced parseable responses (0 unparseable across all experiments).

## Results

### Test Set (67 examples)

| Model | Condition | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|-------|-----------|----------|-----------|--------|-------|-----|-----|-----|-----|
| Gemma2:9b | With Context | 0.507 | 0.750 | 0.515 | 0.362 | 33 | 1 | 33 | 0 |
| Gemma2:9b | Without Context | 0.537 | 0.758 | 0.544 | 0.421 | 33 | 3 | 31 | 0 |
| Qwen2.5:14b | With Context | 0.642 | 0.700 | 0.646 | 0.617 | 30 | 13 | 21 | 3 |
| Qwen2.5:14b | Without Context | 0.627 | 0.628 | 0.627 | 0.627 | 22 | 20 | 14 | 11 |
| Phi3:medium | With Context | 0.701 | 0.714 | 0.703 | 0.698 | 27 | 20 | 14 | 6 |
| Phi3:medium | Without Context | 0.672 | 0.681 | 0.670 | 0.666 | 18 | 27 | 7 | 15 |

### Gold Set (28 examples)

| Model | Condition | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN |
|-------|-----------|----------|-----------|--------|-------|-----|-----|-----|-----|
| Gemma2:9b | With Context | 0.643 | 0.808 | 0.583 | 0.524 | 16 | 2 | 10 | 0 |
| Gemma2:9b | Without Context | 0.643 | 0.808 | 0.583 | 0.524 | 16 | 2 | 10 | 0 |
| Qwen2.5:14b | With Context | 0.714 | 0.833 | 0.667 | 0.650 | 16 | 4 | 8 | 0 |
| Qwen2.5:14b | Without Context | 0.714 | 0.714 | 0.719 | 0.713 | 11 | 9 | 3 | 5 |
| Phi3:medium | With Context | 0.893 | 0.890 | 0.896 | 0.892 | 14 | 11 | 1 | 2 |
| Phi3:medium | Without Context | 0.821 | 0.853 | 0.844 | 0.821 | 11 | 12 | 0 | 5 |

## Observations

**1. Phi3:medium achieved the highest performance across both evaluation sets.**
On the test set, Phi3:medium achieved 70.1% accuracy and 0.698 F1 with context. On the gold set, it achieved 89.3% accuracy and 0.892 F1 with context.

**2. Gemma2:9b exhibited a strong bias toward predicting "sarcastic."**
On the test set with context, Gemma2:9b produced 33 true positives and 33 false positives, with only 1 true negative and 0 false negatives. This pattern persisted across conditions, indicating the model rarely predicted "not sarcastic."

**3. Context effect varied by model:**
- Phi3:medium performed better with context on both test (+0.032 F1) and gold (+0.071 F1)
- Qwen2.5:14b showed mixed results: context helped on test (-0.010 F1) but hurt on gold (-0.063 F1)
- Gemma2:9b showed minimal difference between conditions

**4. Qwen2.5:14b produced the most balanced predictions.**
On the test set without context, Qwen2.5:14b had a relatively even distribution across the confusion matrix (TP=22, TN=20, FP=14, FN=11), compared to Gemma2:9b's heavily skewed distribution.
