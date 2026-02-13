# Sarcasm Detection in Sitcom Dialogue

Binary classification of utterances as **sarcastic** or **literal** using sitcom dialogue data (Friends, The Big Bang Theory, The Golden Girls).

**Authors:** Hasan Can Biyik

---

## Results Summary

### Test Set (67 examples)

| Model | Condition | Accuracy | F1 |
|-------|-----------|----------|-----|
| Random Baseline | — | 50.8% | — |
| XLM-R + LogReg (frozen) | Without Context | 68.7% | 0.686 |
| **XLM-RoBERTa (fine-tuned)** | **With Context** | **71.6%** | **0.707** |
| Phi3:medium (zero-shot) | With Context | 70.1% | 0.698 |

### Gold Set (28 examples)

| Model | Condition | Accuracy | F1 |
|-------|-----------|----------|-----|
| XLM-R + LogReg (frozen) | Without Context | 82.1% | 0.816 |
| DeBERTa-v3 (fine-tuned) | Without Context | 75.0% | 0.750 |
| **Phi3:medium (zero-shot)** | **With Context** | **89.3%** | **0.892** |

---

## Data

| Split | Examples | Label 0 | Label 1 |
|-------|----------|---------|---------|
| Train | 529 | 266 | 263 |
| Val | 66 | 33 | 33 |
| Test | 67 | 34 | 33 |
| Gold | 28 | 12 | 16 |

---

## Pipeline

```
1. step1_split_data.py        # Stratified train/val/test split
2. step2_preprocess.py        # Text preprocessing (12 conditions)
3. step3_run_baselines.py     # TF-IDF, XLM-R frozen baselines
4. xlmr_finetuning.py         # Fine-tune XLM-RoBERTa (with context)
5. xlmr_finetuning_no_context.py
6. deberta_finetuning.py      # Fine-tune DeBERTa-v3 (with context)
7. deberta_finetuning_no_context.py
8. llm_zeroshot_classification.py  # Gemma2, Qwen2.5, Phi3
9. synthetic_generation.py    # Intent generation experiment
```

---

## Key Findings

**1. Fine-tuned XLM-RoBERTa achieved the best test performance (71.6% accuracy).**

**2. Zero-shot Phi3 achieved the best gold performance (89.3% accuracy)** — without any task-specific training.

**3. Context effects are inconsistent.** DeBERTa performed *worse* with context on all evaluation sets. XLM-RoBERTa benefited from context on test but not gold.

**4. LLMs caught sarcasm that all other models missed.** For subtle examples like "Yeah, sure. Why not you?", only zero-shot LLMs classified correctly.

**5. Surface markers correlate with errors, not accuracy.** Examples with interjections, intensifiers, and positive sentiment were *harder* to classify — models cannot distinguish genuine enthusiasm from sarcastic exaggeration.

**6. Synthetic generation experiment reveals sarcasm bias.** When asked to choose intent, Qwen2.5 chose sarcasm 82% of the time vs. humans' 53%. Match rate when humans were literal: only 12.5%.

---

## Hardest Examples

| Models Wrong | True Label | Text |
|--------------|------------|------|
| 16/18 | Literal | "And then you clicked it again, she's dressed..." |
| 13/18 | Literal | "Don't you think looking for a new city is a bit of an overreaction?" |
| 13/18 | Literal | "Leonard, may I present, live from New Delhi, Dr. and Mrs. Koothrappali." |
| 11/18 | Sarcastic | "Yeah, terrific. The astronauts would love a guy named 'Crash.'" |

**Easiest example (18/18 correct):** "All right, let's do it." (Literal)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Max sequence length | 512 |
| Learning rate | 5e-6 |
| Batch size | 8 (effective: 16) |
| Epochs | 5 |
| Warmup steps | 50 |

---

## Requirements

```
torch
transformers
scikit-learn
pandas
numpy
sentence-transformers
requests  # for Ollama API
```

For zero-shot experiments: [Ollama](https://ollama.ai/) with `gemma2:9b`, `qwen2.5:14b`, `phi3:medium`

---

## Usage

```bash
# 1. Prepare data
python step1_split_data.py
python step2_preprocess.py

# 2. Run baselines
python step3_run_baselines.py

# 3. Fine-tune models
python xlmr_finetuning.py
python deberta_finetuning.py

# 4. Zero-shot LLM classification
python llm_zeroshot_classification.py

# 5. Synthetic generation experiment
python synthetic_generation.py
```

---

## License

MIT
