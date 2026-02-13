# Multimodal Sarcasm Detection Experiment

## Purpose

This experiment tests whether adding visual information (video frames) to text improves sarcasm detection in sitcom dialogue.

## Dataset

- Source: MUStARD dataset (Multimodal Sarcasm Detection)
- Examples: 28 utterances from TV sitcoms (Friends, Big Bang Theory)
- Each example has:
  - Text transcript (context dialogue + target utterance)
  - Video files (`context.mp4` and `utterance.mp4`)
  - Ground truth label (Sarcastic or Literal)

## Experiment Conditions

Three conditions are tested:

1. **Text only**: The model sees only the dialogue transcript
2. **Video only**: The model sees only frames extracted from the utterance video (no text)
3. **Text + Video**: The model sees both the dialogue transcript and video frames

## Method

For each example:

1. Load the text (context and utterance) from the CSV
2. Extract 4 evenly-spaced frames from the `utterance.mp4` video
3. Send to the model with a prompt asking for classification
4. Record the prediction (SARCASTIC or LITERAL)
5. Compare with ground truth

## Models Used

- **Text-only condition**: Qwen2.5:14b (via Ollama)
- **Video conditions**: LLaVA:13b (via Ollama) — a vision-language model that can process both images and text

## Prompts

**Text only:**
```
You are analyzing a dialogue from a TV sitcom. Your task is to determine if the response is SARCASTIC or LITERAL.

Context (previous dialogue):
{context}

Response to classify:
{text}

Based on the text alone, is this response sarcastic or literal?

Answer with ONLY one word: SARCASTIC or LITERAL
```

**Video only:**
```
You are analyzing frames from a TV sitcom scene. Look at the person's facial expressions, body language, and gestures.

Based ONLY on what you see in these video frames (not any text), determine if the person speaking appears to be:
- SARCASTIC: showing signs of irony, exaggeration, eye-rolling, smirking, or dismissive body language
- LITERAL: showing genuine, sincere expression matching their apparent message

Answer with ONLY one word: SARCASTIC or LITERAL
```

**Text + Video:**
```
You are analyzing a scene from a TV sitcom. You have both the dialogue text AND video frames showing the speaker.

Context (previous dialogue):
{context}

Response to classify:
{text}

Look at the video frame showing the speaker. Consider:
1. The text content and context
2. The speaker's facial expression and body language
3. Any incongruity between what is said and how it's expressed

Based on BOTH the text and visual cues, is this response SARCASTIC or LITERAL?

Answer with ONLY one word: SARCASTIC or LITERAL
```

## Output

The script produces a CSV with:
- Example ID and text
- True label
- Prediction for each condition
- Whether each prediction was correct
- Processing time per condition

## File Structure

```
videos/
├── sarcastic/
│   ├── 1_6683/
│   │   ├── context.mp4
│   │   └── utterance.mp4
│   └── ...
└── nonsarcastic/
    ├── 1_533/
    │   ├── context.mp4
    │   └── utterance.mp4
    └── ...
```

## Research Question

Does multimodal input (text + video) improve sarcasm detection accuracy compared to text-only or video-only approaches?
