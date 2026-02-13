"""
Synthetic Dialogue Generation with Sarcasm Intent

Purpose: Generate sitcom-style responses using Qwen2.5:14b where the model first
decides whether to respond sarcastically or literally, then generates accordingly.

Dataset: Gold set (28 examples)
Model: Qwen2.5:14b via Ollama
Temperature: 0.7
"""

import pandas as pd
import requests
import json
import re
import time
import os
from datetime import datetime

# Configuration
class Config:
    # Paths
    DATA_PATH = '/Users/hasancan/Desktop/sarcasm_detection/preprocess_and_baseline/outputs/data_splits/gold.csv'
    OUTPUT_DIR = 'outputs/synthetic_generation'
    
    # Ollama settings
    OLLAMA_URL = 'http://localhost:11434/api/generate'
    MODEL = 'qwen2.5:14b'
    TEMPERATURE = 0.7
    
    # Generation settings
    MAX_RESPONSE_WORDS = 10


def extract_speakers_from_context(context: str) -> set:
    """Extract all speaker names from the context."""
    pattern = r'([A-Z][A-Z0-9_]*)\s*:'
    speakers = set(re.findall(pattern, context))
    return speakers


def extract_speaker_from_text(text: str) -> str:
    """Extract the speaker name from the text column."""
    pattern = r'^([A-Z][A-Z0-9_]*)\s*:'
    match = re.match(pattern, text.strip())
    if match:
        return match.group(1)
    return None


def get_last_speaker(context: str) -> str:
    """Get the last speaker in the context."""
    pattern = r'([A-Z][A-Z0-9_]*)\s*:'
    matches = re.findall(pattern, context)
    if matches:
        return matches[-1]
    return None


def should_skip_example(context: str, text: str) -> bool:
    """
    Skip examples where the speaker in text column never appeared in context.
    This filters out 'user C' cases.
    """
    context_speakers = extract_speakers_from_context(context)
    text_speaker = extract_speaker_from_text(text)
    
    if text_speaker is None:
        return True
    
    if text_speaker not in context_speakers:
        return True
    
    return False


def generate_response_with_intent(context: str, last_speaker: str) -> dict:
    """
    Generate a sitcom-style response where the model first decides
    whether to be sarcastic or literal, then generates accordingly.
    """
    
    prompt = f"""You are a participant in a sitcom conversation. Your task is to reply to the last speaker.

STEP 1: First, decide whether your response should be SARCASTIC or LITERAL based on the conversation context. Consider what would be natural and fitting for a sitcom like Friends or Big Bang Theory.

STEP 2: Then, generate your response accordingly.

Rules for your response:
- Reply in 10 words or less
- Sound natural and conversational
- Respond directly to what {last_speaker} just said
- Do NOT include any speaker name or prefix

Conversation so far:
{context}

Provide your answer in this exact format:
INTENT: [SARCASTIC or LITERAL]
RESPONSE: [your response here]"""

    payload = {
        'model': Config.MODEL,
        'prompt': prompt,
        'temperature': Config.TEMPERATURE,
        'stream': False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        elapsed_time = time.time() - start_time
        raw_response = result.get('response', '').strip()
        
        # Parse intent
        intent = None
        intent_match = re.search(r'INTENT:\s*(SARCASTIC|LITERAL)', raw_response, re.IGNORECASE)
        if intent_match:
            intent = intent_match.group(1).upper()
        
        # Parse response
        generated_text = None
        response_match = re.search(r'RESPONSE:\s*(.+?)(?:\n|$)', raw_response, re.IGNORECASE | re.DOTALL)
        if response_match:
            generated_text = response_match.group(1).strip()
            # Clean up - remove any speaker prefix if model added one
            generated_text = re.sub(r'^[A-Z][A-Z0-9_]*\s*:\s*', '', generated_text)
            # Remove quotes if wrapped
            generated_text = generated_text.strip('"\'')
        
        return {
            'success': True,
            'intent': intent,
            'generated_text': generated_text,
            'raw_response': raw_response,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'intent': None,
            'generated_text': None,
            'raw_response': str(e),
            'elapsed_time': elapsed_time
        }


def main():
    print("="*60)
    print("Synthetic Dialogue Generation with Sarcasm Intent")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {Config.MODEL}")
    print(f"Temperature: {Config.TEMPERATURE}")
    print()
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    print(f"Total examples: {len(df)}")
    
    # Filter examples
    print("\nFiltering examples (skipping 'user C' cases)...")
    
    keep_indices = []
    skip_indices = []
    
    for idx, row in df.iterrows():
        context = str(row['context']) if pd.notna(row['context']) else ''
        text = str(row['text']) if pd.notna(row['text']) else ''
        
        if should_skip_example(context, text):
            skip_indices.append(idx)
        else:
            keep_indices.append(idx)
    
    print(f"Examples to process: {len(keep_indices)}")
    print(f"Examples skipped (new speaker): {len(skip_indices)}")
    
    # Process examples
    results = []
    
    print("\nGenerating responses with intent...")
    print("-"*60)
    
    for i, idx in enumerate(keep_indices):
        row = df.loc[idx]
        context = str(row['context'])
        original_text = str(row['text'])
        true_label = row['label']
        true_label_str = "SARCASTIC" if true_label == 1 else "LITERAL"
        
        last_speaker = get_last_speaker(context)
        text_speaker = extract_speaker_from_text(original_text)
        
        print(f"\n[{i+1}/{len(keep_indices)}] Example {idx}")
        print(f"  Last speaker in context: {last_speaker}")
        print(f"  Original response by: {text_speaker}")
        print(f"  Original label: {true_label_str}")
        
        # Generate response with intent
        gen_result = generate_response_with_intent(context, last_speaker)
        
        if gen_result['success'] and gen_result['intent']:
            match = "✓ MATCH" if (gen_result['intent'] == "SARCASTIC" and true_label == 1) or \
                                  (gen_result['intent'] == "LITERAL" and true_label == 0) else "✗ NO MATCH"
            print(f"  Model intent: {gen_result['intent']} {match}")
            print(f"  Generated: {gen_result['generated_text']}")
            print(f"  Original:  {original_text[:80]}...")
            print(f"  Time: {gen_result['elapsed_time']:.2f}s")
        else:
            print(f"  ERROR or parsing failed: {gen_result['raw_response'][:100]}")
        
        # Convert intent to numeric for comparison
        model_label = None
        if gen_result['intent'] == "SARCASTIC":
            model_label = 1
        elif gen_result['intent'] == "LITERAL":
            model_label = 0
        
        # Store result
        results.append({
            'example_id': idx,
            'context': context,
            'original_text': original_text,
            'original_speaker': text_speaker,
            'true_label': true_label,
            'true_label_str': true_label_str,
            'last_speaker_in_context': last_speaker,
            'model_intent': gen_result['intent'],
            'model_label': model_label,
            'generated_text': gen_result['generated_text'],
            'intent_matches_human': model_label == true_label if model_label is not None else None,
            'generation_success': gen_result['success'],
            'generation_time_seconds': gen_result['elapsed_time'],
            'raw_response': gen_result['raw_response']
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(Config.OUTPUT_DIR, 'synthetic_responses_gold_with_intent.csv')
    results_df.to_csv(output_path, index=False)
    
    # Calculate statistics
    valid_results = results_df[results_df['model_label'].notna()]
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total processed: {len(results)}")
    print(f"Successful generations: {results_df['generation_success'].sum()}")
    print(f"Valid intent parsed: {len(valid_results)}")
    print(f"Average generation time: {results_df['generation_time_seconds'].mean():.2f}s")
    
    if len(valid_results) > 0:
        # Intent distribution
        model_sarcastic = (valid_results['model_label'] == 1).sum()
        model_literal = (valid_results['model_label'] == 0).sum()
        human_sarcastic = (valid_results['true_label'] == 1).sum()
        human_literal = (valid_results['true_label'] == 0).sum()
        
        print(f"\n--- INTENT DISTRIBUTION ---")
        print(f"Model chose SARCASTIC: {model_sarcastic} ({model_sarcastic/len(valid_results)*100:.1f}%)")
        print(f"Model chose LITERAL:   {model_literal} ({model_literal/len(valid_results)*100:.1f}%)")
        print(f"Human was SARCASTIC:   {human_sarcastic} ({human_sarcastic/len(valid_results)*100:.1f}%)")
        print(f"Human was LITERAL:     {human_literal} ({human_literal/len(valid_results)*100:.1f}%)")
        
        # Match statistics
        matches = valid_results['intent_matches_human'].sum()
        match_rate = matches / len(valid_results) * 100
        
        print(f"\n--- MATCH STATISTICS ---")
        print(f"Intent matches human: {matches}/{len(valid_results)} ({match_rate:.1f}%)")
        
        # Breakdown by true label
        sarcastic_examples = valid_results[valid_results['true_label'] == 1]
        literal_examples = valid_results[valid_results['true_label'] == 0]
        
        if len(sarcastic_examples) > 0:
            sarc_matches = sarcastic_examples['intent_matches_human'].sum()
            print(f"When human was SARCASTIC, model matched: {sarc_matches}/{len(sarcastic_examples)} ({sarc_matches/len(sarcastic_examples)*100:.1f}%)")
        
        if len(literal_examples) > 0:
            lit_matches = literal_examples['intent_matches_human'].sum()
            print(f"When human was LITERAL, model matched:   {lit_matches}/{len(literal_examples)} ({lit_matches/len(literal_examples)*100:.1f}%)")
    
    print(f"\nSaved to: {output_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
