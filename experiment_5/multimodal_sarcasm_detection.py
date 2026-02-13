"""
Multimodal Sarcasm Detection Experiment

This script uses a multimodal model to classify utterances as sarcastic or literal
based on video (frames), audio (extracted from video), and text (from CSV).

Dataset: Gold set (28 examples) from MUStARD
Videos: context.mp4 (dialogue context) and utterance.mp4 (response to classify)

Experiment conditions:
1. Text only
2. Video only (frames from utterance)
3. Audio only (extracted from utterance)
4. Text + Video
5. Text + Audio
6. Text + Video + Audio (full multimodal)
"""

import pandas as pd
import numpy as np
import os
import subprocess
import base64
import requests
import json
import time
from pathlib import Path
from datetime import datetime
import cv2
import tempfile

# Configuration
class Config:
    # Paths
    DATA_PATH = '/Users/hasancan/Desktop/sarcasm_detection/updated_samples_dataset.csv'  # Your gold CSV with 'id' column
    VIDEOS_BASE_PATH = '/Users/hasancan/Desktop/sarcasm_detection/videos'
    OUTPUT_DIR = 'outputs/multimodal_experiment'
    
    # Model settings (using Ollama with LLaVA for vision)
    OLLAMA_URL = 'http://localhost:11434/api/generate'
    VISION_MODEL = 'llava:13b'  # or 'llava:7b' for faster inference
    TEXT_MODEL = 'qwen2.5:14b'  # for text-only baseline
    
    TEMPERATURE = 0.3  # Lower for more consistent classification
    
    # Frame extraction settings
    NUM_FRAMES = 4  # Number of frames to extract from utterance video
    
    # Experiment conditions to run
    CONDITIONS = ['text_only', 'video_only', 'text_video']
    # Note: Audio-only and audio combinations require additional setup (whisper for transcription or audio models)


def extract_frames(video_path: str, num_frames: int = 4) -> list:
    """
    Extract evenly spaced frames from a video.
    Returns list of base64-encoded images.
    """
    frames = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"    Warning: No frames in {video_path}")
            return frames
        
        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to reduce payload size
                frame = cv2.resize(frame, (384, 384))
                
                # Encode to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_base64)
        
        cap.release()
        
    except Exception as e:
        print(f"    Error extracting frames: {e}")
    
    return frames


def classify_text_only(text: str, context: str) -> dict:
    """Classify using text only."""
    
    prompt = f"""You are analyzing a dialogue from a TV sitcom. Your task is to determine if the response is SARCASTIC or LITERAL.

Context (previous dialogue):
{context}

Response to classify:
{text}

Based on the text alone, is this response sarcastic or literal?

Answer with ONLY one word: SARCASTIC or LITERAL"""

    payload = {
        'model': Config.TEXT_MODEL,
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
        raw_response = result.get('response', '').strip().upper()
        
        # Parse response
        if 'SARCASTIC' in raw_response:
            prediction = 'SARCASTIC'
        elif 'LITERAL' in raw_response:
            prediction = 'LITERAL'
        else:
            prediction = 'UNKNOWN'
        
        return {
            'success': True,
            'prediction': prediction,
            'raw_response': raw_response,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'prediction': None,
            'raw_response': str(e),
            'elapsed_time': time.time() - start_time
        }


def classify_video_only(frames: list) -> dict:
    """Classify using video frames only (no text)."""
    
    if not frames:
        return {
            'success': False,
            'prediction': None,
            'raw_response': 'No frames available',
            'elapsed_time': 0
        }
    
    prompt = """You are analyzing frames from a TV sitcom scene. Look at the person's facial expressions, body language, and gestures.

Based ONLY on what you see in these video frames (not any text), determine if the person speaking appears to be:
- SARCASTIC: showing signs of irony, exaggeration, eye-rolling, smirking, or dismissive body language
- LITERAL: showing genuine, sincere expression matching their apparent message

Answer with ONLY one word: SARCASTIC or LITERAL"""

    # Use the first frame for single-image models, or combine for multi-image
    payload = {
        'model': Config.VISION_MODEL,
        'prompt': prompt,
        'images': frames[:1],  # LLaVA takes one image at a time
        'temperature': Config.TEMPERATURE,
        'stream': False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        elapsed_time = time.time() - start_time
        raw_response = result.get('response', '').strip().upper()
        
        if 'SARCASTIC' in raw_response:
            prediction = 'SARCASTIC'
        elif 'LITERAL' in raw_response:
            prediction = 'LITERAL'
        else:
            prediction = 'UNKNOWN'
        
        return {
            'success': True,
            'prediction': prediction,
            'raw_response': raw_response,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'prediction': None,
            'raw_response': str(e),
            'elapsed_time': time.time() - start_time
        }


def classify_text_video(text: str, context: str, frames: list) -> dict:
    """Classify using both text and video frames."""
    
    if not frames:
        return {
            'success': False,
            'prediction': None,
            'raw_response': 'No frames available',
            'elapsed_time': 0
        }
    
    prompt = f"""You are analyzing a scene from a TV sitcom. You have both the dialogue text AND video frames showing the speaker.

Context (previous dialogue):
{context}

Response to classify:
{text}

Look at the video frame showing the speaker. Consider:
1. The text content and context
2. The speaker's facial expression and body language
3. Any incongruity between what is said and how it's expressed

Based on BOTH the text and visual cues, is this response SARCASTIC or LITERAL?

Answer with ONLY one word: SARCASTIC or LITERAL"""

    payload = {
        'model': Config.VISION_MODEL,
        'prompt': prompt,
        'images': frames[:1],
        'temperature': Config.TEMPERATURE,
        'stream': False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        elapsed_time = time.time() - start_time
        raw_response = result.get('response', '').strip().upper()
        
        if 'SARCASTIC' in raw_response:
            prediction = 'SARCASTIC'
        elif 'LITERAL' in raw_response:
            prediction = 'LITERAL'
        else:
            prediction = 'UNKNOWN'
        
        return {
            'success': True,
            'prediction': prediction,
            'raw_response': raw_response,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'prediction': None,
            'raw_response': str(e),
            'elapsed_time': time.time() - start_time
        }


def get_video_path(video_id: str, sarcasm_label: str) -> tuple:
    """Get paths to context and utterance videos for a given ID."""
    
    # Determine folder based on label
    if sarcasm_label.lower() in ['sarcastic', 'sarcasm', '1', 'true']:
        folder = 'sarcastic'
    else:
        folder = 'nonsarcastic'
    
    base_path = os.path.join(Config.VIDEOS_BASE_PATH, folder, video_id)
    
    context_path = os.path.join(base_path, 'context.mp4')
    utterance_path = os.path.join(base_path, 'utterance.mp4')
    
    return context_path, utterance_path


def main():
    print("=" * 70)
    print("MULTIMODAL SARCASM DETECTION EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Vision model: {Config.VISION_MODEL}")
    print(f"Text model: {Config.TEXT_MODEL}")
    print(f"Conditions: {Config.CONDITIONS}")
    print()
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    print(f"Total examples: {len(df)}")
    
    # Check required columns
    required_cols = ['id', 'context_final', 'utterance_final', 'sarcasm_label']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}'")
            print(f"Available columns: {df.columns.tolist()}")
            return
    
    # Process each example
    results = []
    
    for idx, row in df.iterrows():
        video_id = str(row['id'])
        text = str(row['utterance_final'])
        context = str(row['context_final'])
        true_label = str(row['sarcasm_label'])
        true_label_binary = 1 if true_label.lower() in ['sarcastic', 'sarcasm', '1', 'true'] else 0
        
        print(f"\n[{idx + 1}/{len(df)}] Processing {video_id} (True: {true_label})")
        
        # Get video paths
        context_path, utterance_path = get_video_path(video_id, true_label)
        
        # Check if videos exist
        has_video = os.path.exists(utterance_path)
        if not has_video:
            print(f"  ⚠️  Video not found: {utterance_path}")
        
        # Extract frames if video exists
        frames = []
        if has_video:
            print(f"  Extracting frames from utterance video...")
            frames = extract_frames(utterance_path, Config.NUM_FRAMES)
            print(f"  Extracted {len(frames)} frames")
        
        # Initialize result row
        result_row = {
            'id': video_id,
            'text': text,
            'context': context,
            'true_label': true_label,
            'true_label_binary': true_label_binary,
            'has_video': has_video,
            'num_frames': len(frames)
        }
        
        # Run each condition
        for condition in Config.CONDITIONS:
            print(f"  Running {condition}...")
            
            if condition == 'text_only':
                res = classify_text_only(text, context)
            
            elif condition == 'video_only':
                if has_video and frames:
                    res = classify_video_only(frames)
                else:
                    res = {'success': False, 'prediction': None, 'raw_response': 'No video', 'elapsed_time': 0}
            
            elif condition == 'text_video':
                if has_video and frames:
                    res = classify_text_video(text, context, frames)
                else:
                    res = {'success': False, 'prediction': None, 'raw_response': 'No video', 'elapsed_time': 0}
            
            else:
                res = {'success': False, 'prediction': None, 'raw_response': 'Unknown condition', 'elapsed_time': 0}
            
            # Convert prediction to binary
            pred_binary = None
            if res['prediction'] == 'SARCASTIC':
                pred_binary = 1
            elif res['prediction'] == 'LITERAL':
                pred_binary = 0
            
            # Check if correct
            is_correct = pred_binary == true_label_binary if pred_binary is not None else None
            
            # Store results
            result_row[f'{condition}_prediction'] = res['prediction']
            result_row[f'{condition}_pred_binary'] = pred_binary
            result_row[f'{condition}_correct'] = is_correct
            result_row[f'{condition}_time'] = res['elapsed_time']
            result_row[f'{condition}_raw'] = res['raw_response'][:100] if res['raw_response'] else None
            
            if res['prediction']:
                status = "✓" if is_correct else "✗"
                print(f"    {condition}: {res['prediction']} {status} ({res['elapsed_time']:.2f}s)")
        
        results.append(result_row)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(Config.OUTPUT_DIR, 'multimodal_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for condition in Config.CONDITIONS:
        correct_col = f'{condition}_correct'
        if correct_col in results_df.columns:
            valid = results_df[correct_col].notna()
            if valid.sum() > 0:
                accuracy = results_df.loc[valid, correct_col].mean()
                correct = results_df.loc[valid, correct_col].sum()
                total = valid.sum()
                print(f"{condition}: {correct}/{total} correct ({accuracy*100:.1f}%)")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
