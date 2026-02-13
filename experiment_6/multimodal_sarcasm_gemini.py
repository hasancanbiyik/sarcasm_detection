"""
Multimodal Sarcasm Detection Experiment — Gemini API Edition
For local use (MacBook, no GPU required)

Model  : gemini-2.0-flash  (change to gemini-1.5-pro in Config if needed)
Handles: text / audio / video / text+audio / text+video / text+audio+video

Dataset: Gold set (28 examples) from MUStARD
Videos : context.mp4 (dialogue context) and utterance.mp4 (response to classify)

Install before running:
    pip install google-genai pandas

Set your API key:
    export GEMINI_API_KEY="your_key_here"
    # Get a free key at: https://aistudio.google.com/apikey

Notes on Gemini video handling:
    - Videos are uploaded via the File API and referenced by URI
    - Audio is extracted from the mp4 — Gemini reads the audio track natively
    - For video_only condition, we instruct the model to ignore audio in the prompt
      (Gemini has no server-side "strip audio" flag like Qwen's use_audio_in_video)
    - Uploaded files are deleted after the run to keep your quota clean
"""

import os
import time
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit paths here
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # ── Local paths ───────────────────────────────────────────────────────────
    DATA_PATH        = '/Users/hasancan/Desktop/sarcasm_detection/data/gold_dataset.csv'
    VIDEOS_BASE_PATH = '/Users/hasancan/Desktop/sarcasm_detection/videos'
    OUTPUT_DIR       = '/Users/hasancan/Desktop/sarcasm_detection/experiment_6/outputs'

    # ── Model ────────────────────────────────────────────────────────────────
    # gemini-2.0-flash : fast, cheap, strong multimodal — recommended
    # gemini-1.5-pro   : slower, pricier, potentially stronger reasoning
    MODEL_ID = 'gemini-2.0-flash'

    # ── Generation ───────────────────────────────────────────────────────────
    MAX_OUTPUT_TOKENS = 16      # we only need one word
    TEMPERATURE       = 0.1     # near-greedy for reproducibility

    # ── API ──────────────────────────────────────────────────────────────────
    API_KEY = os.environ.get('GEMINI_API_KEY', '')

    # ── Conditions ───────────────────────────────────────────────────────────
    CONDITIONS = [
        'text_only',        # transcript + dialogue context
        'audio_only',       # audio from utterance mp4, no text
        'video_only',       # video frames, prompted to ignore audio
        'text_audio',       # transcript + audio prosody
        'text_video',       # transcript + visual frames (prompted to ignore audio)
        'text_audio_video', # all three modalities
    ]

    # ── Rate limiting ─────────────────────────────────────────────────────────
    # Increase this if you keep hitting 429s. 10s is safe for most paid tiers.
    SLEEP_BETWEEN_CALLS = 10   # seconds between API calls
    MAX_RETRIES         = 3    # retry on 429
    RETRY_SLEEP         = 30   # seconds to wait before retrying on 429


# ─────────────────────────────────────────────────────────────────────────────
# Client — singleton
# ─────────────────────────────────────────────────────────────────────────────

_client_generate = None
_client_files    = None

def get_client() -> genai.Client:
    """Single client using v1beta — works for both generate_content and File API."""
    global _client_generate
    if _client_generate is None:
        if not Config.API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'"
            )
        _client_generate = genai.Client(
            api_key=Config.API_KEY,
            http_options=types.HttpOptions(api_version='v1beta')
        )
    return _client_generate

def get_files_client() -> genai.Client:
    """Alias for get_client() — kept for compatibility."""
    return get_client()


# ─────────────────────────────────────────────────────────────────────────────
# File upload helper — Gemini requires mp4/wav to be uploaded first
# ─────────────────────────────────────────────────────────────────────────────

def upload_file(file_path: str, mime_type: str) -> types.File | None:
    """
    Upload a local file to Gemini File API.
    Returns a File object with a URI, or None on failure.
    Files are ephemeral — they expire after 48h. We delete them manually after use.
    Note: File API requires v1beta client.
    """
    client = get_files_client()
    try:
        with open(file_path, 'rb') as f:
            uploaded = client.files.upload(
                file=f,
                config=types.UploadFileConfig(mime_type=mime_type)
            )
        # Wait until file is in ACTIVE state (usually instant for small clips)
        while uploaded.state == 'PROCESSING':
            time.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state == 'FAILED':
            print(f"    File upload failed: {uploaded.name}")
            return None
        return uploaded
    except Exception as e:
        print(f"    Upload error: {e}")
        return None


def delete_file(file_obj: types.File) -> None:
    """Delete an uploaded file to keep quota clean."""
    try:
        get_files_client().files.delete(name=file_obj.name)
    except Exception:
        pass  # Non-fatal — files expire anyway


def extract_audio(video_path: str, tmp_dir: str) -> str | None:
    """
    Extract 16kHz mono WAV from mp4 using ffmpeg.
    This is used for audio_only and text_audio conditions where we want
    to send audio without any video frames.
    """
    wav_path = os.path.join(tmp_dir, Path(video_path).stem + '_audio.wav')
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ar', '16000',   # 16kHz — standard for speech models
        '-ac', '1',       # mono
        '-vn',            # drop video stream
        wav_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
        print(f"    ffmpeg error: {result.stderr.decode()[:200]}")
        return None
    except Exception as e:
        print(f"    Audio extraction failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — kept identical to Qwen script for fair comparison
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFICATION_SUFFIX = (
    "Important: approximately half of responses are literal and half are sarcastic. "
    "Do not default to either label.\n\n"
    "Is this response LITERAL or SARCASTIC?\n"
    "Answer with ONLY one word: LITERAL or SARCASTIC"
)

def text_prompt(text: str, context: str, modality_note: str) -> str:
    return (
        "You are analyzing a spoken dialogue exchange. "
        "Classify the final response as LITERAL or SARCASTIC.\n\n"
        "Definitions:\n"
        "- LITERAL: the speaker means exactly what they say, sincerely and directly\n"
        "- SARCASTIC: the speaker means the opposite of what they say, or uses irony/mockery\n\n"
        f"Context (previous dialogue):\n{context}\n\n"
        f"Response to classify:\n{text}\n\n"
        f"{modality_note}\n\n"
        f"{CLASSIFICATION_SUFFIX}"
    )

def visual_only_prompt() -> str:
    return (
        "These are frames from a video of a dialogue scene. "
        "IGNORE the audio track entirely — base your answer ONLY on what you see visually.\n"
        "Look at the speaker's facial expressions, body language, and gestures.\n\n"
        "Important: approximately half of speakers are being sincere and half are being sarcastic. "
        "Do not default to either label.\n\n"
        "Based ONLY on what you see, does the speaker appear:\n"
        "- LITERAL: genuine, sincere expression\n"
        "- SARCASTIC: signs of irony, exaggeration, eye-rolling, smirking, dismissive body language\n\n"
        "Answer with ONLY one word: LITERAL or SARCASTIC"
    )

def audio_only_prompt() -> str:
    return (
        "Listen to this audio clip of a person speaking.\n"
        "Pay attention to tone, intonation, stress patterns, and vocal delivery.\n\n"
        "Important: approximately half of utterances are sincere and half are sarcastic. "
        "Do not default to either label.\n\n"
        "Based ONLY on how they sound (not any words), does the speaker appear:\n"
        "- LITERAL: genuine, sincere vocal delivery\n"
        "- SARCASTIC: exaggerated, flat, mocking, or ironic tone\n\n"
        "Answer with ONLY one word: LITERAL or SARCASTIC"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(contents: list) -> tuple[str, float]:
    """
    Send a contents list to Gemini and return (raw_text, elapsed_seconds).
    Retries up to MAX_RETRIES times on 429 RESOURCE_EXHAUSTED.
    """
    client = get_client()
    for attempt in range(Config.MAX_RETRIES):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=Config.MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=Config.MAX_OUTPUT_TOKENS,
                    temperature=Config.TEMPERATURE,
                )
            )
            elapsed = time.time() - start
            raw = response.text.strip() if response.text else ''
            return raw, elapsed
        except Exception as e:
            if '429' in str(e) and attempt < Config.MAX_RETRIES - 1:
                print(f"\n    [429] Rate limited, waiting {Config.RETRY_SLEEP}s before retry {attempt+2}/{Config.MAX_RETRIES}...", flush=True)
                time.sleep(Config.RETRY_SLEEP)
            else:
                raise


def parse_prediction(raw: str) -> str:
    upper = raw.upper()
    if 'SARCASTIC' in upper:
        return 'SARCASTIC'
    elif 'LITERAL' in upper:
        return 'LITERAL'
    return 'UNKNOWN'


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition classifiers
# ─────────────────────────────────────────────────────────────────────────────

def classify_text_only(text: str, context: str) -> dict:
    prompt = text_prompt(text, context, modality_note='Use the text and dialogue context only.')
    try:
        raw, elapsed = run_inference([prompt])
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        print(f"\n    [ERROR] text_only failed: {e}")
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}


def classify_audio_only(audio_path: str) -> dict:
    """
    Uploads extracted WAV, sends with audio_only prompt.
    Model hears prosody/tone, receives no text or frames.
    """
    print(f"\n    [DEBUG] uploading audio: {audio_path}", end=' ', flush=True)
    uploaded = upload_file(audio_path, 'audio/wav')
    if uploaded is None:
        return {'success': False, 'prediction': None,
                'raw_response': 'Upload failed', 'elapsed_time': 0}
    print(f"uploaded OK: {uploaded.name}", flush=True)
    try:
        contents = [uploaded, audio_only_prompt()]
        raw, elapsed = run_inference(contents)
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        print(f"\n    [ERROR] audio_only inference failed: {e}")
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}
    finally:
        delete_file(uploaded)


def classify_video_only(video_path: str) -> dict:
    """
    Uploads full mp4, but prompt explicitly tells model to ignore audio.
    This is the closest we can get to Qwen's use_audio_in_video=False.
    Important caveat: Gemini will still 'see' the audio track — the isolation
    is prompt-based, not architectural. Keep this in mind when interpreting results.
    """
    uploaded = upload_file(video_path, 'video/mp4')
    if uploaded is None:
        return {'success': False, 'prediction': None,
                'raw_response': 'Upload failed', 'elapsed_time': 0}
    try:
        contents = [uploaded, visual_only_prompt()]
        raw, elapsed = run_inference(contents)
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}
    finally:
        delete_file(uploaded)


def classify_text_audio(text: str, context: str, audio_path: str) -> dict:
    """Transcript + audio prosody. No visual frames."""
    time.sleep(2)  # small buffer after audio_only upload
    uploaded = upload_file(audio_path, 'audio/wav')
    if uploaded is None:
        return {'success': False, 'prediction': None,
                'raw_response': 'Upload failed', 'elapsed_time': 0}
    try:
        prompt = text_prompt(
            text, context,
            modality_note=(
                'You also have the audio of the response. '
                'Consider both the text meaning AND the vocal delivery '
                '(tone, intonation, stress) for your classification.'
            )
        )
        contents = [uploaded, prompt]
        raw, elapsed = run_inference(contents)
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        print(f"\n    [ERROR] text_audio failed: {e}")
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}
    finally:
        delete_file(uploaded)


def classify_text_video(text: str, context: str, video_path: str) -> dict:
    """
    Transcript + visual frames. Prompt tells model to ignore audio track.
    Same caveat as video_only — isolation is prompt-based on Gemini.
    """
    uploaded = upload_file(video_path, 'video/mp4')
    if uploaded is None:
        return {'success': False, 'prediction': None,
                'raw_response': 'Upload failed', 'elapsed_time': 0}
    try:
        prompt = text_prompt(
            text, context,
            modality_note=(
                'You also have the video showing the speaker. '
                'IGNORE the audio track — consider the text AND the speaker\'s '
                'facial expressions and body language ONLY.'
            )
        )
        contents = [uploaded, prompt]
        raw, elapsed = run_inference(contents)
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}
    finally:
        delete_file(uploaded)


def classify_text_audio_video(text: str, context: str, video_path: str) -> dict:
    """
    Full multimodal: text + audio + video.
    Single mp4 upload — Gemini processes audio and visual frames natively from it.
    This is the most architecturally clean condition for Gemini.
    """
    uploaded = upload_file(video_path, 'video/mp4')
    if uploaded is None:
        return {'success': False, 'prediction': None,
                'raw_response': 'Upload failed', 'elapsed_time': 0}
    try:
        prompt = text_prompt(
            text, context,
            modality_note=(
                'You have the full video of the response (visual frames AND audio). '
                'Consider the text meaning, the vocal delivery (tone, intonation), '
                'AND the speaker\'s facial expressions and body language together.'
            )
        )
        contents = [uploaded, prompt]
        raw, elapsed = run_inference(contents)
        return {'success': True, 'prediction': parse_prediction(raw),
                'raw_response': raw, 'elapsed_time': elapsed}
    except Exception as e:
        return {'success': False, 'prediction': None, 'raw_response': str(e), 'elapsed_time': 0}
    finally:
        delete_file(uploaded)


# ─────────────────────────────────────────────────────────────────────────────
# Path helper
# ─────────────────────────────────────────────────────────────────────────────

def get_video_paths(video_id: str, sarcasm_label: str) -> tuple:
    folder = 'sarcastic' if sarcasm_label.lower() in ['sarcastic', 'sarcasm', '1', 'true'] else 'nonsarcastic'
    base = os.path.join(Config.VIDEOS_BASE_PATH, folder, video_id)
    return os.path.join(base, 'context.mp4'), os.path.join(base, 'utterance.mp4')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MULTIMODAL SARCASM DETECTION — Gemini API (All 6 Conditions)")
    print("=" * 70)
    print(f"Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model      : {Config.MODEL_ID}")
    print(f"Conditions : {Config.CONDITIONS}")
    print()

    if not Config.API_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        print("  Run: export GEMINI_API_KEY='your_key_here'")
        return

    # ── Quick API connectivity test before the main loop ─────────────────────
    print("Testing API connection...", end=' ', flush=True)
    try:
        test_response = get_client().models.generate_content(
            model=Config.MODEL_ID,
            contents=["Say the word LITERAL and nothing else."],
            config=types.GenerateContentConfig(max_output_tokens=8, temperature=0.0)
        )
        print(f"OK (response: '{test_response.text.strip()}')\n")
    except Exception as e:
        print(f"\nERROR: API test failed — {e}")
        print("Check your API key and network connection.")
        return

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"Loading data: {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    print(f"Total examples: {len(df)}\n")

    required_cols = ['id', 'context_final', 'utterance_final', 'sarcasm_label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available: {df.columns.tolist()}")
        return

    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:

        for idx, row in df.iterrows():
            video_id   = str(row['id'])
            text       = str(row['utterance_final'])
            context    = str(row['context_final'])
            true_label = str(row['sarcasm_label'])
            true_binary = 1 if true_label.lower() in ['sarcastic', 'sarcasm', '1', 'true'] else 0

            print(f"[{idx+1:02d}/{len(df)}] {video_id}  (true: {true_label})")

            _, utterance_path = get_video_paths(video_id, true_label)
            has_video = os.path.exists(utterance_path)

            if not has_video:
                print(f"  ⚠  Video missing: {utterance_path}")

            # Extract WAV for audio-only and text_audio conditions
            audio_path = None
            if has_video:
                print(f"  Extracting audio...", end=' ', flush=True)
                audio_path = extract_audio(utterance_path, tmp_dir)
                has_audio = audio_path is not None
                print("ok" if has_audio else "FAILED")
            else:
                has_audio = False

            result_row = {
                'id': video_id,
                'text': text,
                'context': context,
                'true_label': true_label,
                'true_label_binary': true_binary,
                'has_video': has_video,
                'has_audio': has_audio,
            }

            for condition in Config.CONDITIONS:
                print(f"  → {condition:<22}", end=' ', flush=True)

                needs_video = condition in ['video_only', 'text_video', 'text_audio_video']
                needs_audio = condition in ['audio_only', 'text_audio']

                if needs_video and not has_video:
                    res = {'success': False, 'prediction': None,
                           'raw_response': 'No video', 'elapsed_time': 0}
                elif needs_audio and not has_audio:
                    res = {'success': False, 'prediction': None,
                           'raw_response': 'No audio', 'elapsed_time': 0}
                elif condition == 'text_only':
                    res = classify_text_only(text, context)
                elif condition == 'audio_only':
                    res = classify_audio_only(audio_path)
                elif condition == 'video_only':
                    res = classify_video_only(utterance_path)
                elif condition == 'text_audio':
                    res = classify_text_audio(text, context, audio_path)
                elif condition == 'text_video':
                    res = classify_text_video(text, context, utterance_path)
                elif condition == 'text_audio_video':
                    res = classify_text_audio_video(text, context, utterance_path)
                else:
                    res = {'success': False, 'prediction': None,
                           'raw_response': 'Unknown condition', 'elapsed_time': 0}

                pred_binary = (1 if res['prediction'] == 'SARCASTIC'
                               else 0 if res['prediction'] == 'LITERAL'
                               else None)
                is_correct = (pred_binary == true_binary) if pred_binary is not None else None

                result_row[f'{condition}_prediction']  = res['prediction']
                result_row[f'{condition}_pred_binary'] = pred_binary
                result_row[f'{condition}_correct']     = is_correct
                result_row[f'{condition}_time']        = round(res['elapsed_time'], 2)
                result_row[f'{condition}_raw']         = (res['raw_response'] or '')[:200]

                status = '✓' if is_correct else ('✗' if is_correct is False else '?')
                print(f"{str(res['prediction']):<12} {status}  ({res['elapsed_time']:.1f}s)")

                # Rate limit pause
                if Config.SLEEP_BETWEEN_CALLS > 0:
                    time.sleep(Config.SLEEP_BETWEEN_CALLS)

            results.append(result_row)

    # ── Save ─────────────────────────────────────────────────────────────────
    results_df  = pd.DataFrame(results)
    output_path = os.path.join(Config.OUTPUT_DIR, 'multimodal_results_gemini.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved → {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 58)

    for condition in Config.CONDITIONS:
        col   = f'{condition}_correct'
        pbcol = f'{condition}_pred_binary'
        valid = results_df[col].notna()
        if valid.sum() == 0:
            print(f"{condition:<22} {'N/A':>8}")
            continue

        sub     = results_df[valid]
        acc     = sub[col].mean()
        correct = int(sub[col].sum())
        total   = int(valid.sum())
        y_true  = sub['true_label_binary'].tolist()
        y_pred  = sub[pbcol].astype(int).tolist()

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        f1_s     = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        f1_l     = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0
        macro_f1 = (f1_s + f1_l) / 2

        print(f"{condition:<22} {correct:>8} {total:>6} {acc*100:>9.1f}% {macro_f1:>10.3f}")

    print(f"\nClass distribution in gold set:")
    print(results_df['true_label'].value_counts().to_string())
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
