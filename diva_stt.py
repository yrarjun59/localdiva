# diva_stt.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import queue
import threading
import numpy as np
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from diva_mic import SAMPLE_RATE, MicVAD

# ---------------- CONFIG ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN: os.environ["HF_TOKEN"] = HF_TOKEN

MODEL_SIZE = "large-v3-turbo"

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except:
    DEVICE = "cpu"

COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BEAM_SIZE = 5  # higher = better accuracy
BEST_OF = 5    # sampling candidates, improves accuracy
VAD_FILTER = True  # use Silero VAD internally in Whisper for better chunking

DEBUG_AUDIO = False
DEBUG_WHISPER = False
DEBUG_LATENCY = True

MIN_SEGMENT_DURATION = 0.3  # ignore very short segments

# ---------------- QUEUE ----------------
transcription_queue = queue.Queue()
utterance_count = 0

# ---------------- LOAD MODEL ----------------
print(f"[STT] Loading Whisper model ({MODEL_SIZE}) on {DEVICE}, compute={COMPUTE_TYPE}...")
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[STT] Model ready\n")

# ---------------- AUDIO → TEXT ----------------
def transcribe_audio(audio_np: np.ndarray) -> str:
    """
    Transcribe numpy audio to text using Whisper.
    Returns full transcript for the given buffer.
    """
    try:
        start_time = time.time()

        # Normalize audio and remove DC offset
        audio_np = audio_np - np.mean(audio_np)
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Audio stats for debugging
        max_amp = np.max(np.abs(audio_np))
        mean_amp = np.mean(audio_np)
        rms = np.sqrt(np.mean(audio_np ** 2))
        clipping_warning = max_amp >= 0.99
        if DEBUG_AUDIO:
            print(f"[AUDIO] Max:{max_amp:.3f}, RMS:{rms:.3f}, Mean:{mean_amp:.3f}, Clipping:{clipping_warning}")

        # Transcribe - faster-whisper expects float32 audio
        segments, _ = whisper_model.transcribe(
            audio_np,
            language="en",
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            vad_filter=VAD_FILTER,
            task="transcribe"
        )

        texts = [seg.text.strip() for seg in segments if (seg.end - seg.start) >= MIN_SEGMENT_DURATION]

        full_text = " ".join(texts).strip()

        if DEBUG_LATENCY:
            print(f"[WHISPER LATENCY] {time.time()-start_time:.2f}s")

        return full_text

    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
        return ""

# ---------------- CALLBACK ----------------
def on_speech_ready(audio_np: np.ndarray, sample_rate: int):
    """Called when MicVAD detects a full utterance"""
    global utterance_count
    utterance_count += 1
    duration = len(audio_np) / sample_rate
    print(f"\n{'-'*50}\n[UTTERANCE #{utterance_count}] Duration:{duration:.2f}s")

    transcription_queue.put((utterance_count, audio_np))

# ---------------- WORKER ----------------
def transcription_worker():
    """Continuously process audio segments from the queue"""
    while True:
        utt_id, audio_np = transcription_queue.get()
        print(f"[WORKER] Processing UTTERANCE #{utt_id}...")
        text = transcribe_audio(audio_np)
        if text:
            print(f"[RESULT #{utt_id}]: {text}")
        else:
            print(f"[RESULT #{utt_id}]: (empty)")

# ---------------- REAL-TIME STT ----------------
def start_real_time_stt():
    """Start live transcription with MicVAD"""
    vad = MicVAD(on_speech_ready)
    threading.Thread(target=transcription_worker, daemon=True).start()
    vad.start()

    print("="*30)
    print(" LIVE TRANSCRIPTION MODE ")
    print("="*30)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C pressed, shutting down...")
    finally:
        vad.stop()
        print("[STOP] Done\n")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    start_real_time_stt()