import os
import time
import queue
import threading
import numpy as np
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from diva_mic import MicVAD, SAMPLE_RATE

# ------------------- CONFIGURATION -------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# MODEL CONFIG
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BEAM_SIZE = 10

# DEBUG FLAGS
DEBUG_AUDIO   = False
DEBUG_WHISPER = True
DEBUG_LATENCY = False

# BUFFER FOR MERGING SHORT UTTERANCES
BUFFER_DURATION = 1.5  # seconds
buffer = []

# UTTERANCE QUEUE
transcription_queue = queue.Queue()
utterance_count = 0

# ------------------- LOAD WHISPER MODEL -------------------
print(f"[STT] Loading Whisper model ({MODEL_SIZE}) on {DEVICE}, compute={COMPUTE_TYPE} ...")
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"[STT] Ready on {DEVICE}\n")

# ------------------- AUDIO → TEXT -------------------
def transcribe_audio(audio_np: np.ndarray) -> str:
    """Convert numpy audio to text using Whisper."""
    try:
        start_time = time.time()
        audio_int16 = np.int16(audio_np * 32767)

        # DC offset removal and clipping
        audio_np = audio_np - np.mean(audio_np)
        audio_np = np.clip(audio_np, -1.0, 1.0)

        # Whisper transcription (no streaming argument)
        segments, _ = whisper_model.transcribe(
            audio_int16,
            language="en",
            beam_size=BEAM_SIZE,
            temperature=[0.0, 0.2, 0.4],
            task="transcribe"
        )

        texts = []
        if DEBUG_WHISPER:
            print("\n[WHISPER DEBUG] Segments:")

        for seg in segments:
            text_piece = seg.text.strip()
            texts.append(text_piece)
            if DEBUG_WHISPER:
                print(f"  [{seg.start:.2f}s → {seg.end:.2f}s] {text_piece}")

        full_text = " ".join(texts).strip()

        if DEBUG_LATENCY:
            latency = time.time() - start_time
            print(f"[WHISPER] Latency: {latency:.2f}s")

        return full_text

    except Exception as e:
        print(f"[WHISPER ERROR]: {e}")
        return ""

# ------------------- CALLBACK FOR VAD -------------------
def on_speech_ready(audio_np: np.ndarray, sample_rate: int):
    """Triggered when MicVAD detects a full utterance."""
    global utterance_count
    utterance_count += 1

    duration = len(audio_np) / sample_rate
    max_amp  = np.max(np.abs(audio_np))
    mean_amp = np.mean(np.abs(audio_np))

    print(f"\n{'─'*70}")
    print(f"[UTTERANCE #{utterance_count}] Duration: {duration:.2f}s | Max Amp: {max_amp:.4f} | Mean Amp: {mean_amp:.4f}")

    if DEBUG_AUDIO:
        if max_amp < 0.01:
            print("  ⚠️ VERY LOW AUDIO (mic issue / too far)")
        elif max_amp > 0.9:
            print("  ⚠️ CLIPPING RISK (too loud)")

    # enqueue for transcription worker
    transcription_queue.put((utterance_count, audio_np))

# ------------------- TRANSCRIPTION WORKER -------------------
def transcription_worker():
    while True:
        utt_id, audio_np = transcription_queue.get()
        print(f"\n[WORKER] Processing UTTERANCE #{utt_id}...")
        text = transcribe_audio(audio_np)
        if text:
            print(f"[RESULT #{utt_id}]: \"{text}\"")
        else:
            print(f"[RESULT #{utt_id}]: (empty)")

# ------------------- MAIN REAL-TIME STT -------------------
def start_real_time_stt():
    vad = MicVAD(on_speech_ready)

    # start transcription worker thread
    threading.Thread(target=transcription_worker, daemon=True).start()

    vad.start()

    print("=" * 30)
    print(" LIVE TRANSCRIPTION MODE ")
    print("=" * 30)
    print("Speak now (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")
    finally:
        vad.stop()
        print("[STOP] Mic closed\n")

# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    start_real_time_stt()