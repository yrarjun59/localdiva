import os
import numpy as np
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from diva_mic import MicVAD, SAMPLE_RATE

# load HF token (if needed)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# CONFIG
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BEAM_SIZE = 5

# load Whisper
print(f"[STT] loading whisper ({MODEL_SIZE})...")
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"[STT] ready on {DEVICE}\n")

# utterance counter
utterance_count = 0

# ------------------- transcribe function -------------------
def transcribe_audio(audio_np):
    """
    Convert numpy audio to text using Whisper.
    Returns the transcription string.
    """
    audio_int16 = np.int16(audio_np * 32767)
    segments, _ = whisper_model.transcribe(
        audio_int16,
        language="en",
        beam_size=BEAM_SIZE
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text
# ------------------------------------------------------------

# ------------------- callback -------------------
def on_speech_ready(audio_np, sample_rate):
    """
    Called by MicVAD whenever a complete speech segment is detected.
    """
    global utterance_count
    utterance_count += 1
    duration = len(audio_np) / sample_rate
    max_amp = np.max(np.abs(audio_np))
    
    print(f"\n{'─'*70}")
    print(f"[UTTERANCE #{utterance_count}]")
    print(f"  Duration: {duration:.2f}s | Amplitude: {max_amp:.4f}")
    
    # transcribe
    print(f"  Transcribing...", end="", flush=True)
    text = transcribe_audio(audio_np)
    print(f" ✓")
    
    if text:
        print(f"  YOU SAID: \"{text}\"")
    else:
        print("  (empty, skipping)")
# ------------------------------------------------------------

def start_real_time_stt():
    """
    Main loop: listen → transcribe → display
    """
    vad = MicVAD(on_speech_ready)
    vad.start()
    
    print("=" * 20)
    print("LIVE TRANSCRIPTION MODE")
    print("=" * 20)
    print("Speak now (Ctrl+C to stop)\n")
    
    try:
        while True:
            pass  # everything handled via callback
    except KeyboardInterrupt:
        print("\n\n[STOP] stopping...")
    finally:
        vad.stop()
        print("[STOP] mic closed\n")


if __name__ == "__main__":
    start_real_time_stt()