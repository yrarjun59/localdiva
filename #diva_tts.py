# diva_tts.py - Streaming TTS with XTTS v2
import threading
import queue
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import torch
import warnings
import re

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TTS] Loading XTTS v2 on {DEVICE}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.to(DEVICE)
print("[TTS] Ready")

SAMPLE_RATE = 24000
audio_queue = queue.Queue()
_token_buffer = ""


def _audio_worker():
    while True:
        try:
            item = audio_queue.get()
            if item is None:
                break
            text, job_id = item
            if not text.strip():
                continue
            wav = tts.tts(text=text, speaker="Ana Florence", language="en")
            audio = np.array(wav, dtype=np.float32).squeeze()
            audio = np.clip(audio, -1.0, 1.0)
            sd.play(audio, samplerate=SAMPLE_RATE, blocking=True)
        except Exception as e:
            print(f"[TTS ERROR] {e}")


threading.Thread(target=_audio_worker, daemon=True).start()


def _split_sentences(text):
    return [s.strip() for s in re.findall(r'[^.!?]+[.!?]+', text) if s.strip()]


def _is_complete(sentence):
    return sentence.strip() and sentence.strip()[-1] in '.!?'


def process_tokens(chunk):
    """Process streaming tokens - queues complete sentences immediately"""
    global _token_buffer
    if not chunk:
        return
    _token_buffer += chunk
    
    while True:
        sentences = _split_sentences(_token_buffer)
        if not sentences:
            return
        
        last_complete = _is_complete(sentences[-1])
        
        for s in sentences[:-1]:
            audio_queue.put((s, 0))
        
        if last_complete:
            audio_queue.put((sentences[-1], 0))
            _token_buffer = ""
            return
        else:
            _token_buffer = sentences[-1]
            return


def flush_tokens():
    """Flush remaining buffered text"""
    global _token_buffer
    if _token_buffer.strip():
        audio_queue.put((_token_buffer.strip(), 0))
        _token_buffer = ""


def speak_chunk(text: str):
    """Push full text to TTS queue"""
    if text.strip():
        audio_queue.put((text.strip(), 0))


def stop_speaking():
    """Stop playback and clear queue"""
    sd.stop()
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    global _token_buffer
    _token_buffer = ""
    print("[TTS] Stopped")


if __name__ == "__main__":
    print("Testing streaming XTTS v2...")
    tokens = ["Hello. ", "How are ", "you? ", "This is ", "great!"]
    for t in tokens:
        print(f"Token: '{t}'")
        process_tokens(t)
    
    flush_tokens()
    import time
    time.sleep(6)
    print("Done")
