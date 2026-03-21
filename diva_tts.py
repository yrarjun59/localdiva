# diva_tts.py
import threading
import queue
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("[TTS] Loading neural TTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
print("[TTS] Ready")

audio_queue = queue.Queue()
_stop_flag = False
_lock = threading.Lock()

def _audio_worker():
    while True:
        try:
            item = audio_queue.get()
            if item is None:  # terminate
                break
            text, job_id = item
            if not text.strip(): continue
            # generate waveform
            wav = tts.tts(text)
            sr = tts.synthesizer.output_sample_rate
            # play and block only inside worker
            print(f"[TTS DEBUG] Speaking: {text}")
            sd.play(np.array(wav), samplerate=sr, blocking=True)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

# start worker thread once
threading.Thread(target=_audio_worker, daemon=True).start()

def speak_chunk(text: str):
    """Push text chunk to TTS queue"""
    if text.strip():
        with _lock:
            audio_queue.put((text.strip(), 0))

def stop_speaking():
    """Stop playback and clear queue"""
    global _stop_flag
    _stop_flag = True
    sd.stop()
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    _stop_flag = False
    print("[TTS] Stopped playback and cleared queue")

# ---------------- TEST ----------------
if __name__=="__main__":
    speak_chunk("Hello, this is a test of parallel TTS streaming")
    speak_chunk("It should speak token by token immediately.")
    import time; time.sleep(10)