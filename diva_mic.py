# diva_mic.py (compact & full debug)
import sounddevice as sd
import numpy as np
import torch
import queue, threading, time
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE
VAD_THRESHOLD = 0.9
SILENCE_MS = 600
MIN_SPEECH = 0.3
MAX_SPEECH = 15.0
DEBUG = False

class MicVAD:
    def __init__(self, on_speech_ready):
        self.on_speech_ready = on_speech_ready
        self.audio_queue = queue.Queue()
        self.running = False
        print("[VAD] Loading Silero VAD...")
        self.vad_model = load_silero_vad()
        print("[VAD] Ready.")

    def _mic_cb(self, indata, frames, *_):
        self.audio_queue.put(indata[:,0].astype(np.float32))

    def _vad_loop(self):
        buf, silence, speaking = [], 0, False
        silence_limit = int((SILENCE_MS/1000)*(SAMPLE_RATE/CHUNK_SAMPLES))
        last_dbg = time.time()

        while self.running:
            try: chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty: continue

            prob = self.vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()
            is_speech = prob >= VAD_THRESHOLD

            # Debug
            if DEBUG and time.time()-last_dbg>0.1:
                print(f"[DEBUG] VAD: {prob:.2f} | Max:{chunk.max():.3f} | Mean:{chunk.mean():.3f} | RMS:{np.sqrt(np.mean(chunk**2)):.3f} | Speaking:{speaking}")
                last_dbg = time.time()

            if is_speech:
                if not speaking:
                    print("[VAD] >>> Speech started")
                    speaking = True
                    silence = 0
                buf.append(chunk)
                silence = 0
                if len(buf)*CHUNK_DURATION >= MAX_SPEECH:
                    self._emit(buf)
                    buf, silence, speaking = [], 0, False
            else:
                if speaking:
                    buf.append(chunk)
                    silence += 1
                    if silence>=silence_limit:
                        total_sec = len(buf)*CHUNK_DURATION
                        if total_sec>=MIN_SPEECH:
                            print(f"[VAD] >>> Speech ended ({total_sec:.2f}s)")
                            self._emit(buf)
                        else:
                            print(f"[VAD] >>> Too short ({total_sec:.2f}s)")
                        buf, silence, speaking = [], 0, False

    def _emit(self, buf):
        self.on_speech_ready(np.concatenate(buf), SAMPLE_RATE)

    def start(self):
        self.running = True
        threading.Thread(target=self._vad_loop, daemon=True).start()
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                     blocksize=CHUNK_SAMPLES, callback=self._mic_cb)
        self.stream.start()
        print("[MIC] Listening...")

    def stop(self):
        self.running=False
        if hasattr(self,'stream'): self.stream.stop(); self.stream.close()
        print("[MIC] Stopped")

# Test
if __name__=="__main__":
    def cb(audio, sr):
        print(f"[CALLBACK] Duration:{len(audio)/sr:.2f}s | Max:{audio.max():.3f} | Mean:{audio.mean():.3f} | RMS:{np.sqrt(np.mean(audio**2)):.3f}")
    mic=MicVAD(cb)
    try:
        mic.start()
        while True: time.sleep(1)
    except KeyboardInterrupt:
        mic.stop()