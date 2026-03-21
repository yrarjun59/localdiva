import sounddevice as sd
import numpy as np
import torch
import queue
import threading
from silero_vad import load_silero_vad
import time

# ── CONFIGURATION ────────────────────────────────────────────────
SAMPLE_RATE           = 16000   # Whisper expects 16kHz
CHUNK_SAMPLES         = 512     # 32ms per chunk
CHUNK_DURATION        = CHUNK_SAMPLES / SAMPLE_RATE

VAD_THRESHOLD         = 0.9     # 0.0–1.0, higher = more strict
SILENCE_DURATION_MS   = 600     # silence before utterance ends
MIN_SPEECH_DURATION   = 0.3     # ignore very short sounds
MAX_SPEECH_DURATION   = 15.0    # force cut very long speech

MIC_DEVICE            = None    # None = system default

DEBUG_MODE = True          # turn debug on/off
DEBUG_INTERVAL = 0.1       # seconds between debug prints

# ── MIC + VAD CLASS ──────────────────────────────────────────────
class MicVAD:
    """
    Continuous mic capture + Silero VAD.
    Emits complete speech segments via a callback.
    """
    def __init__(self, on_speech_ready):
        """
        on_speech_ready(audio_np, sample_rate)
        is called when a complete speech segment is detected.
        """
        self.on_speech_ready = on_speech_ready
        self.audio_queue     = queue.Queue()
        self.running         = False

        # load silero vad (CPU-only)
        print("[VAD] loading Silero VAD model...")
        self.vad_model = load_silero_vad()
        print("[VAD] ready.")

    # ── PRIVATE: MIC CALLBACK ──────────────────────────────
    def _mic_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice for each chunk of audio.
        Just enqueues audio for VAD processing.
        """
        if status:
            print(f"[MIC WARNING] {status}")

        audio_chunk = indata[:, 0].astype(np.float32)
        self.audio_queue.put(audio_chunk)

    # ── PRIVATE: VAD LOOP ───────────────────────────────────
    def _vad_loop(self):
        """
        Background thread: reads audio chunks from queue,
        detects speech boundaries using Silero VAD,
        and emits complete utterances via callback.
        """
        speech_buffer  = []
        silence_chunks = 0
        is_speaking    = False

        chunks_per_sec = SAMPLE_RATE / CHUNK_SAMPLES
        silence_limit  = int((SILENCE_DURATION_MS / 1000) * chunks_per_sec)

        import time
        last_debug_time = time.time()

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # run vad on chunk
            chunk_tensor = torch.from_numpy(chunk)
            speech_prob  = self.vad_model(chunk_tensor, SAMPLE_RATE).item()
            is_speech    = speech_prob >= VAD_THRESHOLD

            # ── DEBUG VISUALIZATION ───────────────────────
            if DEBUG_MODE:
                now = time.time()
                if now - last_debug_time >= DEBUG_INTERVAL:
                    bar = "#" * int(speech_prob * 20)
                    print(
                        f"[DEBUG] [{bar:<20}] "
                        f"prob={speech_prob:.3f} | "
                        f"speech={is_speech} | "
                        f"speaking={is_speaking}"
                    )
                    last_debug_time = now

            if is_speech:
                if not is_speaking:
                    print("\n[VAD] >>> speech started")
                    is_speaking = True
                    silence_chunks = 0

                speech_buffer.append(chunk)
                silence_chunks = 0

                # safety cutoff for very long speech
                total_sec = len(speech_buffer) * CHUNK_DURATION
                if total_sec >= MAX_SPEECH_DURATION:
                    print(f"[VAD] >>> max duration reached ({MAX_SPEECH_DURATION}s)")
                    self._emit(speech_buffer)
                    speech_buffer, silence_chunks, is_speaking = [], 0, False

            else:
                if is_speaking:
                    speech_buffer.append(chunk)
                    silence_chunks += 1

                    if silence_chunks >= silence_limit:
                        total_sec = len(speech_buffer) * CHUNK_DURATION
                        if total_sec >= MIN_SPEECH_DURATION:
                            print(f"[VAD] >>> speech ended ({total_sec:.2f}s)")
                            self._emit(speech_buffer)
                        else:
                            print(f"[VAD] >>> too short ({total_sec:.2f}s), ignored")

                        speech_buffer, silence_chunks, is_speaking = [], 0, False

    # ── PRIVATE: EMIT UTTERANCE ─────────────────────────────
    def _emit(self, buffer):
        """Concatenate buffered chunks and call callback"""
        audio_np = np.concatenate(buffer)
        self.on_speech_ready(audio_np, SAMPLE_RATE)

    # ── PUBLIC: START / STOP ───────────────────────────────
    def start(self):
        """Start mic capture and VAD processing"""
        self.running = True

        # start vad thread
        self.vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
        self.vad_thread.start()

        # start mic capture
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SAMPLES,
            device=MIC_DEVICE,
            callback=self._mic_callback
        )
        self.stream.start()
        print(f"[MIC] Listening... (device={self.stream.device})")

    def stop(self):
        """Stop everything cleanly"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("[MIC] Stopped")


def on_speech_ready(audio_np, sample_rate):
    """
    Callback triggered when a full speech segment is detected.
    """
    duration = len(audio_np) / sample_rate
    print(f"\n[CALLBACK] Speech segment received: {duration:.2f} sec")


if __name__ == "__main__":
    print("[SYSTEM] Initializing MicVAD test...")

    mic_vad = MicVAD(on_speech_ready)

    try:
        mic_vad.start()
        print("[SYSTEM] Running... Speak into the mic (Ctrl+C to stop)\n")

        # Keep main thread alive
        while True:
            import time
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping...")
        mic_vad.stop()

    except Exception as e:
        print(f"[ERROR] {e}")
        mic_vad.stop()